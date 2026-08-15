"""Calibrated H-predictor: fit + held-out validation (FR-07, G4).

See schemas/run_result.v1.json and CLAUDE.md §15.2 `PredictorModel` entity. A recommendation
produced outside `calibration_domain` MUST carry `extrapolation_warning: true` — never silent
(US-05).

STATUS: `[CONFIRMED]` — fitted for real against the shaped CU-grid corpus
(`CLAUDE.md` ADR-035/037, `results/raw/cu_grid-diloco-30m-h*-bw*-r{0,1,2}.json`). Honest about
what that corpus actually is: **one model size** (30,846,720 params), **one algorithm**
(DiLoCo), 4 tested `H` values, 4 tested bandwidth levels, 3 repeats each. `calibration_domain`
reflects that narrowness directly — `model_size_range` is a single point, not a range, because
there is only one measured model size to calibrate against. A wider corpus (more model sizes,
more algorithms) would widen the domain; guessing outside it would violate CLAUDE.md §33.2.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

# The recommendation is only ever one of the H values actually measured — interpolating a
# fractional/untested H would be a config value nobody has evidence for, not just a number
# nobody has evidence for. Bandwidth, by contrast, is interpolated continuously (log-linear)
# since a practitioner's real link is essentially never exactly 50m/200m/1g/5g.
_TESTED_H_VALUES = (1, 8, 32, 128)


@dataclass(frozen=True)
class CalibrationDomain:
    bandwidth_bps_range: tuple[float, float]
    model_size_range: tuple[int, int]
    h_range: tuple[int, int]


@dataclass(frozen=True)
class Recommendation:
    recommended_h: int
    expected_tokens_per_s: float
    expected_cu: float
    expected_bytes_per_hour: float
    extrapolation_warning: bool
    calibration_domain: CalibrationDomain


@dataclass(frozen=True)
class PredictorModel:
    """See CLAUDE.md §15.2 `PredictorModel` entity."""

    model_id: str
    fitted_at: str
    training_run_ids: list[str]
    form: str
    params: dict
    calibration_domain: CalibrationDomain
    holdout_validation: dict | None = None


def _log_linear_interp(points: list[tuple[float, float]], x: float) -> tuple[float, bool]:
    """`points`: sorted [(x, y), ...], log-linear in x. Returns (y, was_extrapolated)."""
    xs = [p[0] for p in points]
    if x <= xs[0]:
        return points[0][1], x < xs[0]
    if x >= xs[-1]:
        return points[-1][1], x > xs[-1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0, False
            frac = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + frac * (y1 - y0), False
    raise AssertionError("unreachable")  # pragma: no cover


def _predicted_bytes_per_rank_per_step(n_params: int, world_size: int, H: int) -> float:
    """Ring all-reduce bytes-on-wire per rank per step (methods/wire_model.md §2), duplicated
    from `measurement/wire.py::predict()`'s formula rather than imported — analysis/ may never
    import measurement/ (CLAUDE.md §11.2 hard rule). Kept in sync by hand; both are pure
    arithmetic with the same unit-tested closed form, so drift would be caught by either
    module's own tests diverging from the shared derivation in `methods/wire_model.md`.
    """
    bytes_per_sync = 2 * (n_params * 4) * (world_size - 1) / world_size
    return bytes_per_sync / H


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def fit(run_results: list[dict]) -> PredictorModel:
    """Fit a predictor mapping (bandwidth, model_config) to a recommended `H`, from real
    `phase == "cu_grid"`, `algorithm == "diloco"`, shaped (`bandwidth_requested_bps is not
    None`) records — the only phase/algorithm this project has calibration data for so far.

    The "fit" is an honest interpolation table (median-of-repeats per (H, bandwidth) cell),
    not a closed-form regression — with a 4x4 grid and one model size, a closed-form fit would
    imply more precision than the data supports. `recommend()` interpolates within this table.
    """
    cells: dict[tuple[int, int], dict[str, list[float]]] = {}
    n_params_seen: set[int] = set()
    world_size_seen: set[int] = set()
    training_run_ids: list[str] = []

    for r in run_results:
        spec = r["spec"]
        if spec.get("phase") != "cu_grid" or spec["algorithm"] != "diloco":
            continue
        bw = spec.get("bandwidth_requested_bps")
        if bw is None:
            continue
        H = spec["H"]
        key = (H, bw)
        cells.setdefault(key, {"cu": [], "tokens_per_s": []})
        cu = r.get("cu")
        throughput = r.get("throughput")
        if cu is None or throughput is None:
            continue
        cells[key]["cu"].append(cu["cu_measured"])
        cells[key]["tokens_per_s"].append(throughput["tokens_per_s"])
        training_run_ids.append(r["run_id"])
        # model_config isn't itself a byte count -- n_params is inferred from the fingerprint-
        # adjacent spec fields we DO have (world_size) plus a fixed constant for this corpus's
        # single model (30.8M params, configs/models/30m-realvocab.toml) since ExperimentSpec
        # doesn't carry n_params directly.
        world_size_seen.add(spec["world_size"])
        n_params_seen.add(30_846_720)

    if not cells:
        raise ValueError(
            "no phase='cu_grid', algorithm='diloco', shaped records found in run_results "
            "-- cannot fit a predictor with zero calibration data"
        )

    table = {
        f"{H}:{bw}": {
            "cu_median": _median(v["cu"]),
            "tokens_per_s_median": _median(v["tokens_per_s"]) if v["tokens_per_s"] else 0.0,
            "n_repeats": len(v["cu"]),
        }
        for (H, bw), v in cells.items()
        if v["cu"]
    }

    bandwidths = sorted({bw for (_, bw) in cells})
    h_values = sorted({H for (H, _) in cells})
    (n_params,) = n_params_seen if len(n_params_seen) == 1 else (30_846_720,)
    (world_size,) = world_size_seen if len(world_size_seen) == 1 else (4,)

    domain = CalibrationDomain(
        bandwidth_bps_range=(float(min(bandwidths)), float(max(bandwidths))),
        model_size_range=(n_params, n_params),
        h_range=(min(h_values), max(h_values)),
    )

    return PredictorModel(
        model_id="diloco-30m-h-predictor-v1",
        fitted_at=datetime.now(UTC).isoformat(),
        training_run_ids=sorted(set(training_run_ids)),
        form=(
            "Median-of-repeats interpolation table over a measured (H, bandwidth) grid; "
            "log-linear interpolation in bandwidth, discrete lookup in H (only tested H "
            "values are ever recommended). bytes_per_hour is analytically derived "
            "(ring-all-reduce formula), not measured -- no real wire-byte accounting exists "
            "yet (CLAUDE.md FR-05 gap, fig5_bytes_on_wire is empty)."
        ),
        params={"table": table, "n_params": n_params, "world_size": world_size},
        calibration_domain=domain,
    )


def recommend(
    model: PredictorModel,
    bandwidth_bps: float,
    model_config: str,
    target_cu: float = 0.5,
) -> Recommendation:
    """Evaluate the fitted model over candidate H values and recommend the SMALLEST tested H
    that clears `target_cu` at `bandwidth_bps` (smaller H syncs more often, which the DiLoCo
    literature treats as generally better for optimization quality when affordable -- so the
    recommendation is "the least aggressive H you can afford," not "the H with the highest
    CU," which would trivially always be the largest tested H).

    CONTRACT: if `bandwidth_bps` or `model_config` falls outside `model.calibration_domain`,
    extrapolation_warning MUST be True — this function must never fail silently into an
    unqualified recommendation (FR-07 failure condition, US-05). Also set when `target_cu`
    isn't reached by ANY tested H at this bandwidth (the recommendation still returns the
    best-available H, but flagged, never silently).
    """
    lo, hi = model.calibration_domain.bandwidth_bps_range
    extrapolation = bandwidth_bps < lo or bandwidth_bps > hi

    # model_config isn't stored in the fitted table under its string name (see fit()'s note on
    # why) -- the only thing checkable here is whether the CALLER'S claimed config matches the
    # single model size this predictor was calibrated against. An unrecognized name is treated
    # as out-of-domain rather than silently assumed to match.
    if model_config != "30m-realvocab":
        extrapolation = True

    table = model.params["table"]
    n_params = model.params["n_params"]
    world_size = model.params["world_size"]

    best_h = None
    best_cu = 0.0
    best_tokens_per_s = 0.0
    reached_target = False

    for H in sorted(_TESTED_H_VALUES):
        points: list[tuple[float, float]] = sorted(
            (float(bw), float(cell["cu_median"]))
            for key, cell in table.items()
            if (parts := key.split(":")) and int(parts[0]) == H
            for bw in [int(parts[1])]
        )
        if not points:
            continue
        cu_at_bw, _ = _log_linear_interp(points, bandwidth_bps)

        tput_points: list[tuple[float, float]] = sorted(
            (float(bw), float(cell["tokens_per_s_median"]))
            for key, cell in table.items()
            if (parts := key.split(":")) and int(parts[0]) == H
            for bw in [int(parts[1])]
        )
        tokens_per_s_at_bw, _ = _log_linear_interp(tput_points, bandwidth_bps)

        # Track the best (highest-CU) option seen so far as a fallback for "target
        # unreachable within the tested H range at this bandwidth."
        if cu_at_bw > best_cu:
            best_cu = cu_at_bw
            best_h = H
            best_tokens_per_s = tokens_per_s_at_bw

        if cu_at_bw >= target_cu and not reached_target:
            reached_target = True
            best_h = H
            best_cu = cu_at_bw
            best_tokens_per_s = tokens_per_s_at_bw
            break  # smallest qualifying H -- stop at the first (sorted ascending)

    if best_h is None:
        raise ValueError("predictor has no calibration data for any tested H -- cannot recommend")

    if not reached_target:
        extrapolation = True  # target_cu unreachable within the tested H range -- not silent

    bytes_per_rank_per_step = _predicted_bytes_per_rank_per_step(n_params, world_size, best_h)
    # bytes/hour = bytes/step * steps/s * 3600; steps/s = tokens/s / tokens_per_step, but we
    # don't have tokens_per_step stored -- approximate via world_size (matches this corpus's
    # fixed micro_batch_size*seq_len*world_size shape) is over-precise for a warned estimate,
    # so instead derive steps/s directly from the interpolated tokens_per_s and the corpus's
    # fixed per-step token count (micro_batch_size=4, seq_len=512, matching every real run
    # this predictor was fitted on -- CLAUDE.md ADR-034/035/037).
    tokens_per_step = 4 * 512 * world_size
    steps_per_s = best_tokens_per_s / tokens_per_step if tokens_per_step > 0 else 0.0
    expected_bytes_per_hour = bytes_per_rank_per_step * steps_per_s * 3600

    return Recommendation(
        recommended_h=best_h,
        expected_tokens_per_s=best_tokens_per_s,
        expected_cu=best_cu,
        expected_bytes_per_hour=expected_bytes_per_hour,
        extrapolation_warning=extrapolation,
        calibration_domain=model.calibration_domain,
    )


def validate_holdout(
    model: PredictorModel, holdout_run_results: list[dict], target_cu: float = 0.5
) -> dict:
    """Compare, per bandwidth level present in `holdout_run_results`, the H `recommend()`
    would pick (using `model`, fitted on OTHER data) against the H that
    `recommend()`'s OWN decision rule ("smallest H whose CU clears `target_cu`") would have
    picked if applied directly to the holdout data's real measured `cu_measured` — i.e. this
    checks whether repeat-to-repeat noise flips the recommendation, using the SAME objective
    on both sides.

    An earlier version of this function compared against "the H with the highest measured
    tokens_per_s" instead — a DIFFERENT objective (throughput is maximized by the largest H
    almost by construction, since CU is monotone increasing in H in every series measured so
    far), which produced a structurally inflated "regret" at every bandwidth level whenever
    `recommend()`'s target-CU rule didn't happen to pick the single largest tested H. Caught by
    testing this function against the real corpus (5 Gbit/s showed ~21% "regret" that had
    nothing to do with prediction error, purely the objective mismatch) before it was ever
    committed — fixed here rather than left in.

    Returns {predicted_H, measured_H, regret_pct} per bandwidth level (methods/statistics.md
    §3), plus an overall summary. Intended usage (CLAUDE.md ADR-035/037): fit on repeats 0-1,
    validate against repeat 2, to check the recommendation is stable across repeat noise --
    NOT a held-out model size or algorithm (this corpus doesn't have another one yet).

    `regret_pct` is reported in THROUGHPUT terms even though the underlying decision rule is
    CU-based, so it is not guaranteed to be >= 0 the way a textbook regret is: if noise makes a
    smaller H spuriously clear `target_cu` in the holdout set, `measured_H` becomes that
    (throughput-worse) smaller H, and following the model's (throughput-better, larger-H)
    prediction instead is a real throughput win despite "disagreeing" with the noisy ground
    truth — `regret_pct` correctly comes out negative in that case, not clamped to zero.
    """
    by_bw: dict[int, list[tuple[int, float, float]]] = {}
    for r in holdout_run_results:
        spec = r["spec"]
        if spec.get("phase") != "cu_grid" or spec["algorithm"] != "diloco":
            continue
        bw = spec.get("bandwidth_requested_bps")
        if bw is None or r.get("throughput") is None or r.get("cu") is None:
            continue
        by_bw.setdefault(bw, []).append(
            (spec["H"], r["cu"]["cu_measured"], r["throughput"]["tokens_per_s"])
        )

    per_bandwidth = {}
    regrets = []
    for bw, points in sorted(by_bw.items()):
        points_by_h = sorted(points, key=lambda p: p[0])
        qualifying = [(h, cu, tps) for h, cu, tps in points_by_h if cu >= target_cu]
        if qualifying:
            measured_h, measured_cu, measured_tps = qualifying[0]  # smallest qualifying H
        else:
            # Same fallback recommend() itself uses: best (highest-CU) available.
            measured_h, measured_cu, measured_tps = max(points_by_h, key=lambda p: p[1])

        rec = recommend(model, bandwidth_bps=bw, model_config="30m-realvocab", target_cu=target_cu)
        predicted_h = rec.recommended_h
        predicted_actual = next(((cu, tps) for h, cu, tps in points_by_h if h == predicted_h), None)

        if predicted_actual is None or measured_tps <= 0:
            regret_pct = None
        else:
            _predicted_cu, predicted_tps = predicted_actual
            # Regret is reported in tokens/s terms (a concrete, interpretable "you'd have
            # trained this much slower"), even though the decision rule itself is CU-based --
            # when predicted_H == measured_H this is exactly 0, not a coincidence.
            regret_pct = (measured_tps - predicted_tps) / measured_tps * 100

        per_bandwidth[str(bw)] = {
            "predicted_H": predicted_h,
            "measured_H": measured_h,
            "regret_pct": regret_pct,
        }
        if regret_pct is not None:
            regrets.append(regret_pct)

    return {
        "per_bandwidth": per_bandwidth,
        "mean_regret_pct": (sum(regrets) / len(regrets)) if regrets else None,
        "n_bandwidth_levels": len(per_bandwidth),
    }
