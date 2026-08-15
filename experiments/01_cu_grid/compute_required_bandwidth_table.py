"""G2: "discrepancy factor F reported at 50/75/90/95% CU, with confidence intervals"
(CLAUDE.md §4.1). Computes, per H, the bandwidth required to reach each CU target -- both
`measured` (from `cu_measured`, aggregated across repeats via `aggregate_repeats`) and
`analytic` (from `cu_analytic_link`, the papers' naive assumption) -- then `discrepancy_factor
= measured_required_bw / analytic_required_bw` at that CU target (`analysis/aggregate.py`,
already implemented, just never fed real multi-repeat data until now).

Log-linear interpolation between the two bracketing measured bandwidth levels (CU is
monotonically increasing in bandwidth in every H series measured so far) -- if a CU target is
never reached even at the highest tested bandwidth (5 Gbit/s) for a given H, that entry is
`null`, not a guess or an extrapolation (CLAUDE.md §33.2.6). Only 4 bandwidth levels were
tested (50m/200m/1g/5g) so this interpolation is coarse; a denser bandwidth sweep would
narrow it, and is real future work, not done here.

Confidence intervals: with 3 repeats per point (CLAUDE.md §40 Q6's "3 repeats each"), each
bandwidth level's aggregated `cu_measured` carries a real IQR (`aggregate_repeats`) -- this
script propagates that into a [low, high] bound on the required-bandwidth estimate by
interpolating at the CU target using the Q1 and Q3 curves as well as the median, which is a
simple, honest (if not statistically rigorous) way to surface repeat variance in the same
units as the headline number, not a formal confidence interval. Flagged as such in the output.
"""

from __future__ import annotations

import json
from pathlib import Path

from diloco_measured.analysis.aggregate import aggregate_repeats, discrepancy_factor
from diloco_measured.analysis.filter import apply as filter_apply
from diloco_measured.analysis.load import load_run_results

REPO_ROOT = Path(__file__).resolve().parents[2]
CU_TARGETS = (0.50, 0.75, 0.90, 0.95)


def _log_linear_interp_bandwidth(
    points: list[tuple[int, float]], target_cu: float
) -> float | None:
    """`points`: sorted [(bandwidth_bps, cu_value), ...], CU assumed monotonically
    non-decreasing in bandwidth (log-linear in bandwidth, per the analytic model's own
    functional form, methods/cu_model.md §2). Returns the interpolated bandwidth to reach
    `target_cu`, or None if `target_cu` is above every measured point (never extrapolated).
    """
    if target_cu <= points[0][1]:
        return float(points[0][0])  # already reached at (or below) the lowest tested level
    if target_cu > points[-1][1]:
        return None  # never reached within the tested range -- do not extrapolate
    for (bw0, cu0), (bw1, cu1) in zip(points, points[1:], strict=False):
        if cu0 <= target_cu <= cu1:
            if cu1 == cu0:
                return float(bw0)
            log_bw0, log_bw1 = __import__("math").log(bw0), __import__("math").log(bw1)
            frac = (target_cu - cu0) / (cu1 - cu0)
            return float(__import__("math").exp(log_bw0 + frac * (log_bw1 - log_bw0)))
    return None  # unreachable in practice given the checks above


def build_table(records: list[dict], algorithm: str) -> dict:
    """Returns {H: {cu_target: {"measured_bw_bps": ..., "measured_bw_bps_iqr": [lo, hi],
    "analytic_bw_bps": ..., "discrepancy_factor": ..., "n_repeats": ...}}}.
    """
    by_h: dict[int, dict[int, dict[str, list[float]]]] = {}
    for r in records:
        spec = r["spec"]
        if spec.get("phase") != "cu_grid" or spec["algorithm"] != algorithm:
            continue
        bw = spec.get("bandwidth_requested_bps")
        if bw is None:
            continue  # unshaped baseline has no place on a bandwidth axis
        H = spec["H"]
        by_h.setdefault(H, {}).setdefault(bw, {"measured": [], "link": []})
        by_h[H][bw]["measured"].append(r["cu"]["cu_measured"])
        by_h[H][bw]["link"].append(r["cu"]["cu_analytic_link"])

    table: dict[int, dict] = {}
    for H, bw_map in sorted(by_h.items()):
        bandwidths = sorted(bw_map)
        measured_median_points = []
        measured_q1_points = []
        measured_q3_points = []
        link_median_points = []
        n_repeats_per_bw = {}
        for bw in bandwidths:
            m = aggregate_repeats(bw_map[bw]["measured"])
            link = aggregate_repeats(bw_map[bw]["link"])
            measured_median_points.append((bw, m.median))
            measured_q1_points.append((bw, m.q1))
            measured_q3_points.append((bw, m.q3))
            link_median_points.append((bw, link.median))
            n_repeats_per_bw[bw] = m.n

        table[H] = {}
        for target in CU_TARGETS:
            measured_bw = _log_linear_interp_bandwidth(measured_median_points, target)
            # IQR bound: Q3 curve reaches a target CU at a LOWER bandwidth than the median
            # curve (better repeats), Q1 curve needs MORE bandwidth (worse repeats) -- hence
            # swapped relative to the usual [q1, q3] ordering when read as "bandwidth needed".
            measured_bw_lo = _log_linear_interp_bandwidth(measured_q3_points, target)
            measured_bw_hi = _log_linear_interp_bandwidth(measured_q1_points, target)
            analytic_bw = _log_linear_interp_bandwidth(link_median_points, target)

            entry: dict = {
                "measured_bw_bps": measured_bw,
                "measured_bw_bps_iqr": (
                    [measured_bw_lo, measured_bw_hi]
                    if measured_bw_lo is not None or measured_bw_hi is not None
                    else None
                ),
                "analytic_bw_bps": analytic_bw,
                "discrepancy_factor": None,
                "n_repeats": min(n_repeats_per_bw.values()) if n_repeats_per_bw else 0,
            }
            if measured_bw is not None and analytic_bw is not None:
                entry["discrepancy_factor"] = discrepancy_factor(measured_bw, analytic_bw)
            table[H][target] = entry

    return table


def main() -> None:
    records = load_run_results(REPO_ROOT / "results" / "raw")
    kept, report = filter_apply(records)
    print(f"loaded {len(records)}, kept {len(kept)} after filtering: {report}")

    table = build_table(kept, algorithm="diloco")

    print("\nG2 required-bandwidth table (DiLoCo, shaped grid):\n")
    cols = ("H", "target_CU", "measured_bw", "analytic_bw", "F", "n_repeats")
    print(f"{cols[0]:>5} {cols[1]:>10} {cols[2]:>14} {cols[3]:>14} {cols[4]:>8} {cols[5]:>10}")
    for H, targets in table.items():
        for target, entry in targets.items():
            mbw = f"{entry['measured_bw_bps']:.3e}" if entry["measured_bw_bps"] else "n/a"
            abw = f"{entry['analytic_bw_bps']:.3e}" if entry["analytic_bw_bps"] else "n/a"
            disc = f"{entry['discrepancy_factor']:.2f}" if entry["discrepancy_factor"] else "n/a"
            print(f"{H:>5} {target:>10.0%} {mbw:>14} {abw:>14} {disc:>8} {entry['n_repeats']:>10}")

    out_path = Path(__file__).resolve().parent / "required_bandwidth_table.json"
    # NOTE: this is a DERIVED analysis artifact, not a RunResult -- it does NOT belong in
    # results/raw/ (CLAUDE.md §14.1: that directory is immutable, append-only RunResult
    # records only; load_run_results() globs *.json there and would choke on anything else
    # shaped differently). Regenerable from results/raw/*.json at any time by re-running this
    # script; not itself a primary record.
    with open(out_path, "w") as out_f:
        json.dump(
            {str(H): {f"{t:.2f}": e for t, e in targets.items()} for H, targets in table.items()},
            out_f, indent=2,
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
