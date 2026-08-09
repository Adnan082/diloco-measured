"""Calibrated H-predictor: fit + held-out validation (FR-07, G4).

See schemas/run_result.v1.json and CLAUDE.md §15.2 `PredictorModel` entity. A recommendation
produced outside `calibration_domain` MUST carry `extrapolation_warning: true` — never silent
(US-05).

STATUS: [PROPOSED] scaffold — blocked on Phase A/B result corpora (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass


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
    """See CLAUDE.md §15.2 `PredictorModel` entity. Field shape is [CONFIRMED] (it's already
    specified there); the fitting procedure that populates it is [PROPOSED], blocked on
    Phase A/B data (`fit()` below).
    """

    model_id: str
    fitted_at: str
    training_run_ids: list[str]
    form: str
    params: dict
    calibration_domain: CalibrationDomain
    holdout_validation: dict | None = None


def fit(run_results: list[dict]) -> PredictorModel:
    """Fit a predictor mapping (measured bandwidth, model size, local step time) to H."""
    raise NotImplementedError("Phase 5 — requires Phase A + B corpora")


def recommend(model: PredictorModel, bandwidth_bps: float, model_config: str) -> Recommendation:
    """Evaluate the fitted model over candidate H values and recommend one.

    CONTRACT: if `bandwidth_bps` or `model_config` falls outside `model.calibration_domain`,
    extrapolation_warning MUST be True — this function must never fail silently into an
    unqualified recommendation (FR-07 failure condition, US-05).
    """
    raise NotImplementedError("Phase 5")


def validate_holdout(model: PredictorModel, holdout_run_results: list[dict]) -> dict:
    """Compare predicted vs. measured optimal H on configurations never used for fitting.

    Returns {predicted_H, measured_H, regret_pct} per methods/statistics.md §3.
    """
    raise NotImplementedError("Phase 5")
