"""G4: fit the H-predictor on real data and validate it on a genuinely held-out repeat
(CLAUDE.md §4.1, FR-07). Fits on repeats 0 and 1 of the shaped CU grid
(`CLAUDE.md` ADR-035/037), validates against repeat 2 -- which was never used for fitting, a
real (if narrow -- same bandwidth/H grid, different repeat, not a different model size or
algorithm) held-out configuration per FR-07's requirement.

Run from the repository root with the project's venv:
    python experiments/05_predictor_validation/fit_and_validate_predictor.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from diloco_measured.analysis import predictor
from diloco_measured.analysis.filter import apply as filter_apply
from diloco_measured.analysis.load import load_run_results

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    records = load_run_results(REPO_ROOT / "results" / "raw")
    kept, report = filter_apply(records)
    print(f"loaded {len(records)}, kept {len(kept)}: {report}")

    fit_set = [r for r in kept if r["run_id"].endswith(("-r0", "-r1"))]
    holdout_set = [
        r for r in kept
        if r["run_id"].endswith("-r2") and r["spec"].get("phase") == "cu_grid"
    ]
    print(f"fit_set: {len(fit_set)} records (repeats 0+1), holdout_set: {len(holdout_set)} "
          f"records (repeat 2, never seen during fitting)")

    model = predictor.fit(fit_set)
    holdout_result = predictor.validate_holdout(model, holdout_set)

    print(f"\nCalibration domain: {model.calibration_domain}")
    print(f"Fitted on {len(model.training_run_ids)} runs")
    print("\nHoldout validation (target_cu=0.5):")
    print(json.dumps(holdout_result, indent=2))

    # Also demonstrate a few representative recommendations (not part of the holdout check --
    # just real, printed output of what the tool "diloco-measured plan" would eventually give).
    print("\nSample recommendations:")
    for bw in [50_000_000, 200_000_000, 1_000_000_000, 5_000_000_000]:
        rec = predictor.recommend(model, bandwidth_bps=bw, model_config="30m-realvocab")
        print(
            f"  bandwidth={bw:>12,} bps -> H={rec.recommended_h:>4}  "
            f"expected_cu={rec.expected_cu:.3f}  "
            f"expected_tokens_per_s={rec.expected_tokens_per_s:>10.1f}  "
            f"expected_bytes_per_hour={rec.expected_bytes_per_hour:.3e}  "
            f"extrapolation={rec.extrapolation_warning}"
        )

    model_dict = asdict(model)
    out_path = OUT_DIR / "fitted_predictor_model.json"
    with open(out_path, "w") as f:
        json.dump(model_dict, f, indent=2)
    print(f"\nwrote {out_path}")

    holdout_path = OUT_DIR / "holdout_validation_result.json"
    with open(holdout_path, "w") as f:
        json.dump(holdout_result, f, indent=2)
    print(f"wrote {holdout_path}")


if __name__ == "__main__":
    main()
