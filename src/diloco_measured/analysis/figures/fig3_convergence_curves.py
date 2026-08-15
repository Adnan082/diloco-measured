"""Figure 3 — training loss vs. tokens processed, single-GPU reference vs. DiLoCo at each H
(CLAUDE.md §10.2's "Loss@budget vs H" slot; G3/FR-06). At a fixed bandwidth level, since the
loss trajectory is bandwidth-independent by construction (same seed, same H, same step count
-> the exact same sequence of optimizer updates regardless of how long each sync physically
takes to complete — CLAUDE.md ADR-037 verified this directly against real per-point wall-clock
times before trusting it as a finding rather than a bug).

Not a CU-style measured-vs-analytic comparison (no literature model predicts a loss curve
here) — every line in this figure is a real measured training curve, so the §18 solid/dashed
convention doesn't apply the way it does in fig1/fig4. Instead: the single-GPU reference is
styled distinctly (dashed, black) since it's the *thing being compared against*, not one of
the configs under test; each H gets its own solid color. A horizontal dotted line marks L*
(the reference's own final loss, ADR-021/FR-06 SS40 Q5) — visually, "did this curve cross the
line" is exactly what TTTL asks.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt

_H_COLORS = {1: "#1f77b4", 8: "#ff7f0e", 32: "#2ca02c", 128: "#d62728"}


def build(
    records: list[dict],
    algorithm: str,
    bandwidth_requested_bps: int | None,
    harness_version: str | None = None,
) -> matplotlib.figure.Figure:
    """Build Fig 3 for one `algorithm` at one fixed `bandwidth_requested_bps` (None =
    unshaped), plus the single-GPU reference (world_size=1) if present in `records` — the
    reference is bandwidth-independent (it never shapes anything) so it's included regardless
    of the `bandwidth_requested_bps` filter applied to the DiLoCo curves.

    Raises `ValueError` if no matching convergence-phase record exists.
    """
    reference = None
    by_h: dict[int, dict] = {}
    for r in records:
        spec = r["spec"]
        if spec.get("phase") != "convergence":
            continue
        if r.get("convergence") is None:
            continue
        if spec["world_size"] == 1:
            reference = r
            continue
        if spec["algorithm"] != algorithm:
            continue
        if spec.get("bandwidth_requested_bps") != bandwidth_requested_bps:
            continue
        by_h[spec["H"]] = r

    if not by_h and reference is None:
        raise ValueError(
            f"no convergence-phase records for algorithm={algorithm!r} "
            f"bandwidth_requested_bps={bandwidth_requested_bps!r} — nothing to plot"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    target_loss = None

    if reference is not None:
        curve = reference["convergence"]
        target_loss = curve["target_loss"]
        xs = [p["tokens"] for p in curve["points"]]
        ys = [p["train_loss"] for p in curve["points"]]
        ax.plot(xs, ys, color="black", linestyle="--", label="single-GPU reference")

    for H in sorted(by_h):
        curve = by_h[H]["convergence"]
        if target_loss is None:
            target_loss = curve["target_loss"]
        xs = [p["tokens"] for p in curve["points"]]
        ys = [p["train_loss"] for p in curve["points"]]
        color = _H_COLORS.get(H, None)
        ax.plot(xs, ys, color=color, linestyle="-", marker="o", markersize=3, label=f"H={H}")

    if target_loss is not None:
        ax.axhline(target_loss, color="gray", linestyle=":", linewidth=1, label="L* (target)")

    ax.set_xlabel("Tokens processed")
    ax.set_ylabel("Training loss")
    ax.legend(fontsize=8, loc="upper right")

    version_label = harness_version if harness_version is not None else "mixed/unspecified"
    bw_label = "unshaped" if bandwidth_requested_bps is None else f"{bandwidth_requested_bps} bps"
    n_runs = len(by_h) + (1 if reference is not None else 0)
    ax.set_title(
        f"Training loss vs. tokens — {algorithm}, bandwidth={bw_label}\n"
        f"harness_version={version_label} · {n_runs} contributing runs\n"
        "dashed=single-GPU reference, dotted=L* (target loss)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig
