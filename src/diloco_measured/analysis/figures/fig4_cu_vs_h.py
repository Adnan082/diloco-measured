"""Figure 4 — compute utilization vs. synchronization interval H, at one fixed bandwidth
level (CLAUDE.md §10.2 names this slot "Loss@budget vs H + throughput vs H"; this module
covers the CU-vs-H half — the complementary plot to fig1_cu_surface's CU-vs-bandwidth, at a
fixed H, for the cases where the corpus has multiple H values at a SINGLE bandwidth level
rather than multiple bandwidth levels — exactly what an unshaped-baseline H-sweep produces).

Presentation convention (CLAUDE.md §18, same as fig1): measured series SOLID, analytic series
DASHED/DOTTED. Pure presentation — `records` must already be filtered (analysis/filter.py).
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt

_SERIES_STYLE: dict[str, dict[str, str | None]] = {
    "cu_measured": {"linestyle": "-", "marker": "o", "label": "measured"},
    "cu_analytic_link": {"linestyle": "--", "marker": None, "label": "analytic (link BW)"},
    "cu_analytic_achieved": {
        "linestyle": ":", "marker": None, "label": "analytic (achieved BW)",
    },
}


def build(
    records: list[dict],
    algorithm: str,
    bandwidth_requested_bps: int | None,
    harness_version: str | None = None,
) -> matplotlib.figure.Figure:
    """Build Fig 4 for one `algorithm` at one fixed `bandwidth_requested_bps` (None =
    unshaped) — the complementary grouping to fig1_cu_surface.py's per-H, per-bandwidth
    curves. No default for either — silently mixing algorithms or bandwidth levels on one
    H-axis would conflate different experiments, exactly the failure mode this project's
    naming/grouping conventions exist to prevent.

    Raises `ValueError` if no matching, completed, `phase == "cu_grid"` record exists.
    """
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        spec = r["spec"]
        if spec.get("phase") != "cu_grid":
            continue
        if spec["algorithm"] != algorithm:
            continue
        if spec.get("bandwidth_requested_bps") != bandwidth_requested_bps:
            continue
        grouped[spec["H"]].append(r)

    if not grouped:
        raise ValueError(
            f"no completed, phase='cu_grid' records for algorithm={algorithm!r} "
            f"bandwidth_requested_bps={bandwidth_requested_bps!r} — nothing to plot"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    h_values = sorted(grouped)
    counted_run_ids: set[str] = set()

    for series_key, style in _SERIES_STYLE.items():
        xs: list[int] = []
        ys: list[float] = []
        for H in h_values:
            recs = grouped[H]
            values = [
                r["cu"][series_key] for r in recs if r.get("cu", {}).get(series_key) is not None
            ]
            if not values:
                continue
            xs.append(H)
            ys.append(sum(values) / len(values))
            if series_key == "cu_measured":
                counted_run_ids.update(r["run_id"] for r in recs)
        if xs:
            ax.plot(
                xs, ys, color="#1f77b4",
                linestyle=style["linestyle"], marker=style["marker"], label=style["label"],
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Synchronization interval H (inner steps per outer sync)")
    ax.set_ylabel("Compute utilization")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")

    version_label = harness_version if harness_version is not None else "mixed/unspecified"
    bw_label = "unshaped" if bandwidth_requested_bps is None else f"{bandwidth_requested_bps} bps"
    ax.set_title(
        f"Compute utilization vs. H — {algorithm}, bandwidth={bw_label}\n"
        f"harness_version={version_label} · {len(counted_run_ids)} contributing runs\n"
        "solid=measured, dashed/dotted=analytic",
        fontsize=10,
    )
    fig.tight_layout()
    return fig
