"""Figure 1 — THE headline figure (CLAUDE.md §10.2): measured vs. analytic compute
utilization across the H sweep, for one algorithm, at a fixed harness_version.

Presentation convention (CLAUDE.md §18, non-negotiable — "this single convention carries
the project's entire visual argument"): measured series are SOLID, analytic/simulated
series are DASHED, matching colours per H. Every figure states, in caption or metadata, the
harness_version, the number of contributing runs, and whether values are measured, analytic,
or interpolated (FR-13).

Pure presentation: this module does not filter, aggregate, or fetch data itself beyond
grouping already-loaded records for plotting — `records` must already be the output of
`analysis/filter.py::apply()` (status=="completed", single harness_version). It never opens
a socket or a CUDA context (CLAUDE.md §14.2 forbidden edge: figures -> anything that opens a
socket or a CUDA context).
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless by construction — this module must work with no display,
# no GPU, no network (FR-11: `make figures` runs on a reviewer's laptop).
import matplotlib.figure
import matplotlib.pyplot as plt

from diloco_measured.analysis.aggregate import aggregate_repeats

# CLAUDE.md §18: measured solid, analytic dashed. cu_analytic_achieved gets a distinct dotted
# style so it's visually distinguishable from cu_analytic_link even though both are "analytic."
_SERIES_STYLE: dict[str, dict[str, str | None]] = {
    "cu_measured": {"linestyle": "-", "marker": "o", "label": "measured"},
    "cu_analytic_link": {"linestyle": "--", "marker": None, "label": "analytic (link BW)"},
    "cu_analytic_achieved": {
        "linestyle": ":", "marker": None, "label": "analytic (achieved BW)",
    },
}

_H_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _group_by_h_and_bandwidth(records: list[dict]) -> dict[int, dict[int, list[dict]]]:
    """`{H: {bandwidth_bps: [records...]}}`, restricted to `phase == "cu_grid"` (this figure
    is specifically the Phase A grid surface, per CLAUDE.md §10.2 — a `convergence` or
    `compression` run at the same (algorithm, H, bandwidth) point is a DIFFERENT experiment
    with a different token/step budget and must not be silently averaged in here; it belongs
    to fig3/fig4 instead). Unshaped records (`bandwidth_requested_bps` is `None`) are also
    excluded — a CU-vs-bandwidth curve has no defined x-position for "no shaping applied."
    """
    grouped: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["spec"].get("phase") != "cu_grid":
            continue
        bw = r["spec"].get("bandwidth_requested_bps")
        if bw is None:
            continue
        grouped[r["spec"]["H"]][bw].append(r)
    return grouped


def build(
    records: list[dict],
    algorithm: str,
    harness_version: str | None = None,
) -> matplotlib.figure.Figure:
    """Build Fig 1 for one `algorithm` (there is no default — mixing algorithms on one CU-
    vs-bandwidth curve would conflate different communication patterns at the same H, which
    is exactly the kind of silent conflation this project exists to avoid).

    Any record whose `cu` observation is missing a given series value (e.g.
    `cu_analytic_achieved` is `null` — FR-04 failure condition, no NCCL curve coverage) is
    skipped for THAT series/point only, not dropped from the figure entirely.

    Raises `ValueError` if no matching, shaped-bandwidth record exists for `algorithm` —
    an empty figure would be a silent gap, not a result.
    """
    algorithm_records = [r for r in records if r["spec"]["algorithm"] == algorithm]
    grouped = _group_by_h_and_bandwidth(algorithm_records)
    if not grouped:
        raise ValueError(
            f"no completed, phase='cu_grid', shaped-bandwidth records for "
            f"algorithm={algorithm!r} — nothing to plot"
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    counted_run_ids: set[str] = set()
    for i, H in enumerate(sorted(grouped)):
        color = _H_COLORS[i % len(_H_COLORS)]
        bw_to_records = grouped[H]
        bandwidths = sorted(bw_to_records)

        for series_key, style in _SERIES_STYLE.items():
            xs: list[int] = []
            ys: list[float] = []
            for bw in bandwidths:
                recs = bw_to_records[bw]
                values = [
                    r["cu"][series_key]
                    for r in recs
                    if r.get("cu", {}).get(series_key) is not None
                ]
                if not values:
                    continue
                xs.append(bw)
                ys.append(aggregate_repeats(values).median)
                if series_key == "cu_measured":
                    counted_run_ids.update(r["run_id"] for r in recs)

            if xs:
                ax.plot(
                    xs, ys,
                    color=color, linestyle=style["linestyle"], marker=style["marker"],
                    label=f"H={H}, {style['label']}",
                )

    ax.set_xscale("log")
    ax.set_xlabel("Requested bandwidth (bps)")
    ax.set_ylabel("Compute utilization")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    version_label = harness_version if harness_version is not None else "mixed/unspecified"
    ax.set_title(
        f"Compute utilization vs. bandwidth — {algorithm}\n"
        f"harness_version={version_label} · {len(counted_run_ids)} contributing runs · "
        "solid=measured, dashed/dotted=analytic"
    )
    fig.tight_layout()
    return fig
