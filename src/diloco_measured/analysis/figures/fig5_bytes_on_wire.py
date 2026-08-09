"""Figure 5 (CLAUDE.md §10.2): bytes-on-wire per training token, predicted vs. measured,
across the H sweep, for one algorithm at a fixed harness_version.

Unlike fig1 (CU vs. bandwidth), the x-axis here is H, not bandwidth: per methods/wire_model.md
§2, communication volume per training token is `O(N / H)` and, critically, does NOT depend on
bandwidth — bandwidth affects how long a sync takes, not how many bytes it moves. So this
figure deliberately POOLS records across every bandwidth level at a given H (more repeats,
more statistical power), which is the opposite grouping choice from fig1 and is the point,
not an oversight — see `_group_by_h()`.

Presentation convention (CLAUDE.md §18): predicted (the analytic/first-principles ring
all-reduce prediction) is DASHED; measured (the `/proc/net/dev` ground truth) is SOLID. When
the two diverge, the gap is TCP/IP headers, NCCL protocol overhead, and retransmits
(FR-05 design note) — quantifying that gap is itself a result, which is exactly why both
series are drawn rather than just the ratio.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless by construction, matching fig1_cu_surface.py (FR-11).
import matplotlib.figure
import matplotlib.pyplot as plt

from diloco_measured.analysis.aggregate import aggregate_repeats

_SERIES_STYLE: dict[str, dict[str, str | None]] = {
    "bytes_per_training_token_measured": {
        "linestyle": "-", "marker": "o", "label": "measured",
    },
    "bytes_per_training_token_predicted": {
        "linestyle": "--", "marker": None, "label": "predicted",
    },
}
_COLOR_MEASURED = "#1f77b4"
_COLOR_PREDICTED = "#d62728"
_SERIES_COLOR = {
    "bytes_per_training_token_measured": _COLOR_MEASURED,
    "bytes_per_training_token_predicted": _COLOR_PREDICTED,
}


def _group_by_h(records: list[dict]) -> dict[int, list[dict]]:
    """`{H: [records...]}`, restricted to `phase == "cu_grid"`, POOLED across every bandwidth
    level at that `H` (see module docstring for why that pooling is intentional here, unlike
    fig1's per-bandwidth grouping).
    """
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        if r["spec"].get("phase") != "cu_grid":
            continue
        grouped[r["spec"]["H"]].append(r)
    return grouped


def build(
    records: list[dict],
    algorithm: str,
    harness_version: str | None = None,
) -> matplotlib.figure.Figure:
    """Build Fig 5 for one `algorithm` (required, no default — see fig1_cu_surface.py's
    docstring for why mixing algorithms on one curve would conflate different communication
    patterns).

    Raises `ValueError` if no matching, `wire`-having `cu_grid` record exists for `algorithm`.
    """
    algorithm_records = [r for r in records if r["spec"]["algorithm"] == algorithm]
    grouped = _group_by_h(algorithm_records)
    if not grouped:
        raise ValueError(
            f"no completed, phase='cu_grid' records for algorithm={algorithm!r} — "
            "nothing to plot"
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    counted_run_ids: set[str] = set()
    for series_key, style in _SERIES_STYLE.items():
        xs: list[int] = []
        ys: list[float] = []
        for H in sorted(grouped):
            values = [
                r["wire"][series_key]
                for r in grouped[H]
                if r.get("wire", {}).get(series_key) is not None
            ]
            if not values:
                continue
            xs.append(H)
            ys.append(aggregate_repeats(values).median)
            if series_key == "bytes_per_training_token_measured":
                counted_run_ids.update(r["run_id"] for r in grouped[H])

        if xs:
            ax.plot(
                xs, ys,
                color=_SERIES_COLOR[series_key], linestyle=style["linestyle"],
                marker=style["marker"], label=style["label"],
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("H (synchronization interval)")
    ax.set_ylabel("Bytes on wire per training token")
    ax.legend(fontsize=9, loc="upper right")

    version_label = harness_version if harness_version is not None else "mixed/unspecified"
    ax.set_title(
        f"Bytes-on-wire per token vs. H — {algorithm}\n"
        f"harness_version={version_label} · {len(counted_run_ids)} contributing runs "
        "(pooled across bandwidth levels — see module docstring) · solid=measured, dashed=predicted"
    )
    fig.tight_layout()
    return fig
