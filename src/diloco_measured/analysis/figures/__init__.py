"""One module per figure; each is a pure function of loaded/filtered/aggregated records.

Mapping (CLAUDE.md §10.2):
  fig1_cu_surface.py         — CU surface: measured vs. analytic (THE headline figure)
  fig2_nccl_bandwidth.py     — NCCL BW vs. message size (mechanism)
  fig3_tttl.py               — TTTL vs. bandwidth
  fig4_loss_throughput_vs_h.py — Loss@budget vs. H + throughput vs. H
  fig5_bytes_on_wire.py      — Bytes-on-wire per token
  fig6_predictor.py          — Predicted vs. measured H

RULE (CLAUDE.md §18): measured series are drawn solid; analytic/simulated series are dashed,
in matching colours. This convention alone carries the project's entire visual argument — do
not violate it in a new figure module. Every figure states, in caption or metadata, the
harness version, the number of contributing runs, and whether values are measured, analytic,
or interpolated (FR-13).

STATUS: [PROPOSED] scaffold — no figure modules implemented yet (blocked on real data, Phase 3+).
"""
