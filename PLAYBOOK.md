# Playbook

**Audience:** a practitioner with a slow multi-node cluster who wants an `H` recommendation
and nothing else (Journey C, `CLAUDE.md` §9.3).

**Status:** `[UNKNOWN]` — this file cannot be populated until a `PredictorModel` is fitted
(FR-07, blocked on Phase 5 / M6). Everything below is a placeholder for the eventual shape.

---

## Quick answer (once available)

```bash
diloco-measured plan --probe --model 1b --gpus 4
```

Measures your live inter-node bandwidth, evaluates the calibrated model, and prints:

- recommended `H`
- expected tokens/s
- expected compute utilization
- expected bytes-on-wire per hour
- an **explicit extrapolation warning** if your network falls outside the calibration domain
  (never silent — `CLAUDE.md` FR-07)

## Lookup table (once available)

`[UNKNOWN]` — populated from the fitted `PredictorModel`'s validated domain after Phase 3/5.

| GPUs | Bandwidth | Recommended H | Expected CU | Expected tok/s | Confidence |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | No predictor fitted yet |

## Calibration domain

`[UNKNOWN]` — will state the bandwidth range, model-size range, and `H` range the predictor was
actually validated on. A recommendation outside this domain must always carry an extrapolation
warning; if this document ever gives one without that warning, that is a bug in the tool, not
a feature of the playbook.

## What to do if your situation isn't covered

Until this playbook is populated: don't copy an `H` from a paper. See `CLAUDE.md` §2.5–2.6 for
why that's exactly the practice this project exists to replace.
