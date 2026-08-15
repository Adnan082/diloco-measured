# Playbook

**Audience:** a practitioner with a slow multi-node cluster who wants an `H` recommendation
and nothing else (Journey C, `CLAUDE.md` §9.3).

**Status:** `[PARTIAL]` — the fitted `PredictorModel` (FR-07, G4) is still not built, so the
`diloco-measured plan --probe` tool and its calibration-domain warnings below remain
`[UNKNOWN]`/placeholder. But real measured data now exists (`CLAUDE.md` ADR-035/037) for one
specific configuration — see "What we've actually measured so far," immediately below — which
is real, not extrapolated, and narrower than a fitted predictor would eventually cover.

---

## What we've actually measured so far (real data, not a fitted predictor)

**Scope — read this before using the numbers below:** DiLoCo, exactly 4 replicas (1 GPU
each, `g6e.2xlarge`/L40S), a 30.8M-parameter model, real FineWeb-Edu data. If your setup
doesn't look roughly like this, these numbers are **not validated for you** — they're
measured points, not a calibrated, interpolated model with a stated domain (that's what G4
will eventually provide). Source: `results/raw/cu_grid-diloco-30m-h*-bw*-r{0,1,2}.json`
(median of 3 real repeats each).

| Your bandwidth | H=1 (≈DDP) | H=8 | H=32 | H=128 |
| --- | --- | --- | --- | --- |
| 50 Mbit/s | 0.07% CU — don't | 0.55% CU — don't | 2.41% CU — weak | **11.55% CU** |
| 200 Mbit/s | 0.32% CU — don't | 2.48% CU — weak | **10.01% CU** | **36.09% CU** |
| 1 Gbit/s | 1.74% CU — weak | 12.35% CU | **36.01% CU** | **71.52% CU** |
| 5 Gbit/s | 7.76% CU | 40.24% CU | **68.10% CU** | **87.00% CU** |

Reading this: at 50–200 Mbit/s, only `H=128` gets meaningfully above single-digit compute
utilization — anything lower and your GPUs spend nearly all their time waiting on the network,
not training. At 1 Gbit/s and above, `H=32` starts becoming reasonable if you need more
frequent synchronization than `H=128` for quality reasons. **`H=1` (i.e., plain DDP) is not
viable anywhere in this table below 5 Gbit/s** — CU never exceeds 8%.

**Caveat found in this same data (`CLAUDE.md` ADR-037):** a real convergence campaign at these
same `H`/bandwidth points found none of them reached a single-GPU reference's target loss
within a 400,000-token budget, and the ordering across `H` was **not** monotone (H=8 slightly
outperformed H=128 on final loss, on a single seed). High compute utilization is not the same
as fast convergence — this table answers "how much of my GPU time is wasted on the network,"
not "which `H` reaches a target loss fastest." Use both this table and that caveat together,
not this table alone.

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
