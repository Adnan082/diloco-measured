# Results Log

**Status:** `[CONFIRMED format, EMPTY content]` — no campaigns have run yet.

This file is the human-readable ledger of every campaign: what ran, what didn't, and why.
Per `CLAUDE.md` §16.3 and §25, nothing in `results/raw/` is ever edited or deleted — this file
is where the *narrative* of failures, crashes, and abandoned lines is published rather than
hidden. It is updated after every campaign (`CLAUDE.md` §34.2 Definition of Done).

Do not write a number here that is not backed by a record in `results/raw/`.

---

## Campaign log

| Date | Campaign | `harness_version` | Points attempted | Completed | Crashed | Aborted (shaping) | Diverged | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — | No campaigns run yet |

## Cumulative cost

| Date | Cluster-hours | Estimated spend | Running total |
| --- | --- | --- | --- |
| — | — | — | $0 (cluster not yet launched) |

## Known-bad / superseded records

`[CONFIRMED]` None yet. When a `harness_version` bump supersedes prior records, list the
superseded run IDs here per §16.3, rather than deleting them.

## Null / negative results

`[CONFIRMED]` Reported here as a first-class outcome, per §2.7 and §25.3 — a silent skip is a
bug. None recorded yet.
