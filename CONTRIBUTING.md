# Contributing

This project is **closed** (see the README's "Why the project stops here" and `CLAUDE.md`
ADR-040) — it reached its stated goals (G1–G4) and the author has moved on. It is not
actively maintained, and pull requests are not being solicited.

That said:

- **Issues are welcome** — if you spot a real bug, a factual error, or something that
  contradicts the repository's own stated claims, please open one. It may not get fixed
  quickly (or at all), but it will be read.
- **Forks are explicitly encouraged.** Every open item this project didn't get to (FSDP2 on
  real hardware, the 1B model, `/proc/net/dev` wire-byte accounting, fault injection,
  compression — see `RESULTS.md` and each ADR's "Not resolved" section for the precise list)
  is scoped precisely enough that someone else could pick it up without archaeology. Apache-2.0
  permits this without asking.
- **Small, well-justified fixes** (a broken link, a stale number, a typo in `CLAUDE.md`) may
  still get merged if they don't touch the measurement path — see `CLAUDE.md` §33's coding
  rules and §44's change management before touching anything under `src/diloco_measured/
  measurement/` or `results/raw/`, both of which have real integrity constraints that aren't
  obvious from the diff alone.

If you do send a PR, the checks in `.github/workflows/ci.yml` (ruff, mypy, the analysis-only
test subset, and figure regeneration) all need to pass.
