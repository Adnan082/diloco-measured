# Security

This is a closed research artifact (see `CLAUDE.md`'s "Document status" and the README's
"Why the project stops here") — a measurement rig and a committed result corpus, not a running
service. There is no production deployment, no hosted endpoint, and no user data anywhere in
this repository (see `CLAUDE.md` §24 Privacy and Data Handling).

## Reporting an issue

If you find something here that looks like a real security problem — a credential leaked into
a committed file, a script that would do something unsafe if run as written, or similar —
please open a GitHub issue on this repository, or contact the author directly if you'd rather
not disclose it publicly first. There is no bug bounty and no SLA; this is a solo research
project, not a maintained product.

## What's in scope

- Anything in `infra/` that could affect someone else's AWS account if they ran it
  themselves (e.g. an overly permissive IAM policy suggestion, a script that doesn't clean up
  after itself).
- Any credential, account identifier, or private network detail that made it into a committed
  file despite the project's stated scrubbing policy (`CLAUDE.md` §23).

## What's out of scope

- The measurement results themselves (`results/raw/`) are immutable, append-only data — not
  a thing that can have a "vulnerability."
- The project's own documented, deliberate trade-offs (e.g. `tc`/`tbf` shaping's lack of
  latency/jitter/loss modeling, or any item in `LIMITATIONS.md`) are known scope decisions,
  not bugs.
