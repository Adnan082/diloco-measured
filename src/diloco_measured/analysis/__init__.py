"""Analysis package — pure functions over committed results/. No GPU, no network, no credentials.

FORBIDDEN (CLAUDE.md §11.2, §14.2): nothing in this package may import `diloco_measured.measurement`,
open a socket, or initialize a CUDA context. This is what makes `make figures` work on a
reviewer's laptop (FR-11) — if that stops being true, it is the single worst regression this
package can have.
"""
