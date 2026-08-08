"""diloco-measured — a bandwidth-controlled evaluation of semi-synchronous LLM training.

See CLAUDE.md at the repository root for the full specification. This package is split into
two halves that must never import each other (CLAUDE.md §11.2, §14.2):

    diloco_measured.measurement   — needs GPUs + AWS. Produces results/.
    diloco_measured.analysis      — pure. Reads results/, never writes measurement inputs.

STATUS: [PROPOSED] scaffold — Phase 0 implementation has not started.
"""

__version__ = "0.1.0"
