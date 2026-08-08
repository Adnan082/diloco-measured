"""Pluggable pseudo-gradient compression codecs (FR-10, secondary goal G6).

Codecs: fp16, int8-with-error-feedback, top-k. See methods/diloco.md §3 invariant 4: the
error-feedback residual accumulator MUST persist across outer steps and MUST be included in
checkpoints — this is called out explicitly (§30.2) as the invariant most likely to be
silently broken.

STATUS: [PROPOSED — secondary goal].
"""

from __future__ import annotations

from typing import Protocol


class Codec(Protocol):
    """A pseudo-gradient compression codec.

    CONTRACT: encode/decode round-trip error must be bounded and tested (§30.2). Any codec
    with persistent state (e.g. an error-feedback residual) owns that state internally and
    must expose it for checkpointing — it must never be dropped implicitly between rounds.
    """

    def encode(self, tensor): ...
    def decode(self, encoded): ...


class Fp16Codec:
    """Stateless. STATUS: [PROPOSED] scaffold."""

    def encode(self, tensor):
        raise NotImplementedError("Phase 4")

    def decode(self, encoded):
        raise NotImplementedError("Phase 4")


class Int8ErrorFeedbackCodec:
    """Stateful — owns a persistent residual accumulator. STATUS: [PROPOSED] scaffold.

    CONTRACT: the residual must be included whenever the owning trainer's checkpoint is
    written, and must be re-hydrated on resume. Dropping it silently reintroduces the exact
    bug this codec exists to avoid (methods/diloco.md §3 invariant 4).
    """

    def __init__(self):
        raise NotImplementedError("Phase 4")

    def encode(self, tensor):
        raise NotImplementedError("Phase 4")

    def decode(self, encoded):
        raise NotImplementedError("Phase 4")

    def state_dict(self) -> dict:
        raise NotImplementedError("Phase 4")

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError("Phase 4")


class TopKCodec:
    """Stateless (or optionally error-feedback — TBD Phase 4). STATUS: [PROPOSED] scaffold."""

    def encode(self, tensor):
        raise NotImplementedError("Phase 4")

    def decode(self, encoded):
        raise NotImplementedError("Phase 4")
