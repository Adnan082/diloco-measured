"""Pluggable pseudo-gradient compression codecs (FR-10, secondary goal G6).

Codecs: fp16, int8-with-error-feedback, top-k. See methods/diloco.md §3 invariant 4: the
error-feedback residual accumulator MUST persist across outer steps and MUST be included in
checkpoints — this is called out explicitly (CLAUDE.md §30.2) as the invariant most likely to
be silently broken. `Int8ErrorFeedbackCodec` and `TopKCodec(error_feedback=True)` below carry
their residual as instance state precisely so a caller cannot forget to thread it through.

STATUS: [PROPOSED — secondary goal]. Quantization is deliberately simple (symmetric linear /
top-k-by-magnitude) — nothing here claims to be state-of-the-art; it exists to measure the
wire-vs-quality tradeoff (G6), not to win a compression benchmark.
"""

from __future__ import annotations

from typing import Protocol

import torch


class Codec(Protocol):
    """A pseudo-gradient compression codec.

    CONTRACT: encode/decode round-trip error must be bounded and tested (CLAUDE.md §30.2).
    A codec with persistent state (an error-feedback residual) owns that state internally and
    exposes it via state_dict()/load_state_dict() for checkpointing — it must never be
    dropped implicitly between rounds.
    """

    def encode(self, tensor: torch.Tensor): ...
    def decode(self, encoded): ...


# ---------------------------------------------------------------------------------------
# fp16 — stateless
# ---------------------------------------------------------------------------------------


class Fp16Codec:
    """Stateless half-precision cast. Round-trip error is bounded by fp16's representable
    precision relative to the input magnitude (no accumulation across calls — nothing to
    checkpoint).
    """

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.half()

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded.float()


# ---------------------------------------------------------------------------------------
# int8 with error feedback — stateful
# ---------------------------------------------------------------------------------------


def _quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor linear quantization to int8. Returns (q_int8, scale)."""
    max_abs = x.abs().max()
    # Avoid a zero scale (all-zero tensor, or a residual that has fully cancelled out).
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def _dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


class Int8ErrorFeedbackCodec:
    """Stateful — owns a persistent residual accumulator (methods/diloco.md §3 invariant 4).

    CONTRACT: the residual must be included whenever the owning trainer's checkpoint is
    written (state_dict()) and re-hydrated on resume (load_state_dict()). Dropping it
    silently reintroduces the exact bug this codec exists to avoid: quantization error would
    be destroyed each round instead of deferred and corrected on the next one.
    """

    def __init__(self) -> None:
        self._residual: torch.Tensor | None = None

    def encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._residual is None:
            self._residual = torch.zeros_like(tensor)
        elif self._residual.shape != tensor.shape:
            raise ValueError(
                "Int8ErrorFeedbackCodec is stateful per fixed tensor shape; got a shape "
                f"change from {tuple(self._residual.shape)} to {tuple(tensor.shape)}. "
                "Use one codec instance per parameter tensor."
            )
        biased = tensor + self._residual
        q, scale = _quantize_int8(biased)
        dequantized = _dequantize_int8(q, scale)
        # The error is CARRIED FORWARD, not dropped — this line is the invariant.
        self._residual = biased - dequantized
        return q, scale

    def decode(self, encoded: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        q, scale = encoded
        return _dequantize_int8(q, scale)

    def state_dict(self) -> dict:
        return {"residual": None if self._residual is None else self._residual.clone()}

    def load_state_dict(self, state: dict) -> None:
        residual = state.get("residual")
        self._residual = None if residual is None else residual.clone()


# ---------------------------------------------------------------------------------------
# top-k — stateless by default, optionally error-feedback
# ---------------------------------------------------------------------------------------


class TopKCodec:
    """Keeps the `k_fraction` largest-magnitude elements, zeroing the rest.

    With `error_feedback=True`, the elements dropped this round are carried forward and
    added into next round's tensor before re-selecting top-k — same invariant as
    `Int8ErrorFeedbackCodec`, applied to sparsification instead of quantization.
    """

    def __init__(self, k_fraction: float = 0.1, error_feedback: bool = False) -> None:
        if not 0 < k_fraction <= 1:
            raise ValueError("k_fraction must be in (0, 1]")
        self.k_fraction = k_fraction
        self.error_feedback = error_feedback
        self._residual: torch.Tensor | None = None

    def encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        shape = tensor.shape
        flat = tensor.reshape(-1)

        if self.error_feedback:
            if self._residual is None:
                self._residual = torch.zeros_like(flat)
            elif self._residual.shape != flat.shape:
                raise ValueError(
                    "TopKCodec(error_feedback=True) is stateful per fixed tensor shape; "
                    "use one codec instance per parameter tensor."
                )
            biased = flat + self._residual
        else:
            biased = flat

        k = max(1, min(biased.numel(), round(self.k_fraction * biased.numel())))
        _, indices = torch.topk(biased.abs(), k)
        values = biased[indices].clone()

        if self.error_feedback:
            reconstructed = torch.zeros_like(biased)
            reconstructed[indices] = values
            self._residual = biased - reconstructed

        return indices, values, shape

    def decode(self, encoded: tuple[torch.Tensor, torch.Tensor, torch.Size]) -> torch.Tensor:
        indices, values, shape = encoded
        numel = 1
        for s in shape:
            numel *= s
        flat = torch.zeros(numel, dtype=values.dtype)
        flat[indices] = values
        return flat.reshape(shape)

    def state_dict(self) -> dict:
        return {"residual": None if self._residual is None else self._residual.clone()}

    def load_state_dict(self, state: dict) -> None:
        residual = state.get("residual")
        self._residual = None if residual is None else residual.clone()
