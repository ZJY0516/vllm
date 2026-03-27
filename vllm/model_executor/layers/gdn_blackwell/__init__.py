"""Blackwell CuTe-DSL GDN (Gated Delta Networks) prefill kernel.

Standalone port of the FlashInfer Blackwell GDN kernel — no flashinfer
dependency.  Only requires ``cutlass`` (CuTe DSL) and ``cuda.bindings``.

Public API:
    chunk_gated_delta_rule(q, k, v, g, beta, ...)
        Same signature as flashinfer.gdn_prefill.chunk_gated_delta_rule.
"""

from .gdn import GDN, chunk_gated_delta_rule

__all__ = ["GDN", "chunk_gated_delta_rule"]
