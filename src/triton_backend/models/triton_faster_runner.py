"""Hybrid OmniVoice runner: Triton kernel fusion + CUDA Graph."""

import logging

from triton_backend.models.faster_runner import FasterRunner
from triton_backend.models.patching import (
    apply_sage_attention,
    apply_triton_kernels,
    find_patchable_model,
)

logger = logging.getLogger(__name__)


class TritonFasterRunner(FasterRunner):
    """FasterRunner with Triton kernel patching applied BEFORE graph capture.

    Combines:
      1. Triton kernel fusion (RMSNorm, SwiGLU, Fused Norm+Residual)
      2. Optional SageAttention
      3. CUDA Graph capture and replay

    Triton patches (and SageAttention) are applied first so that the
    captured graph includes the optimized kernels.

    Args:
        patch_range: Half-open ``(start, end)`` range of decoder layer
            indices to patch. ``None`` patches all layers.
        enable_sage_attention: Replace SDPA with SageAttention. Requires
            ``pip install sageattention``. Gracefully skips if unavailable.
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID.
        dtype: Model dtype string.
    """

    def __init__(
        self,
        patch_range: tuple[int, int] | None = None,
        enable_sage_attention: bool = False,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "fp16",
        cuda_graph_max_batch_size: int = 16,
        cuda_graph_min_width: int = 32,
        cuda_graph_max_width: int = 128,
    ) -> None:
        super().__init__(
            device=device,
            model_id=model_id,
            dtype=dtype,
            cuda_graph_max_batch_size=cuda_graph_max_batch_size,
            cuda_graph_min_width=cuda_graph_min_width,
            cuda_graph_max_width=cuda_graph_max_width,
        )
        self.patch_range = patch_range
        self.enable_sage_attention = enable_sage_attention

    def load_model(self) -> None:
        """Load model, apply Triton patches, then install CUDA Graph wrapper."""
        # Load base model (without CUDA Graph wrapper yet)
        from triton_backend.models.base_runner import BaseRunner

        BaseRunner.load_model(self)

        # Apply Triton kernel patches FIRST
        patchable = find_patchable_model(self._model)
        apply_triton_kernels(patchable, patch_range=self.patch_range)

        # Apply SageAttention if enabled
        if self.enable_sage_attention:
            apply_sage_attention(patchable, patch_range=self.patch_range)

        # Then install CUDA Graph wrapper on the patched model
        from triton_backend.models.faster_runner import _CUDAGraphForward

        self._graph_forward = _CUDAGraphForward(
            self._model,
            max_batch_size=self.cuda_graph_max_batch_size,
            min_width=self.cuda_graph_min_width,
            max_width=self.cuda_graph_max_width,
        )
        self._graph_forward.prewarm()
        self._model.forward = self._graph_forward

        logger.info(
            "HybridRunner ready (Triton kernels + CUDA Graph wrapper installed)."
        )
