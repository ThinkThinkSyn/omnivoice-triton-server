"""CUDA Graph optimized OmniVoice runner with fixed prewarmed shapes."""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch

from triton_backend.models.base_runner import BaseRunner

logger = logging.getLogger(__name__)


_GIB = 1024**3


def _fallback_max_batch_candidates(requested: int) -> list[int]:
    requested = max(1, int(requested))
    candidates = {requested}
    value = 1
    while value < requested:
        candidates.add(value)
        value *= 2
    return sorted(candidates, reverse=True)


def _fallback_max_width_candidates(requested: int, min_width: int) -> list[int]:
    requested = max(1, int(requested))
    min_width = max(1, int(min_width))
    anchors = [64, 128, 160, 256, 512]
    candidates = {requested}

    for width in reversed(anchors):
        if min_width <= width < requested:
            candidates.add(width)

    if min_width < requested:
        candidates.add(min_width)

    return sorted(candidates, reverse=True)


def _dedupe_sorted(values: list[int]) -> list[int]:
    return sorted({max(1, int(value)) for value in values})


def _capped(values: list[int], max_value: int) -> list[int]:
    return _dedupe_sorted([value for value in values if value <= max_value])


class _CUDAGraphForward:
    """Wraps OmniVoice.forward() with a fixed prewarmed CUDA Graph table.

    Runtime inputs are padded up to the nearest prewarmed ``(batch, codebooks,
    width)`` shape. New shapes are not captured on demand.
    """

    def __init__(
        self,
        model: Any,
        max_batch_size: int = 16,
        min_width: int = 32,
        max_width: int = 128,
    ) -> None:
        self._model = model
        self._original_forward = model.forward
        self._graphs: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._requested_max_batch_size = max(1, int(max_batch_size))
        self._max_batch_size = self._requested_max_batch_size
        self._num_codebooks = int(getattr(model.config, "num_audio_codebook", 8))
        self._pad_id = int(getattr(model.config, "audio_mask_id", 1024))
        self._min_width = max(1, int(min_width))
        self._requested_max_width = max(self._min_width, int(max_width))
        self._max_width = self._requested_max_width
        self._batch_buckets: list[int] = []
        self._width_buckets: list[int] = []
        self._width_batch_buckets: dict[int, list[int]] = {}
        self._skipped_shapes: list[dict[str, Any]] = []
        self._memory_before: dict[str, int] = {}
        self._memory_after: dict[str, int] = {}
        self._memory_headroom_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._captures = 0
        self._capture_failures = 0

    def prewarm(self) -> None:
        """Capture all configured CUDA Graph shapes during startup."""
        self._memory_before = self._memory_snapshot()
        total_bytes = self._memory_before.get("total_bytes", 0)
        self._memory_headroom_bytes = max(2 * _GIB, int(total_bytes * 0.12))

        last_error: Exception | None = None
        graph_plan_ready = False

        for max_width in _fallback_max_width_candidates(self._requested_max_width, self._min_width):
            self._max_width = max_width
            for max_batch in _fallback_max_batch_candidates(self._requested_max_batch_size):
                self._clear_graph_entries()
                self._max_batch_size = max_batch
                shape_plan = self._shape_plan(max_batch)
                try:
                    logger.info(
                        "Prewarming mandatory CUDA Graphs: requested_max_batch=%d "
                        "requested_max_width=%d effective_batch_candidate=%d "
                        "effective_width_candidate=%d width_batch_plan=%s",
                        self._requested_max_batch_size,
                        self._requested_max_width,
                        max_batch,
                        max_width,
                        shape_plan,
                    )
                    for width, batch_buckets in shape_plan.items():
                        self._capture_width_group(width, batch_buckets)
                        self._width_batch_buckets[width] = list(batch_buckets)
                    if not self._has_memory_headroom():
                        raise RuntimeError(
                            "Mandatory CUDA Graph shapes exceeded memory headroom: "
                            f"memory={self._memory_snapshot()}"
                        )
                    self._width_buckets = sorted(shape_plan)
                    self._batch_buckets = sorted(
                        {bucket for buckets in shape_plan.values() for bucket in buckets}
                    )
                    graph_plan_ready = True
                    break
                except Exception as exc:
                    last_error = exc
                    self._skipped_shapes.append(
                        {
                            "phase": "mandatory",
                            "max_business_batch_size": max_batch,
                            "max_width": max_width,
                            "width_batch_plan": shape_plan,
                            "reason": repr(exc),
                        }
                    )
                    logger.warning(
                        "Skipping CUDA Graph effective max batch %d max width %d "
                        "after prewarm failure",
                        max_batch,
                        max_width,
                        exc_info=True,
                    )
                    self._clear_graph_entries()
                    torch.cuda.empty_cache()
            if graph_plan_ready:
                break

        if not graph_plan_ready:
            raise RuntimeError("No CUDA Graph shape set fits available memory") from last_error

        self._prewarm_optional_shapes()
        self._memory_after = self._memory_snapshot()
        logger.info(
            "Prewarmed %d CUDA Graph shapes. requested_max_batch=%d "
            "requested_max_width=%d effective_max_batch=%d effective_max_width=%d "
            "business_batch_buckets=%s width_buckets=%s "
            "memory_headroom_gb=%.2f memory_after=%s",
            len(self._graphs),
            self._requested_max_batch_size,
            self._requested_max_width,
            self._max_batch_size,
            self._max_width,
            self._batch_buckets,
            self._width_buckets,
            self._memory_headroom_bytes / _GIB,
            self._memory_after,
        )

    def _shape_plan(self, max_batch: int) -> dict[int, list[int]]:
        """Return width -> business batch buckets for fixed graph prewarm.

        Small widths keep low-latency b1 graphs and a few short-batch graphs.
        Wider graphs are expensive, so they only keep b1 plus one throughput
        shape where it is likely to pay off.
        """
        max_batch = max(1, int(max_batch))
        widths = self._planned_widths()
        plan: dict[int, list[int]] = {}

        for width in widths:
            if width <= 64:
                batches = [1, 4, max_batch]
            elif width <= 128:
                batches = [1, 4, 8, max_batch]
            elif width <= 160:
                batches = [1, 4, 8, max_batch]
            elif width <= 256:
                batches = [1, 4, 8]
            else:
                batches = [1]
            plan[width] = _capped(batches, max_batch)

        return {width: buckets for width, buckets in plan.items() if buckets}

    def _optional_shape_plan(self) -> dict[int, list[int]]:
        plan: dict[int, list[int]] = {}
        if self._max_width >= 512:
            large_batches = [4]
            total_bytes = self._memory_before.get("total_bytes", 0)
            if total_bytes >= 28 * _GIB:
                large_batches.extend([8, self._max_batch_size])
            plan[512] = _capped(large_batches, self._max_batch_size)
        if self._min_width <= 64 <= self._max_width:
            plan[64] = _capped([4, self._max_batch_size], self._max_batch_size)
        if self._max_width > 512:
            plan[self._max_width] = _capped([4, 8], self._max_batch_size)
        return {width: buckets for width, buckets in plan.items() if buckets}

    def _prewarm_optional_shapes(self) -> None:
        for width, batch_buckets in self._optional_shape_plan().items():
            for business_batch in batch_buckets:
                key = (business_batch * 2, self._num_codebooks, width)
                if key in self._graphs:
                    continue
                try:
                    self._capture_shape(width, business_batch)
                    if not self._has_memory_headroom():
                        raise RuntimeError(
                            "Optional CUDA Graph shape exceeded memory headroom: "
                            f"memory={self._memory_snapshot()}"
                        )
                except Exception as exc:
                    self._drop_graph_keys([key])
                    self._skipped_shapes.append(
                        {
                            "phase": "optional",
                            "business_batch_size": business_batch,
                            "width": width,
                            "reason": repr(exc),
                        }
                    )
                    logger.info(
                        "Skipping optional CUDA Graph shape business_batch=%d width=%d",
                        business_batch,
                        width,
                        exc_info=True,
                    )
                    torch.cuda.empty_cache()
                    break

                self._width_buckets = sorted({*self._width_buckets, width})
                buckets = self._width_batch_buckets.setdefault(width, [])
                if business_batch not in buckets:
                    buckets.append(business_batch)
                    buckets.sort()
                if business_batch not in self._batch_buckets:
                    self._batch_buckets.append(business_batch)
                    self._batch_buckets.sort()

    def _planned_widths(self) -> list[int]:
        anchors = [128, 160, 256]
        widths = [width for width in anchors if self._min_width <= width <= self._max_width]
        if not widths:
            widths.append(self._max_width)

        if self._max_width > 256:
            widths.append(min(self._max_width, 512))
        if self._max_width > 512:
            widths.append(self._max_width)

        return _dedupe_sorted(widths)

    def _capture_width_group(
        self,
        width: int,
        batch_buckets: list[int],
    ) -> list[tuple[int, int, int]]:
        captured: list[tuple[int, int, int]] = []
        for business_batch in batch_buckets:
            try:
                key = self._capture_shape(width, business_batch)
            except Exception:
                self._capture_failures += 1
                logger.exception(
                    "CUDA Graph prewarm failed for business_batch=%d width=%d",
                    business_batch,
                    width,
                )
                self._drop_graph_keys(captured)
                raise
            captured.append(key)
        return captured

    def _capture_shape(self, width: int, business_batch: int) -> tuple[int, int, int]:
        key = (business_batch * 2, self._num_codebooks, width)
        input_ids, audio_mask, attention_mask = self._dummy_inputs(key)
        self._capture(input_ids, audio_mask, attention_mask)
        return key

    def _memory_snapshot(self) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }

    def _has_memory_headroom(self) -> bool:
        if not torch.cuda.is_available():
            return True
        free_bytes, _ = torch.cuda.mem_get_info()
        return int(free_bytes) >= self._memory_headroom_bytes

    def _dummy_inputs(
        self,
        key: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, codebooks, width = key
        device = self._model.device
        input_ids = torch.full(
            (batch, codebooks, width),
            self._pad_id,
            dtype=torch.long,
            device=device,
        )
        audio_mask = torch.zeros((batch, width), dtype=torch.bool, device=device)
        attention_mask = torch.zeros(
            (batch, 1, width, width),
            dtype=torch.bool,
            device=device,
        )
        diag = torch.arange(width, device=device)
        attention_mask[:, :, diag, diag] = True
        return input_ids, audio_mask, attention_mask

    def _select_key(self, input_ids: torch.Tensor) -> tuple[int, int, int] | None:
        batch, codebooks, width = tuple(input_ids.shape)
        if codebooks != self._num_codebooks:
            return None

        candidates = [
            key
            for key in self._graphs
            if key[1] == codebooks and key[0] >= batch and key[2] >= width
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda key: (key[2], key[0]))

    def _capture(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        key = tuple(input_ids.shape)
        logger.info("Capturing prewarmed CUDA Graph for shape %s ...", key)
        torch.cuda.synchronize()

        static_input_ids = input_ids.clone()
        static_audio_mask = audio_mask.clone()
        static_attn_mask = (
            attention_mask.clone() if attention_mask is not None else None
        )
        static_doc_ids = document_ids.clone() if document_ids is not None else None
        static_pos_ids = position_ids.clone() if position_ids is not None else None

        kwargs: dict[str, Any] = {}
        if static_attn_mask is not None:
            kwargs["attention_mask"] = static_attn_mask
        if static_doc_ids is not None:
            kwargs["document_ids"] = static_doc_ids
        if static_pos_ids is not None:
            kwargs["position_ids"] = static_pos_ids

        torch.cuda.synchronize()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            static_output = self._original_forward(
                static_input_ids,
                static_audio_mask,
                **kwargs,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_output = self._original_forward(
                static_input_ids,
                static_audio_mask,
                **kwargs,
            )

        entry = {
            "graph": graph,
            "replay_lock": threading.Lock(),
            "static_input_ids": static_input_ids,
            "static_audio_mask": static_audio_mask,
            "static_attn_mask": static_attn_mask,
            "static_attn_mask_base": (
                static_attn_mask.clone() if static_attn_mask is not None else None
            ),
            "static_doc_ids": static_doc_ids,
            "static_pos_ids": static_pos_ids,
            "static_output": static_output,
        }
        self._graphs[key] = entry
        self._captures += 1
        torch.cuda.synchronize()
        logger.info("CUDA Graph prewarmed for shape %s", key)
        return entry

    def _prepare_replay_buffers(
        self,
        entry: dict[str, Any],
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> None:
        batch, _, width = tuple(input_ids.shape)

        static_input_ids = entry["static_input_ids"]
        static_audio_mask = entry["static_audio_mask"]
        static_input_ids.fill_(self._pad_id)
        static_audio_mask.zero_()
        static_input_ids[:batch, :, :width].copy_(input_ids)
        static_audio_mask[:batch, :width].copy_(audio_mask)

        static_attn_mask = entry["static_attn_mask"]
        if static_attn_mask is not None:
            static_attn_mask.copy_(entry["static_attn_mask_base"])
            if attention_mask is not None:
                static_attn_mask[:batch, :, :width, :width].copy_(attention_mask)

        static_doc_ids = entry["static_doc_ids"]
        if static_doc_ids is not None and document_ids is not None:
            static_doc_ids.fill_(-1)
            static_doc_ids[: document_ids.shape[0], : document_ids.shape[1]].copy_(
                document_ids
            )

        static_pos_ids = entry["static_pos_ids"]
        if static_pos_ids is not None and position_ids is not None:
            static_pos_ids.zero_()
            static_pos_ids[: position_ids.shape[0], : position_ids.shape[1]].copy_(
                position_ids
            )

    def _slice_output(self, output: Any, batch: int, width: int) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None:
            return output
        values = dict(output.items()) if hasattr(output, "items") else {}
        values["logits"] = logits[:batch, :, :width, :]
        return output.__class__(**values)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> Any:
        if labels is not None or self._model.training:
            return self._original_forward(
                input_ids,
                audio_mask,
                labels,
                attention_mask,
                document_ids,
                position_ids,
            )

        actual_batch, _, actual_width = tuple(input_ids.shape)
        key = self._select_key(input_ids)
        if key is None or key not in self._graphs:
            self._cache_misses += 1
            logger.warning(
                "No prewarmed CUDA Graph shape can cover input_shape=%s; "
                "falling back to eager forward.",
                tuple(input_ids.shape),
            )
            return self._original_forward(
                input_ids,
                audio_mask,
                labels,
                attention_mask,
                document_ids,
                position_ids,
            )

        self._cache_hits += 1
        entry = self._graphs[key]
        with entry["replay_lock"]:
            self._prepare_replay_buffers(
                entry,
                input_ids,
                audio_mask,
                attention_mask,
                document_ids,
                position_ids,
            )
            entry["graph"].replay()
            return self._slice_output(entry["static_output"], actual_batch, actual_width)

    def clear(self) -> None:
        self._clear_graph_entries()
        torch.cuda.empty_cache()

    def _drop_graph_keys(self, keys: list[tuple[int, int, int]]) -> None:
        for key in keys:
            entry = self._graphs.pop(key, None)
            if entry is None:
                continue
            graph = entry.get("graph")
            if graph is not None and hasattr(graph, "reset"):
                graph.reset()
            entry.clear()

    def _clear_graph_entries(self) -> None:
        for entry in self._graphs.values():
            graph = entry.get("graph")
            if graph is not None and hasattr(graph, "reset"):
                graph.reset()
            entry.clear()
        self._graphs.clear()
        self._width_batch_buckets.clear()
        self._batch_buckets.clear()
        self._width_buckets.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": "prewarmed_fixed_shapes",
            "requested_max_business_batch_size": self._requested_max_batch_size,
            "max_business_batch_size": self._max_batch_size,
            "requested_max_width": self._requested_max_width,
            "max_width": self._max_width,
            "business_batch_buckets": self._batch_buckets,
            "width_buckets": self._width_buckets,
            "width_batch_buckets": {
                str(width): buckets for width, buckets in sorted(self._width_batch_buckets.items())
            },
            "entries": len(self._graphs),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "captures": self._captures,
            "capture_failures": self._capture_failures,
            "memory_headroom_bytes": self._memory_headroom_bytes,
            "memory_before": self._memory_before,
            "memory_after": self._memory_after,
            "skipped_shapes": self._skipped_shapes,
            "shapes": [list(key) for key in sorted(self._graphs.keys())],
        }


class FasterRunner(BaseRunner):
    """BaseRunner with prewarmed CUDA Graph shape bucketing."""

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "fp16",
        cuda_graph_max_batch_size: int = 16,
        cuda_graph_min_width: int = 32,
        cuda_graph_max_width: int = 128,
    ) -> None:
        super().__init__(device=device, model_id=model_id, dtype=dtype)
        self.cuda_graph_max_batch_size = cuda_graph_max_batch_size
        self.cuda_graph_min_width = cuda_graph_min_width
        self.cuda_graph_max_width = cuda_graph_max_width
        self._graph_forward: _CUDAGraphForward | None = None

    def load_model(self) -> None:
        super().load_model()
        self._graph_forward = _CUDAGraphForward(
            self._model,
            max_batch_size=self.cuda_graph_max_batch_size,
            min_width=self.cuda_graph_min_width,
            max_width=self.cuda_graph_max_width,
        )
        self._graph_forward.prewarm()
        self._model.forward = self._graph_forward
        logger.info("FasterRunner ready (prewarmed CUDA Graph wrapper installed).")

    def unload_model(self) -> None:
        if self._graph_forward is not None:
            self._graph_forward.clear()
            self._graph_forward = None
        super().unload_model()

    def graph_cache_stats(self) -> dict[str, Any]:
        if self._graph_forward is None:
            return {"enabled": False, "entries": 0}
        return self._graph_forward.stats()
