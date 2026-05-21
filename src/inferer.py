from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audio import float_to_pcm16
from chunking import truncate_text_by_word_count
from config import Settings
from logging_config import configure_logging
from modeling.models.omnivoice import (
    VoiceClonePrompt,
    _ZH_RE,
    _combine_text,
    _resolve_instruct,
    _resolve_language,
    _tokenize_with_nonverbal_tags,
)
from metrics_shm import SharedMetricsWriter
from protocol import InferRequest, InferTask

logger = logging.getLogger(__name__)


def collapse_equal(values: list[Any], default: Any = None) -> Any:
    if not values:
        return default
    first = values[0]
    if all(value == first for value in values):
        return first
    return values


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _cuda_graph_width_bucket(value: int) -> int:
    value = max(1, int(value))
    for bucket in (64, 128, 160, 256, 512, 640, 768, 1024, 1536, 2048):
        if value <= bucket:
            return bucket
    return _next_power_of_two(value)


def _model_max_position_embeddings(model_id: str) -> int | None:
    config_path = Path(model_id).expanduser() / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None

    llm_config = config.get("llm_config") if isinstance(config, dict) else None
    value = llm_config.get("max_position_embeddings") if isinstance(llm_config, dict) else None
    return value if isinstance(value, int) and value > 0 else None


@dataclass
class QueuedTask:
    task: InferTask
    futures: list[asyncio.Future[dict[str, Any]]]
    enqueued_at: float = field(default_factory=time.monotonic)

    @property
    def chunk_count(self) -> int:
        return len(self.futures)


@dataclass
class ChunkJob:
    task: InferTask
    seq: int
    text: str
    duration: float | None
    prompt: VoiceClonePrompt | None
    prompt_affects_duration: bool = True


@dataclass
class TimingStats:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        elapsed_s = max(0.0, float(elapsed_s))
        self.count += 1
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)

    def snapshot(self) -> dict[str, Any]:
        avg_s = self.total_s / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total_s": round(self.total_s, 3),
            "avg_ms": round(avg_s * 1000.0, 3),
            "max_ms": round(self.max_s * 1000.0, 3),
        }


class TritonBackend:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.runner: Any = None
        self.streams: list[torch.cuda.Stream | None] = []
        self.generate_lock = threading.RLock()
        self.clone_prompt_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self.clone_prompt_cache_lock = threading.RLock()
        self.clone_prompt_cache_hits = 0
        self.clone_prompt_cache_misses = 0
        self.clone_prompt_cache_evictions = 0
        self.shared_clone_prompt_cache_hits = 0
        self.shared_clone_prompt_cache_stores = 0
        self.shared_clone_prompt_cache_load_errors = 0
        self.chunk_job_width_cache: OrderedDict[tuple[Any, ...], int] = OrderedDict()
        self.chunk_job_width_cache_hits = 0
        self.chunk_job_width_cache_misses = 0
        self.profile: dict[str, TimingStats] = {}

    def _profile_add(self, key: str, elapsed_s: float) -> None:
        stats = self.profile.get(key)
        if stats is None:
            stats = TimingStats()
            self.profile[key] = stats
        stats.add(elapsed_s)

    def profile_stats(self) -> dict[str, Any]:
        return {
            key: stats.snapshot()
            for key, stats in sorted(self.profile.items())
            if stats.count
        }

    def load(self) -> None:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[self.cfg.dtype]
        if self.cfg.runner_mode == "official":
            from modeling import OmniVoice

            self.model = OmniVoice.from_pretrained(
                self.cfg.model_id,
                device_map=self.cfg.device,
                dtype=dtype,
            )
        else:
            from triton_backend import create_runner

            runner_kwargs: dict[str, Any] = {
                "model_id": self.cfg.model_id,
                "device": self.cfg.device,
                "dtype": self.cfg.dtype,
            }
            if self.cfg.runner_mode == "hybrid":
                self.cfg.cuda_graph_max_width = self._resolve_cuda_graph_max_width()
                runner_kwargs["cuda_graph_max_batch_size"] = self.cfg.batch_size
                runner_kwargs["cuda_graph_min_width"] = self.cfg.cuda_graph_min_width
                runner_kwargs["cuda_graph_max_width"] = self.cfg.cuda_graph_max_width
            self.runner = create_runner(
                self.cfg.runner_mode,
                **runner_kwargs,
            )
            self.runner.load_model()
            self.model = self.runner.model

        if self.cfg.cuda_streams > 1:
            if str(self.cfg.device).startswith("cuda"):
                self.streams = [torch.cuda.Stream() for _ in range(self.cfg.cuda_streams)]
            else:
                logger.warning("cuda_streams=%d ignored for non-CUDA inferer", self.cfg.cuda_streams)
                self.streams = [None]
        else:
            self.streams = [None]

        logger.info(
            "Infer backend loaded: mode=%s cuda_streams=%d",
            self.cfg.runner_mode,
            len(self.streams),
        )

    def _resolve_cuda_graph_max_width(self) -> int:
        if self.cfg.cuda_graph_max_width > 0:
            return self.cfg.cuda_graph_max_width

        expected_chunk_words = self.cfg.text_chunk_words
        expected_width = _cuda_graph_width_bucket(
            expected_chunk_words * self.cfg.cuda_graph_auto_width_tokens_per_word + 128
        )
        model_limit = _model_max_position_embeddings(self.cfg.model_id)
        upper_bound = self.cfg.cuda_graph_auto_max_width
        if model_limit is not None:
            upper_bound = min(upper_bound, model_limit)
        resolved = min(max(128, expected_width), upper_bound)
        resolved = _cuda_graph_width_bucket(resolved)
        if resolved > upper_bound:
            resolved = upper_bound
        logger.info(
            "Resolved cuda_graph_max_width=%d from text_chunk_words=%d "
            "tokens_per_word=%d prompt_margin_tokens=128 "
            "model_max_position_embeddings=%s auto_max_width=%d",
            resolved,
            self.cfg.text_chunk_words,
            self.cfg.cuda_graph_auto_width_tokens_per_word,
            model_limit,
            self.cfg.cuda_graph_auto_max_width,
        )
        return resolved

    def _create_clone_prompt(self, ref_audio_bytes: bytes, ref_text: str | None) -> Any:
        from modeling.utils.audio import load_audio_bytes

        if not ref_text or not ref_text.strip():
            raise ValueError("ref_text is required for clone prompt creation")

        started_at = time.perf_counter()
        sample_rate = int(getattr(self.model, "sampling_rate", self.cfg.sample_rate))
        ref_wav = load_audio_bytes(ref_audio_bytes, sample_rate)
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=(ref_wav, sample_rate),
            ref_text=ref_text.strip(),
        )
        if str(self.cfg.device).startswith("cuda"):
            torch.cuda.synchronize()
        self._profile_add("clone_prompt_create", time.perf_counter() - started_at)
        return prompt

    def _clone_prompt_from_b64(self, ref_audio_b64: str, ref_text: str | None) -> Any:
        started_at = time.perf_counter()
        ref_audio_bytes = base64.b64decode(ref_audio_b64.encode("ascii"), validate=True)
        audio_hash = hashlib.sha256(ref_audio_bytes).hexdigest()
        text_hash = hashlib.sha256((ref_text or "").encode("utf-8")).hexdigest()
        key_id = f"{audio_hash}_{text_hash}"
        key = (audio_hash, ref_text or "")

        if self.cfg.max_clone_audio_prompt_cache <= 0:
            with self.clone_prompt_cache_lock:
                self.clone_prompt_cache_misses += 1
            try:
                return self._create_clone_prompt(ref_audio_bytes, ref_text)
            finally:
                self._profile_add("clone_prompt_total", time.perf_counter() - started_at)

        with self.clone_prompt_cache_lock:
            try:
                cached = self.clone_prompt_cache.get(key)
                if cached is not None:
                    self.clone_prompt_cache.move_to_end(key)
                    self.clone_prompt_cache_hits += 1
                    return cached

                self.clone_prompt_cache_misses += 1
                prompt = self._load_shared_clone_prompt(key_id)
                if prompt is None:
                    prompt = self._create_clone_prompt(ref_audio_bytes, ref_text)
                    self._store_shared_clone_prompt(key_id, prompt)
                self.clone_prompt_cache[key] = prompt
                while len(self.clone_prompt_cache) > self.cfg.max_clone_audio_prompt_cache:
                    _, evicted = self.clone_prompt_cache.popitem(last=False)
                    del evicted
                    self.clone_prompt_cache_evictions += 1
                return prompt
            finally:
                self._profile_add("clone_prompt_total", time.perf_counter() - started_at)

    def _shared_clone_prompt_path(self, key_id: str) -> Path | None:
        if not self.cfg.clone_prompt_shared_cache_dir:
            return None
        cache_dir = Path(self.cfg.clone_prompt_shared_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key_id}.pt"

    def _load_shared_clone_prompt(self, key_id: str) -> VoiceClonePrompt | None:
        path = self._shared_clone_prompt_path(key_id)
        if path is None or not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            tokens = payload["ref_audio_tokens"].to(getattr(self.model, "device", self.cfg.device))
            self.shared_clone_prompt_cache_hits += 1
            return VoiceClonePrompt(
                ref_audio_tokens=tokens,
                ref_text=str(payload["ref_text"]),
                ref_rms=float(payload["ref_rms"]),
            )
        except Exception:
            self.shared_clone_prompt_cache_load_errors += 1
            logger.warning("Failed to load shared clone prompt cache %s", path, exc_info=True)
            return None

    def _store_shared_clone_prompt(self, key_id: str, prompt: VoiceClonePrompt) -> None:
        path = self._shared_clone_prompt_path(key_id)
        if path is None:
            return
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                {
                    "ref_audio_tokens": prompt.ref_audio_tokens.detach().cpu(),
                    "ref_text": prompt.ref_text,
                    "ref_rms": prompt.ref_rms,
                },
                tmp,
            )
            tmp.replace(path)
            self.shared_clone_prompt_cache_stores += 1
            self._trim_shared_clone_prompt_cache()
        except Exception:
            logger.warning("Failed to store shared clone prompt cache %s", path, exc_info=True)
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def _trim_shared_clone_prompt_cache(self) -> None:
        if self.cfg.max_clone_audio_prompt_cache <= 0 or not self.cfg.clone_prompt_shared_cache_dir:
            return
        cache_dir = Path(self.cfg.clone_prompt_shared_cache_dir)
        entries = sorted(cache_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in entries[self.cfg.max_clone_audio_prompt_cache :]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def generation_config(self):
        from modeling import OmniVoiceGenerationConfig

        return OmniVoiceGenerationConfig(
            denoise=self.cfg.denoise,
            num_step=self.cfg.default_num_step,
            guidance_scale=self.cfg.guidance_scale,
            t_shift=self.cfg.t_shift,
            position_temperature=self.cfg.position_temperature,
            class_temperature=self.cfg.class_temperature,
            layer_penalty_factor=self.cfg.layer_penalty_factor,
            postprocess_output=False,
            audio_chunk_duration=self.cfg.audio_chunk_duration,
            audio_chunk_threshold=self.cfg.audio_chunk_threshold,
        )

    def generate_batch(self, tasks: list[InferTask], stream_idx: int) -> list[list[dict[str, Any]]]:
        with self.generate_lock:
            if any(len(self._task_chunks(task)) > 1 for task in tasks):
                return self._generate_batch_with_chunk_continuity(tasks, stream_idx)
            return [[result] for result in self._generate_single_chunk_batch(tasks, stream_idx)]

    def _generate_single_chunk_batch(
        self,
        tasks: list[InferTask],
        stream_idx: int,
    ) -> list[dict[str, Any]]:
        texts = [t.text for t in tasks]
        languages = [t.language for t in tasks]
        speeds = [t.speed for t in tasks]
        durations = [t.duration for t in tasks]
        instructs = [t.instruct for t in tasks]
        ref_texts = [t.ref_text for t in tasks]
        ref_audios = [t.ref_audio for t in tasks]
        ref_audio_b64s = [t.ref_audio_b64 for t in tasks]
        voice_clone_prompts_for_width: list[Any] | None = None

        kwargs: dict[str, Any] = {
            "text": texts,
            "generation_config": self.generation_config(),
        }
        if any(lang is not None for lang in languages):
            kwargs["language"] = collapse_equal(languages)
        if not any(duration is not None for duration in durations) and any(
            speed != 1.0 for speed in speeds
        ):
            kwargs["speed"] = collapse_equal(speeds)
        if any(duration is not None for duration in durations):
            kwargs["duration"] = collapse_equal(durations)
        if any(instruct is not None for instruct in instructs):
            kwargs["instruct"] = collapse_equal(instructs)
        if any(ref_text is not None for ref_text in ref_texts):
            kwargs["ref_text"] = collapse_equal(ref_texts)
        if any(ref_audio_b64 is not None for ref_audio_b64 in ref_audio_b64s):
            if not all(ref_audio_b64 is not None for ref_audio_b64 in ref_audio_b64s):
                raise ValueError("All clone tasks in a batch must include ref_audio_b64")
            kwargs.pop("ref_text", None)
            prompts_by_payload: dict[tuple[str, str], Any] = {}
            voice_clone_prompts: list[Any] = []
            for ref_audio_b64, ref_text in zip(ref_audio_b64s, ref_texts):
                prompt_key = (ref_audio_b64, ref_text or "")
                prompt = prompts_by_payload.get(prompt_key)
                if prompt is None:
                    prompt = self._clone_prompt_from_b64(ref_audio_b64, ref_text)
                    prompts_by_payload[prompt_key] = prompt
                voice_clone_prompts.append(prompt)
            kwargs["voice_clone_prompt"] = voice_clone_prompts
            voice_clone_prompts_for_width = voice_clone_prompts
        elif any(ref_audio is not None for ref_audio in ref_audios):
            kwargs["ref_audio"] = collapse_equal(ref_audios)

        split_results = self._maybe_split_single_chunk_batch(
            tasks,
            voice_clone_prompts_for_width,
            stream_idx,
        )
        if split_results is not None:
            return split_results

        gen_config = kwargs["generation_config"]
        num_steps = [self._task_num_step(task) for task in tasks]
        if not any(ref_audio is not None for ref_audio in ref_audios):
            stream = self.streams[stream_idx % len(self.streams)]
            start = time.perf_counter()
            if stream is None:
                with torch.inference_mode():
                    tokens = self._generate_tokens(
                        texts=texts,
                        language=languages,
                        instruct=instructs,
                        speed=speeds,
                        durations=durations,
                        voice_clone_prompts=voice_clone_prompts_for_width,
                        num_steps=num_steps,
                    )
                if str(self.cfg.device).startswith("cuda"):
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(stream):
                    with torch.inference_mode():
                        tokens = self._generate_tokens(
                            texts=texts,
                            language=languages,
                            instruct=instructs,
                            speed=speeds,
                            durations=durations,
                            voice_clone_prompts=voice_clone_prompts_for_width,
                            num_steps=num_steps,
                        )
                stream.synchronize()
            generate_elapsed = time.perf_counter() - start
            out: list[dict[str, Any]] = []
            for task, token, prompt in zip(
                tasks,
                tokens,
                voice_clone_prompts_for_width or [None] * len(tasks),
            ):
                decode_started_at = time.perf_counter()
                audio = self.model._decode_and_post_process(  # noqa: SLF001
                    token.detach(),
                    prompt.ref_rms if prompt is not None else None,
                    self.generation_config(),
                )
                self._profile_add("decode", time.perf_counter() - decode_started_at)
                out.append(
                    self._audio_result(
                        task,
                        task.seq,
                        audio,
                        generate_elapsed,
                        task.duration,
                    )
                )
            return out

        if len(set(num_steps)) == 1:
            gen_config.num_step = num_steps[0]

        stream = self.streams[stream_idx % len(self.streams)]
        start = time.perf_counter()
        if stream is None:
            audios = self.model.generate(**kwargs)
            if str(self.cfg.device).startswith("cuda"):
                torch.cuda.synchronize()
        else:
            with torch.cuda.stream(stream):
                audios = self.model.generate(**kwargs)
            stream.synchronize()
        elapsed = time.perf_counter() - start

        if len(audios) != len(tasks):
            raise RuntimeError(f"model returned {len(audios)} audios for {len(tasks)} tasks")

        out: list[dict[str, Any]] = []
        for task, audio in zip(tasks, audios):
            out.append(self._audio_result(task, task.seq, audio, elapsed, task.duration))
        return out

    def _maybe_split_single_chunk_batch(
        self,
        tasks: list[InferTask],
        voice_clone_prompts: list[Any] | None,
        stream_idx: int,
    ) -> list[dict[str, Any]] | None:
        if len(tasks) <= 1:
            return None
        gen_config = self.generation_config()
        widths: list[int] = []
        for idx, task in enumerate(tasks):
            prompt = voice_clone_prompts[idx] if voice_clone_prompts is not None else None
            widths.append(
                self._estimate_chunk_job_width(
                    ChunkJob(task, task.seq, task.text, task.duration, prompt),
                    gen_config,
                )
            )
        limit = self._graph_business_batch_limit(max(widths))
        if limit >= len(tasks):
            return None

        out: list[dict[str, Any]] = []
        for start in range(0, len(tasks), max(1, limit)):
            out.extend(self._generate_single_chunk_batch(tasks[start : start + limit], stream_idx))
        return out

    def _generate_batch_with_chunk_continuity(
        self,
        tasks: list[InferTask],
        stream_idx: int,
    ) -> list[list[dict[str, Any]]]:
        stream = self.streams[stream_idx % len(self.streams)]
        start = time.perf_counter()
        if stream is None:
            with torch.inference_mode():
                results = self._generate_mixed_chunk_batch(tasks)
            if str(self.cfg.device).startswith("cuda"):
                torch.cuda.synchronize()
        else:
            with torch.cuda.stream(stream):
                with torch.inference_mode():
                    results = self._generate_mixed_chunk_batch(tasks)
            stream.synchronize()
        elapsed = time.perf_counter() - start
        for request_results in results:
            for result in request_results:
                result["batch_elapsed_s"] = elapsed
        return results

    def _generate_mixed_chunk_batch(self, tasks: list[InferTask]) -> list[list[dict[str, Any]]]:
        if any(
            task.chunk_mode in {"sequential", "none"} and len(self._task_chunks(task)) > 1
            for task in tasks
        ):
            return self._generate_sequential_wave_chunks(tasks)
        return self._generate_concurrent_request_chunks(tasks)

    def _generate_sequential_wave_chunks(self, tasks: list[InferTask]) -> list[list[dict[str, Any]]]:
        chunks_by_task = [self._task_chunks(task) for task in tasks]
        durations_by_task = [
            self._task_chunk_durations(task, len(chunks))
            for task, chunks in zip(tasks, chunks_by_task)
        ]
        base_prompts = [self._base_voice_prompt(task) for task in tasks]
        results: list[list[dict[str, Any] | None]] = [
            [None] * len(chunks) for chunks in chunks_by_task
        ]
        tokens_by_task: list[list[torch.Tensor | None]] = [
            [None] * len(chunks) for chunks in chunks_by_task
        ]
        previous_texts: list[str | None] = [None] * len(tasks)
        previous_tokens: list[torch.Tensor | None] = [None] * len(tasks)
        max_chunks = max((len(chunks) for chunks in chunks_by_task), default=0)

        for seq in range(max_chunks):
            jobs: list[ChunkJob] = []
            for task_index, task in enumerate(tasks):
                chunks = chunks_by_task[task_index]
                if seq >= len(chunks):
                    continue
                prompt = self._continuity_prompt(
                    base_prompt=base_prompts[task_index],
                    previous_text=previous_texts[task_index],
                    previous_tokens=previous_tokens[task_index],
                )
                jobs.append(
                    ChunkJob(
                        task,
                        seq,
                        chunks[seq],
                        durations_by_task[task_index][seq],
                        prompt,
                        prompt_affects_duration=self._prompt_affects_duration(
                            task,
                            prompt,
                        ),
                    )
                )

            self._run_chunk_jobs(jobs, results, tokens_by_task, tasks)

            for task_index, chunks in enumerate(chunks_by_task):
                if seq >= len(chunks):
                    continue
                token = tokens_by_task[task_index][seq]
                if token is None:
                    raise RuntimeError(
                        f"missing sequential chunk token for request "
                        f"{tasks[task_index].request_id} seq={seq}"
                    )
                previous_texts[task_index] = chunks[seq]
                previous_tokens[task_index] = token

        final_results: list[list[dict[str, Any]]] = []
        for task, task_results in zip(tasks, results):
            if any(item is None for item in task_results):
                missing = [idx for idx, item in enumerate(task_results) if item is None]
                raise RuntimeError(f"missing chunk results for request {task.request_id}: {missing}")
            final_results.append([item for item in task_results if item is not None])
        return final_results

    def _generate_concurrent_request_chunks(self, tasks: list[InferTask]) -> list[list[dict[str, Any]]]:
        chunks_by_task = [self._task_chunks(task) for task in tasks]
        durations_by_task = [
            self._task_chunk_durations(task, len(chunks))
            for task, chunks in zip(tasks, chunks_by_task)
        ]
        base_prompts = [self._base_voice_prompt(task) for task in tasks]
        results: list[list[dict[str, Any] | None]] = [
            [None] * len(chunks) for chunks in chunks_by_task
        ]
        tokens_by_task: list[list[torch.Tensor | None]] = [
            [None] * len(chunks) for chunks in chunks_by_task
        ]

        initial_jobs: list[ChunkJob] = []
        deferred_task_indices: list[int] = []
        for task_index, (task, chunks, durations, base_prompt) in enumerate(
            zip(tasks, chunks_by_task, durations_by_task, base_prompts)
        ):
            if len(chunks) > 1 and task.mode in {"auto", "design"} and base_prompt is None:
                initial_jobs.append(ChunkJob(task, 0, chunks[0], durations[0], None))
                deferred_task_indices.append(task_index)
                continue
            for seq, chunk in enumerate(chunks):
                initial_jobs.append(
                    ChunkJob(task, seq, chunk, durations[seq], base_prompt)
                )

        self._run_chunk_jobs(initial_jobs, results, tokens_by_task, tasks)

        deferred_jobs: list[ChunkJob] = []
        for task_index in deferred_task_indices:
            task = tasks[task_index]
            chunks = chunks_by_task[task_index]
            durations = durations_by_task[task_index]
            first_token = tokens_by_task[task_index][0]
            if first_token is None:
                raise RuntimeError(f"missing first chunk token for request {task.request_id}")
            prompt = self._continuity_prompt(
                base_prompt=None,
                previous_text=chunks[0],
                previous_tokens=first_token,
            )
            for seq in range(1, len(chunks)):
                deferred_jobs.append(
                    ChunkJob(
                        task,
                        seq,
                        chunks[seq],
                        durations[seq],
                        prompt,
                        prompt_affects_duration=False,
                    )
                )

        self._run_chunk_jobs(deferred_jobs, results, tokens_by_task, tasks)

        final_results: list[list[dict[str, Any]]] = []
        for task, task_results in zip(tasks, results):
            if any(item is None for item in task_results):
                missing = [idx for idx, item in enumerate(task_results) if item is None]
                raise RuntimeError(f"missing chunk results for request {task.request_id}: {missing}")
            final_results.append([item for item in task_results if item is not None])
        return final_results

    def _run_chunk_jobs(
        self,
        jobs: list[ChunkJob],
        results: list[list[dict[str, Any] | None]],
        tokens_by_task: list[list[torch.Tensor | None]],
        tasks: list[InferTask],
    ) -> None:
        if not jobs:
            return
        task_indices = {id(task): idx for idx, task in enumerate(tasks)}
        gen_config = self.generation_config()
        grouped: OrderedDict[tuple[bool, int], list[tuple[ChunkJob, int]]] = OrderedDict()
        for job in jobs:
            width = self._estimate_chunk_job_width(job, gen_config)
            key = (job.prompt is not None, self._width_bucket_for_job(width))
            grouped.setdefault(key, []).append((job, width))

        for (_, _), grouped_jobs in grouped.items():
            for batch in self._chunk_job_microbatches(grouped_jobs):
                prompt_list = (
                    [job.prompt for job, _ in batch]
                    if batch[0][0].prompt is not None
                    else None
                )
                batch_jobs = [job for job, _ in batch]
                tokens = self._generate_tokens(
                    texts=[job.text for job in batch_jobs],
                    language=[job.task.language for job in batch_jobs],
                    instruct=[job.task.instruct for job in batch_jobs],
                    speed=[job.task.speed for job in batch_jobs],
                    durations=[self._generation_duration(job) for job in batch_jobs],
                    voice_clone_prompts=prompt_list,
                    num_steps=[self._task_num_step(job.task) for job in batch_jobs],
                )
                for job, token in zip(batch_jobs, tokens):
                    task_index = task_indices[id(job.task)]
                    tokens_by_task[task_index][job.seq] = token.detach()
                    decode_started_at = time.perf_counter()
                    audio = self.model._decode_and_post_process(  # noqa: SLF001
                        token,
                        job.prompt.ref_rms if job.prompt is not None else None,
                        gen_config,
                        apply_edge_fade_pad=False,
                    )
                    self._profile_add("decode", time.perf_counter() - decode_started_at)
                    results[task_index][job.seq] = self._audio_result(
                        job.task,
                        job.seq,
                        audio,
                        0.0,
                        self._generation_duration(job),
                        apply_postprocess=False,
                    )

    def _generation_duration(self, job: ChunkJob) -> float | None:
        if job.duration is not None:
            return job.duration
        if job.prompt is None or job.prompt_affects_duration:
            return None
        if job.task.mode not in {"auto", "design"}:
            return None

        target_tokens = self.model._estimate_target_tokens(  # noqa: SLF001
            job.text,
            None,
            None,
            speed=job.task.speed,
        )
        frame_rate = float(getattr(self.model.audio_tokenizer.config, "frame_rate", 0.0))
        if frame_rate <= 0.0:
            return None
        return max(1, int(target_tokens)) / frame_rate

    def _task_num_step(self, task: InferTask) -> int:
        return max(1, int(task.num_step or self.cfg.default_num_step))

    def _estimate_chunk_job_width(self, job: ChunkJob, gen_config: Any) -> int:
        cache_key = self._chunk_job_width_cache_key(job, gen_config)
        cached = self.chunk_job_width_cache.get(cache_key)
        if cached is not None:
            self.chunk_job_width_cache.move_to_end(cache_key)
            self.chunk_job_width_cache_hits += 1
            return cached

        self.chunk_job_width_cache_misses += 1
        started_at = time.perf_counter()
        generation_duration = self._generation_duration(job)
        ref_text = job.prompt.ref_text if job.prompt is not None else None
        ref_audio_tokens = job.prompt.ref_audio_tokens if job.prompt is not None else None
        ref_audio_width = int(ref_audio_tokens.size(-1)) if ref_audio_tokens is not None else 0

        if generation_duration is not None:
            frame_rate = float(getattr(self.model.audio_tokenizer.config, "frame_rate", 0.0))
            target_tokens = max(1, int(generation_duration * frame_rate)) if frame_rate > 0 else 1
        else:
            target_tokens = self.model._estimate_target_tokens(  # noqa: SLF001
                job.text,
                ref_text,
                ref_audio_width or None,
                speed=job.task.speed,
            )

        lang = _resolve_language(job.task.language)
        instruct = job.task.instruct
        if instruct is not None:
            instruct = _resolve_instruct(
                instruct,
                use_zh=bool(job.text and _ZH_RE.search(job.text)),
            )

        style_text = ""
        if gen_config.denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        style_text += f"<|lang_start|>{lang if lang else 'None'}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct if instruct else 'None'}<|instruct_end|>"
        style_width = len(self.model.text_tokenizer(style_text).input_ids)

        full_text = _combine_text(ref_text=ref_text, text=job.text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_width = int(
            _tokenize_with_nonverbal_tags(
                wrapped_text,
                self.model.text_tokenizer,
            ).size(-1)
        )
        width = style_width + text_width + ref_audio_width + int(target_tokens)
        self._profile_add("width_estimate", time.perf_counter() - started_at)
        self.chunk_job_width_cache[cache_key] = width
        while len(self.chunk_job_width_cache) > 4096:
            self.chunk_job_width_cache.popitem(last=False)
        return width

    def _chunk_job_width_cache_key(self, job: ChunkJob, gen_config: Any) -> tuple[Any, ...]:
        prompt = job.prompt
        prompt_key: tuple[Any, ...]
        if prompt is None:
            prompt_key = (None,)
        else:
            prompt_key = (
                prompt.ref_text,
                int(prompt.ref_audio_tokens.size(-1)),
                str(prompt.ref_audio_tokens.device),
            )
        return (
            job.text,
            job.task.language,
            job.task.instruct,
            job.task.speed,
            self._generation_duration(job),
            job.prompt_affects_duration,
            bool(gen_config.denoise),
            prompt_key,
        )

    def _width_bucket_for_job(self, width: int) -> int:
        for bucket in (64, 128, 160, 192, 256, 512, 640, 768, self.cfg.cuda_graph_max_width):
            if bucket > 0 and width <= bucket:
                return bucket
        return width

    def _chunk_job_microbatches(
        self,
        jobs: list[tuple[ChunkJob, int]],
    ) -> list[list[tuple[ChunkJob, int]]]:
        batches: list[list[tuple[ChunkJob, int]]] = []
        index = 0
        while index < len(jobs):
            width = max(width for _, width in jobs[index : index + self.cfg.batch_size])
            limit = min(self.cfg.batch_size, self._graph_business_batch_limit(width))
            limit = max(1, limit)
            batches.append(jobs[index : index + limit])
            index += limit
        return batches

    def _graph_business_batch_limit(self, width: int) -> int:
        stats = self.graph_cache_stats()
        shapes = stats.get("shapes") if isinstance(stats, dict) else None
        if not isinstance(shapes, list):
            return self.cfg.batch_size
        candidates: list[int] = []
        for shape in shapes:
            if (
                isinstance(shape, list)
                and len(shape) == 3
                and int(shape[2]) >= width
            ):
                candidates.append(max(1, int(shape[0]) // 2))
        return max(candidates) if candidates else self.cfg.batch_size

    def _task_chunks(self, task: InferTask) -> list[str]:
        chunks = [chunk.strip() for chunk in task.chunks if chunk.strip()]
        return chunks or [task.text.strip()]

    def _task_chunk_durations(self, task: InferTask, count: int) -> list[float | None]:
        return [
            task.chunk_durations[idx] if idx < len(task.chunk_durations) else task.duration
            for idx in range(count)
        ]

    def _generate_request_chunks(self, task: InferTask) -> list[dict[str, Any]]:
        chunks = self._task_chunks(task)
        durations = self._task_chunk_durations(task, len(chunks))
        if len(chunks) == 1:
            prompt = (
                self._clone_prompt_from_b64(task.ref_audio_b64, task.ref_text)
                if task.ref_audio_b64
                else None
            )
            audio = self.model.generate(
                text=chunks[0],
                language=task.language,
                ref_audio=None if prompt is not None else task.ref_audio,
                ref_text=None if prompt is not None else task.ref_text,
                voice_clone_prompt=prompt,
                instruct=task.instruct,
                speed=None if durations[0] is not None else task.speed,
                duration=durations[0],
                generation_config=self.generation_config(),
            )
            return [self._audio_result(task, 0, audio[0], 0.0, durations[0])]

        base_prompt = self._base_voice_prompt(task)
        return self._generate_chunks_with_sequential_context(
            task,
            chunks,
            durations,
            base_prompt,
        )

    def _base_voice_prompt(self, task: InferTask) -> VoiceClonePrompt | None:
        if task.ref_audio_b64:
            return self._clone_prompt_from_b64(task.ref_audio_b64, task.ref_text)
        if task.ref_audio:
            return self.model.create_voice_clone_prompt(
                ref_audio=task.ref_audio,
                ref_text=task.ref_text,
                preprocess_prompt=self.generation_config().preprocess_prompt,
            )
        return None

    def _generate_chunks_with_sequential_context(
        self,
        task: InferTask,
        chunks: list[str],
        durations: list[float | None],
        base_prompt: VoiceClonePrompt | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        previous_text: str | None = None
        previous_tokens: torch.Tensor | None = None

        for seq, chunk in enumerate(chunks):
            prompt = self._continuity_prompt(
                base_prompt=base_prompt,
                previous_text=previous_text,
                previous_tokens=previous_tokens,
            )
            job = ChunkJob(
                task,
                seq,
                chunk,
                durations[seq],
                prompt,
                prompt_affects_duration=self._prompt_affects_duration(task, prompt),
            )
            generation_duration = self._generation_duration(job)
            token = self._generate_tokens(
                texts=[chunk],
                language=task.language,
                instruct=task.instruct,
                speed=task.speed,
                durations=[generation_duration],
                voice_clone_prompts=[prompt] if prompt is not None else None,
            )[0]
            results.append(
                self._audio_result(
                    task,
                    seq,
                    self.model._decode_and_post_process(  # noqa: SLF001
                        token,
                        prompt.ref_rms if prompt is not None else None,
                        self.generation_config(),
                        apply_edge_fade_pad=False,
                    ),
                    0.0,
                    generation_duration,
                    apply_postprocess=False,
                )
            )
            previous_text = chunk
            previous_tokens = token.detach()
        return results

    def _continuity_prompt(
        self,
        *,
        base_prompt: VoiceClonePrompt | None,
        previous_text: str | None,
        previous_tokens: torch.Tensor | None,
    ) -> VoiceClonePrompt | None:
        if previous_tokens is None or previous_text is None:
            return base_prompt

        previous_tokens = self._continuity_audio_tokens(previous_tokens)
        previous_text = self._continuity_text(previous_text)
        if base_prompt is None:
            return VoiceClonePrompt(
                ref_audio_tokens=previous_tokens,
                ref_text=previous_text,
                ref_rms=0.1,
            )

        base_tokens = base_prompt.ref_audio_tokens.detach()
        return VoiceClonePrompt(
            ref_audio_tokens=torch.cat([base_tokens, previous_tokens], dim=-1),
            ref_text=f"{base_prompt.ref_text} {previous_text}".strip(),
            ref_rms=base_prompt.ref_rms,
        )

    def _prompt_affects_duration(
        self,
        task: InferTask,
        prompt: VoiceClonePrompt | None,
    ) -> bool:
        if prompt is None:
            return True
        return task.mode not in {"auto", "design"}

    def _continuity_audio_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.detach()
        max_tokens = self.cfg.max_continuity_audio_tokens
        if max_tokens > 0 and tokens.size(-1) > max_tokens:
            return tokens[..., -max_tokens:]
        return tokens

    def _continuity_text(self, text: str) -> str:
        max_words = self.cfg.max_continuity_text_words
        if max_words <= 0:
            return text
        return truncate_text_by_word_count(text, max_words)

    def _generate_tokens(
        self,
        *,
        texts: list[str],
        language: str | list[str | None] | None,
        instruct: str | list[str | None] | None,
        speed: float | list[float],
        durations: list[float | None],
        voice_clone_prompts: list[Any] | None,
        num_steps: list[int] | None = None,
    ) -> list[torch.Tensor]:
        gen_config = self.generation_config()
        if num_steps:
            gen_config.num_step = max(max(1, int(step)) for step in num_steps)
        has_duration = any(duration is not None for duration in durations)
        preprocess_started_at = time.perf_counter()
        full_task = self.model._preprocess_all(  # noqa: SLF001
            text=texts,
            language=language,
            voice_clone_prompt=voice_clone_prompts,
            instruct=instruct,
            preprocess_prompt=gen_config.preprocess_prompt,
            speed=None if has_duration else speed,
            duration=durations if has_duration else None,
        )
        self._profile_add("preprocess", time.perf_counter() - preprocess_started_at)
        if num_steps:
            full_task.num_steps = [max(1, int(step)) for step in num_steps]
        iterative_started_at = time.perf_counter()
        tokens = self.model._generate_iterative(full_task, gen_config)  # noqa: SLF001
        self._profile_add("iterative_launch", time.perf_counter() - iterative_started_at)
        return tokens

    def _audio_result(
        self,
        task: InferTask,
        seq: int,
        audio: Any,
        elapsed: float,
        duration: float | None,
        *,
        apply_postprocess: bool = True,
    ) -> dict[str, Any]:
        numpy_started_at = time.perf_counter()
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._profile_add("audio_numpy", time.perf_counter() - numpy_started_at)
        empty_audio_fallback = False
        if arr.size == 0:
            fallback_s = duration or 0.25
            arr = np.zeros(max(1, int(self.cfg.sample_rate * fallback_s)), dtype=np.float32)
            empty_audio_fallback = True
            logger.warning(
                "Model returned empty audio; using %.3fs silence fallback task_id=%s seq=%d",
                fallback_s,
                task.task_id,
                seq,
            )
        min_postprocess_samples = int(self.cfg.sample_rate * 1.0)
        if (
            apply_postprocess
            and
            self.cfg.postprocess_output
            and arr.size >= min_postprocess_samples
            and hasattr(self.model, "_post_process_audio")
        ):
            try:
                postprocess_started_at = time.perf_counter()
                processed = self.model._post_process_audio(  # noqa: SLF001
                    arr,
                    postprocess_output=True,
                    ref_rms=None,
                )
                self._profile_add("postprocess", time.perf_counter() - postprocess_started_at)
                processed_arr = np.asarray(processed, dtype=np.float32).reshape(-1)
                if processed_arr.size:
                    arr = processed_arr
            except Exception as exc:
                logger.warning(
                    "Postprocess failed; returning unprocessed audio (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
        pcm_started_at = time.perf_counter()
        pcm = float_to_pcm16(arr)
        self._profile_add("pcm16", time.perf_counter() - pcm_started_at)
        b64_started_at = time.perf_counter()
        pcm_b64 = base64.b64encode(pcm).decode("ascii")
        self._profile_add("base64_encode", time.perf_counter() - b64_started_at)
        return {
            "request_id": task.request_id,
            "task_id": f"{task.request_id}:{seq}",
            "seq": seq,
            "pcm_b64": pcm_b64,
            "sample_rate": self.cfg.sample_rate,
            "pcm_bytes": len(pcm),
            "batch_elapsed_s": elapsed,
            "empty_audio_fallback": empty_audio_fallback,
        }

    def graph_cache_stats(self) -> dict[str, Any]:
        if self.runner is not None and hasattr(self.runner, "graph_cache_stats"):
            return self.runner.graph_cache_stats()
        return {"enabled": False, "entries": 0}

    def clone_prompt_cache_stats(self) -> dict[str, Any]:
        with self.clone_prompt_cache_lock:
            return {
                "enabled": self.cfg.max_clone_audio_prompt_cache > 0,
                "max_entries": self.cfg.max_clone_audio_prompt_cache,
                "entries": len(self.clone_prompt_cache),
                "hits": self.clone_prompt_cache_hits,
                "misses": self.clone_prompt_cache_misses,
                "evictions": self.clone_prompt_cache_evictions,
                "shared_dir": self.cfg.clone_prompt_shared_cache_dir,
                "shared_hits": self.shared_clone_prompt_cache_hits,
                "shared_stores": self.shared_clone_prompt_cache_stores,
                "shared_load_errors": self.shared_clone_prompt_cache_load_errors,
            }

    def chunk_job_width_cache_stats(self) -> dict[str, Any]:
        return {
            "entries": len(self.chunk_job_width_cache),
            "hits": self.chunk_job_width_cache_hits,
            "misses": self.chunk_job_width_cache_misses,
        }


class Inferer:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.backend = TritonBackend(cfg)
        self.pending: list[QueuedTask] = []
        self.ready_batches: deque[list[QueuedTask]] = deque()
        self.dispatch_worker_count = 1
        self.ready_batch_queue_limit = 1
        self.condition = asyncio.Condition()
        self.running_batches = 0
        self.total_batches = 0
        self.total_tasks = 0
        self.total_errors = 0
        self.total_batch_elapsed_s = 0.0
        self.total_queue_wait_ms = 0.0
        self.total_pcm_bytes = 0
        self.total_empty_audio_fallbacks = 0
        self.max_batch_size_seen = 0
        self.last_batch: dict[str, Any] | None = None
        self.started_at = time.monotonic()
        self.metrics_writer: SharedMetricsWriter | None = (
            SharedMetricsWriter(cfg.metrics_shm_path, cfg.metrics_shm_size)
            if cfg.metrics_shm_path
            else None
        )

    async def start(self) -> None:
        await asyncio.to_thread(self.backend.load)
        graph_stats = self.backend.graph_cache_stats()
        effective_batch_size = graph_stats.get("max_business_batch_size")
        if isinstance(effective_batch_size, int) and effective_batch_size < self.cfg.batch_size:
            logger.warning(
                "Reducing scheduler batch_size from requested %d to CUDA Graph effective %d",
                self.cfg.batch_size,
                effective_batch_size,
            )
            self.cfg.batch_size = effective_batch_size
        self.write_metrics_snapshot()
        for idx in range(self.dispatch_worker_count):
            asyncio.create_task(self.dispatch_worker(idx))
        asyncio.create_task(self.scheduler())
        asyncio.create_task(self.metrics_snapshot_loop())

    async def metrics_snapshot_loop(self) -> None:
        while True:
            self.write_metrics_snapshot()
            await asyncio.sleep(self.cfg.metrics_snapshot_interval_s)

    def write_metrics_snapshot(self) -> None:
        if self.metrics_writer is None:
            return
        try:
            self.metrics_writer.write(self.health())
        except Exception:
            logger.exception("Failed to write metrics shared memory snapshot")

    async def enqueue_request(self, req: InferRequest) -> list[asyncio.Future[dict[str, Any]]]:
        chunks = [chunk.strip() for chunk in req.chunks if chunk.strip()] or [req.input.strip()]
        if not chunks:
            raise ValueError("input is empty after text normalization")

        futures: list[asyncio.Future[dict[str, Any]]] = [
            asyncio.get_running_loop().create_future() for _ in chunks
        ]
        now_id = req.request_id or uuid.uuid4().hex
        durations = [
            req.chunk_durations[seq] if seq < len(req.chunk_durations) else req.duration
            for seq in range(len(chunks))
        ]
        task = InferTask(
            request_id=now_id,
            task_id=now_id,
            seq=0,
            text=chunks[0],
            chunks=chunks,
            chunk_durations=durations,
            mode=req.mode,
            instruct=req.instruct,
            speed=req.speed,
            duration=req.duration,
            language=req.language,
            chunk_mode=req.chunk_mode,
            num_step=req.num_step,
            ref_text=req.ref_text,
            ref_audio=req.ref_audio,
            ref_audio_b64=req.ref_audio_b64,
            extra_fields=req.extra_fields,
        )
        async with self.condition:
            self.pending.append(QueuedTask(task=task, futures=futures))
            self.condition.notify_all()
        return futures

    def group_key(self, queued: QueuedTask) -> tuple[Any, ...]:
        task = queued.task
        if task.mode in {"auto", "design"}:
            return ("voice",)
        return (
            task.mode,
        )

    def scheduler_key(self, queued: QueuedTask) -> tuple[Any, ...]:
        return (
            *self.group_key(queued),
            self._task_num_step(queued.task),
        )

    def _task_num_step(self, task: InferTask) -> int:
        return max(1, int(task.num_step or self.cfg.default_num_step))

    def batch_class(self, queued: QueuedTask) -> str:
        if queued.chunk_count <= 1:
            return "single"
        if queued.task.chunk_mode in {"sequential", "none"}:
            return "multi_sequential"
        return "multi_concurrent"

    async def scheduler(self) -> None:
        while True:
            async with self.condition:
                while True:
                    if self.pending and len(self.ready_batches) < self.ready_batch_queue_limit:
                        batch = self.pop_next_batch_locked(time.monotonic())
                        if batch is not None:
                            self.ready_batches.append(batch)
                            self.condition.notify_all()
                            continue

                    timeout = self.scheduler_wait_timeout_locked(time.monotonic())
                    try:
                        if timeout is None:
                            await self.condition.wait()
                        else:
                            await asyncio.wait_for(self.condition.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        continue

    def pending_buckets_locked(self) -> OrderedDict[tuple[Any, ...], list[QueuedTask]]:
        buckets: OrderedDict[tuple[Any, ...], list[QueuedTask]] = OrderedDict()
        for item in self.pending:
            buckets.setdefault(self.scheduler_key(item), []).append(item)
        return buckets

    def scheduler_wait_timeout_locked(self, now: float) -> float | None:
        if not self.pending or len(self.ready_batches) >= self.ready_batch_queue_limit:
            return None
        oldest = min(item.enqueued_at for item in self.pending)
        age_ms = (now - oldest) * 1000.0
        return max(0.001, (self.cfg.batch_wait_ms - age_ms) / 1000.0)

    def pop_next_batch_locked(self, now: float) -> list[QueuedTask] | None:
        if not self.pending:
            return None

        oldest_item = min(self.pending, key=lambda item: item.enqueued_at)
        oldest_age_ms = (now - oldest_item.enqueued_at) * 1000.0
        if oldest_age_ms >= self.cfg.batch_wait_ms:
            return self.pop_batch_for_group_locked(
                self.group_key(oldest_item),
                self.scheduler_key(oldest_item),
            )

        buckets = self.pending_buckets_locked()
        full_buckets = [
            (key, items)
            for key, items in buckets.items()
            if self.batchable_count_for_key_locked(key) >= self.batch_size_for_key_locked(key)
        ]
        if not full_buckets:
            return None

        key, _ = max(
            full_buckets,
            key=lambda item: (
                self.batchable_count_for_key_locked(item[0]),
                -item[1][0].enqueued_at,
            ),
        )
        return self.pop_batch_for_key_locked(key)

    def batch_size_for_key_locked(self, key: tuple[Any, ...]) -> int:
        return self.cfg.batch_size

    def batchable_count_for_key_locked(self, key: tuple[Any, ...]) -> int:
        count = 0
        for item in self.pending:
            if self.scheduler_key(item) != key:
                continue
            count += 1
        return count

    def pop_batch_for_key_locked(self, key: tuple[Any, ...]) -> list[QueuedTask]:
        batch: list[QueuedTask] = []
        rest: list[QueuedTask] = []
        limit = self.cfg.batch_size
        for item in self.pending:
            if (
                len(batch) < limit
                and self.scheduler_key(item) == key
            ):
                batch.append(item)
            else:
                rest.append(item)
        self.pending = rest
        return batch

    def pop_batch_for_group_locked(
        self,
        group_key: tuple[Any, ...],
        preferred_scheduler_key: tuple[Any, ...],
    ) -> list[QueuedTask]:
        batch: list[QueuedTask] = []
        rest: list[QueuedTask] = []
        limit = self.cfg.batch_size

        for item in self.pending:
            if (
                len(batch) < limit
                and self.group_key(item) == group_key
                and self.scheduler_key(item) == preferred_scheduler_key
            ):
                batch.append(item)
            else:
                rest.append(item)

        if len(batch) < limit:
            new_rest: list[QueuedTask] = []
            for item in rest:
                if len(batch) < limit and self.group_key(item) == group_key:
                    batch.append(item)
                else:
                    new_rest.append(item)
            rest = new_rest

        self.pending = rest
        return batch

    async def dispatch_worker(self, stream_idx: int) -> None:
        while True:
            async with self.condition:
                await self.condition.wait_for(lambda: bool(self.ready_batches))
                batch = self.ready_batches.popleft()
                self.condition.notify_all()

            await self.dispatch_batch(batch, stream_idx)

    async def dispatch_batch(self, batch: list[QueuedTask], stream_idx: int) -> None:
        self.running_batches += 1
        batch_started_at = time.perf_counter()
        queue_wait_ms = (time.monotonic() - min(item.enqueued_at for item in batch)) * 1000.0
        try:
            tasks = [item.task for item in batch]
            chunk_count = sum(item.chunk_count for item in batch)
            num_step_counts: dict[int, int] = {}
            for task in tasks:
                step = self._task_num_step(task)
                num_step_counts[step] = num_step_counts.get(step, 0) + 1
            logger.info(
                "Dispatching batch size=%d chunks=%d mode=%s key=%s num_steps=%s",
                len(tasks),
                chunk_count,
                tasks[0].mode if tasks else None,
                self.group_key(batch[0]) if batch else None,
                num_step_counts,
            )
            results_by_item = await asyncio.to_thread(self.backend.generate_batch, tasks, stream_idx)
            flat_results = [result for item_results in results_by_item for result in item_results]
            dispatch_elapsed_s = time.perf_counter() - batch_started_at
            pcm_bytes = sum(int(result.get("pcm_bytes", 0)) for result in flat_results)
            self.total_batches += 1
            self.total_tasks += chunk_count
            self.total_batch_elapsed_s += dispatch_elapsed_s
            self.total_queue_wait_ms += queue_wait_ms
            self.total_pcm_bytes += pcm_bytes
            self.total_empty_audio_fallbacks += sum(
                1 for result in flat_results if result.get("empty_audio_fallback")
            )
            self.max_batch_size_seen = max(self.max_batch_size_seen, chunk_count)
            self.last_batch = {
                "size": len(batch),
                "chunks": chunk_count,
                "mode": tasks[0].mode if tasks else None,
                "key": list(self.group_key(batch[0])) if batch else None,
                "num_steps": {str(k): v for k, v in sorted(num_step_counts.items())},
                "queue_wait_ms": round(queue_wait_ms, 1),
                "elapsed_s": round(dispatch_elapsed_s, 3),
                "pcm_bytes": pcm_bytes,
                "stream_idx": stream_idx,
            }
            logger.info(
                "Completed batch size=%d chunks=%d mode=%s num_steps=%s elapsed=%.3fs queue_wait=%.1fms pcm_bytes=%d",
                len(batch),
                chunk_count,
                tasks[0].mode if tasks else None,
                num_step_counts,
                dispatch_elapsed_s,
                queue_wait_ms,
                pcm_bytes,
            )
            self.write_metrics_snapshot()
            for item, item_results in zip(batch, results_by_item):
                for fut, result in zip(item.futures, item_results):
                    if not fut.done():
                        fut.set_result(result)
        except Exception as exc:
            self.total_errors += 1
            logger.exception("Batch failed")
            self.write_metrics_snapshot()
            for item in batch:
                for fut in item.futures:
                    if not fut.done():
                        fut.set_exception(exc)
        finally:
            self.running_batches -= 1
            self.write_metrics_snapshot()

    def queued_backlog_stats(self, now: float) -> dict[str, Any]:
        queued_tasks = sum(item.chunk_count for batch in self.ready_batches for item in batch)
        pending_chunks = sum(item.chunk_count for item in self.pending)
        queued_oldest = min(
            (item.enqueued_at for batch in self.ready_batches for item in batch),
            default=None,
        )
        pending_oldest = min((item.enqueued_at for item in self.pending), default=None)
        backlog_oldest = min(
            (ts for ts in (queued_oldest, pending_oldest) if ts is not None),
            default=None,
        )

        return {
            "queued_batches": len(self.ready_batches),
            "queued_tasks": queued_tasks,
            "pending_chunks": pending_chunks,
            "queued_batch_limit": self.ready_batch_queue_limit,
            "oldest_queued_task_age_ms": round((now - queued_oldest) * 1000.0, 1)
            if queued_oldest is not None
            else 0.0,
            "oldest_pending_task_age_ms": round((now - pending_oldest) * 1000.0, 1)
            if pending_oldest is not None
            else 0.0,
            "oldest_backlog_task_age_ms": round((now - backlog_oldest) * 1000.0, 1)
            if backlog_oldest is not None
            else 0.0,
        }

    def health(self) -> dict[str, Any]:
        now = time.monotonic()
        backlog_stats = self.queued_backlog_stats(now)
        return {
            "status": "healthy",
            "ready": self.backend.model is not None,
            "inferer_name": self.cfg.inferer_name,
            "inferer_kind": self.cfg.inferer_kind,
            "uptime_s": round(now - self.started_at, 1),
            "pending_tasks": len(self.pending),
            **backlog_stats,
            "running_batches": self.running_batches,
            "total_batches": self.total_batches,
            "total_tasks": self.total_tasks,
            "total_errors": self.total_errors,
            "total_pcm_bytes": self.total_pcm_bytes,
            "total_empty_audio_fallbacks": self.total_empty_audio_fallbacks,
            "avg_batch_size": round(self.total_tasks / self.total_batches, 3)
            if self.total_batches
            else 0.0,
            "avg_batch_elapsed_s": round(self.total_batch_elapsed_s / self.total_batches, 3)
            if self.total_batches
            else 0.0,
            "avg_queue_wait_ms": round(self.total_queue_wait_ms / self.total_batches, 1)
            if self.total_batches
            else 0.0,
            "max_batch_size_seen": self.max_batch_size_seen,
            "last_batch": self.last_batch,
            "batch_size": self.cfg.batch_size,
            "default_num_step": self.cfg.default_num_step,
            "batch_wait_ms": self.cfg.batch_wait_ms,
            "dispatch_workers": self.dispatch_worker_count,
            "log_file": self.cfg.log_file,
            "pid_file": self.cfg.pid_file,
            "cuda_streams": self.cfg.cuda_streams,
            "runner_mode": self.cfg.runner_mode,
            "cuda_graph_cache": self.backend.graph_cache_stats(),
            "clone_audio_prompt_cache": self.backend.clone_prompt_cache_stats(),
            "chunk_job_width_cache": self.backend.chunk_job_width_cache_stats(),
            "profile": self.backend.profile_stats(),
        }


async def write_json(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    writer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
    await writer.drain()


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    inferer: Inferer,
) -> None:
    try:
        line = await reader.readline()
        if not line:
            return
        message = json.loads(line)
        msg_type = message.get("type")
        if msg_type == "health":
            await write_json(writer, {"type": "health", **inferer.health()})
            return
        if msg_type != "infer":
            await write_json(writer, {"type": "error", "error": f"unknown type {msg_type!r}"})
            return

        req = InferRequest.model_validate(message["request"])
        futures = await inferer.enqueue_request(req)
        await write_json(writer, {"type": "accepted", "tasks": len(futures)})
        if req.stream:
            for fut in futures:
                result = await fut
                pcm_b64 = result.pop("pcm_b64")
                max_chars = inferer.cfg.max_sse_audio_b64_chars
                parts = [
                    pcm_b64[i : i + max_chars]
                    for i in range(0, len(pcm_b64), max_chars)
                ] or [""]
                for part_idx, part in enumerate(parts):
                    await write_json(
                        writer,
                        {
                            "type": "chunk",
                            **result,
                            "part": part_idx,
                            "parts": len(parts),
                            "pcm_b64": part,
                        },
                    )
            await write_json(writer, {"type": "done", "sample_rate": inferer.cfg.sample_rate})
        else:
            results = [await fut for fut in futures]
            await write_json(
                writer,
                {
                    "type": "done",
                    "sample_rate": inferer.cfg.sample_rate,
                    "chunks": results,
                },
            )
    except Exception as exc:
        logger.exception("Client handling failed")
        await write_json(writer, {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        writer.close()
        await writer.wait_closed()


async def amain() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-run-id", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-retention-days", type=int, default=None)
    parser.add_argument("--inferer-name", default=None)
    parser.add_argument("--inferer-kind", default=None, choices=["gpu"])
    parser.add_argument("--cuda-graph-min-width", type=int, default=None)
    parser.add_argument("--cuda-graph-max-width", type=int, default=None)
    parser.add_argument("--max-clone-audio-prompt-cache", type=int, default=None)
    parser.add_argument("--default-num-step", type=int, default=None)
    args = parser.parse_args()

    cfg = Settings()
    if args.host is not None:
        cfg.infer_host = args.host
    if args.port is not None:
        cfg.infer_port = args.port
    if args.log_level is not None:
        cfg.log_level = args.log_level
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.log_run_id is not None:
        cfg.log_run_id = args.log_run_id
    if args.log_file is not None:
        cfg.log_file = args.log_file
    if args.log_retention_days is not None:
        cfg.log_retention_days = args.log_retention_days
    if args.inferer_name is not None:
        cfg.inferer_name = args.inferer_name
    if args.inferer_kind is not None:
        cfg.inferer_kind = args.inferer_kind
    if args.cuda_graph_min_width is not None:
        cfg.cuda_graph_min_width = args.cuda_graph_min_width
    if args.cuda_graph_max_width is not None:
        cfg.cuda_graph_max_width = args.cuda_graph_max_width
    if args.max_clone_audio_prompt_cache is not None:
        cfg.max_clone_audio_prompt_cache = args.max_clone_audio_prompt_cache
    if args.default_num_step is not None:
        cfg.default_num_step = args.default_num_step
    if not cfg.log_file:
        run_id = cfg.log_run_id or time.strftime("%Y%m%d-%H%M%S")
        cfg.log_file = str(Path(cfg.log_dir).expanduser() / run_id / "inferer.log")

    configure_logging(cfg.log_level, cfg.log_file, cfg.log_retention_days)
    inferer = Inferer(cfg)
    await inferer.start()
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, inferer),
        host=cfg.infer_host,
        port=cfg.infer_port,
        limit=256 * 1024 * 1024,
    )
    sockets = server.sockets or []
    port = sockets[0].getsockname()[1] if sockets else cfg.infer_port
    print(f"OMNIVOICE_INFERER_READY host={cfg.infer_host} port={port}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
