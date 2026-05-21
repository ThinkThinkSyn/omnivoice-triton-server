from __future__ import annotations

import base64
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from audio import media_type_for_format, pcm_chunks_to_format
from chunking import TextChunk, split_text_by_word_count
from config import Settings
from infer_client import InfererClient
from metrics_shm import SharedMetricsGroupReader, SharedMetricsReader
from protocol import InferRequest, SpeechRequest
from text_normalization import normalize_tts_text_for_language
from voices import resolve_mode_and_instruct


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Settings()
    app.state.cfg = cfg
    app.state.infer_client = InfererClient(cfg)
    inferers = _load_inferer_entries(cfg)
    app.state.metrics_reader = (
        SharedMetricsGroupReader(inferers, cfg.metrics_shm_size)
        if inferers
        else SharedMetricsReader(cfg.metrics_shm_path, cfg.metrics_shm_size)
    )
    app.state.start_time = time.monotonic()
    try:
        yield
    finally:
        app.state.metrics_reader.close()


def create_app() -> FastAPI:
    app = FastAPI(title="OmniVoice Triton Server", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        cfg: Settings = request.app.state.cfg
        client: InfererClient = request.app.state.infer_client
        try:
            infer = await client.health()
            ready = bool(infer.get("ready"))
        except Exception as exc:
            infer = {"status": "unreachable", "error": str(exc)}
            ready = False
        return {
            "status": "healthy" if ready else "starting",
            "ready": ready,
            "uptime_s": round(time.monotonic() - request.app.state.start_time, 1),
            "model_id": cfg.model_id,
            "log_file": cfg.log_file,
            "pid_file": cfg.pid_file,
            "worker_pid": __import__("os").getpid(),
            "text_chunk_words": cfg.text_chunk_words,
            "text_chunk_soft_overflow_ratio": cfg.text_chunk_soft_overflow_ratio,
            "text_chunk_same_sentence_penalty": cfg.text_chunk_same_sentence_penalty,
            "text_chunk_sentence_boundary_penalty": cfg.text_chunk_sentence_boundary_penalty,
            "text_chunk_fragment_boundary_penalty": cfg.text_chunk_fragment_boundary_penalty,
            "text_chunk_short_underfill_ratio": cfg.text_chunk_short_underfill_ratio,
            "text_chunk_short_underfill_penalty": cfg.text_chunk_short_underfill_penalty,
            "inferer": infer,
        }

    @app.get("/metrics")
    async def metrics(request: Request):
        snapshot = request.app.state.metrics_reader.read()
        if snapshot is not None:
            return snapshot
        return {
            "type": "health",
            "status": "metrics_unavailable",
            "ready": False,
            "metrics_transport": "shared_memory",
            "metrics_shm_path": request.app.state.cfg.metrics_shm_path,
        }

    @app.get("/v1/models")
    async def list_models(request: Request):
        cfg: Settings = request.app.state.cfg
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "k2-fsa",
                    "root": cfg.model_id,
                    "parent": None if model_id == "omnivoice" else "omnivoice",
                }
                for model_id in ("omnivoice", "tts-1", "tts-1-hd")
            ],
        }

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str, request: Request):
        if model_id not in {"omnivoice", "tts-1", "tts-1-hd"}:
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")
        cfg: Settings = request.app.state.cfg
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "k2-fsa",
            "root": cfg.model_id,
            "parent": None if model_id == "omnivoice" else "omnivoice",
        }

    @app.post("/v1/audio/speech")
    async def speech(body: SpeechRequest, request: Request):
        cfg: Settings = request.app.state.cfg
        client: InfererClient = request.app.state.infer_client
        voice_selector = body.speaker if body.speaker else body.voice
        mode, instruct = resolve_mode_and_instruct(
            voice_selector,
            body.instructions,
            cfg.default_voice_instructions,
        )
        normalized_input = normalize_tts_text_for_language(body.input, body.language)
        infer_req = InferRequest(
            request_id=uuid.uuid4().hex,
            input=normalized_input,
            **_chunk_request_text(normalized_input, body.duration, body.chunk_mode, cfg),
            mode=mode,
            instruct=instruct,
            response_format=body.response_format,
            speed=body.speed,
            duration=body.duration,
            language=body.language,
            chunk_mode=body.chunk_mode,
            num_step=body.num_step,
            stream=body.stream,
            extra_fields=body.extra_fields,
        )
        timeout_s = body.request_timeout_s or cfg.request_timeout_s
        if body.stream:
            return StreamingResponse(
                _sse_events(client, infer_req, timeout_s),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = await client.infer(infer_req, timeout_s)
            pcm_chunks = [_b64decode(chunk["pcm_b64"]) for chunk in result.get("chunks", [])]
            audio = pcm_chunks_to_format(pcm_chunks, body.response_format, cfg.sample_rate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(
            content=audio,
            media_type=media_type_for_format(body.response_format),
            headers={
                "X-Audio-Sample-Rate": str(cfg.sample_rate),
                "X-OmniVoice-Extra-Fields": json.dumps(body.extra_fields, ensure_ascii=False),
            },
        )

    @app.post("/v1/audio/design")
    async def design_voice(
        request: Request,
        text: str = Form(...),
        instruct: str = Form(...),
        language: str | None = Form(None),
        language_id: str | None = Form(None),
        speed: float | None = Form(None),
        duration: float | None = Form(None),
        chunk_mode: Literal["concurrent", "sequential", "none"] = Form("concurrent"),
        num_step: int | None = Form(None),
        response_format: str = Form("wav"),
        request_timeout_s: float | None = Form(None),
    ):
        cfg: Settings = request.app.state.cfg
        resolved_language = language or language_id
        normalized_text = normalize_tts_text_for_language(text, resolved_language)
        infer_req = InferRequest(
            request_id=uuid.uuid4().hex,
            input=normalized_text,
            **_chunk_request_text(normalized_text, duration, chunk_mode, cfg),
            mode="design",
            instruct=instruct,
            response_format=response_format,
            speed=speed or 1.0,
            duration=duration,
            language=resolved_language,
            chunk_mode=chunk_mode,
            num_step=num_step,
        )
        return await _audio_response(request, infer_req, response_format, request_timeout_s or cfg.request_timeout_s)

    @app.post("/v1/audio/clone")
    async def clone_voice(
        request: Request,
        text: str = Form(...),
        ref_audio: UploadFile | None = File(None),
        ref_audio_base64: str | None = Form(None),
        ref_text: str | None = Form(None),
        language: str | None = Form(None),
        language_id: str | None = Form(None),
        speed: float | None = Form(None),
        duration: float | None = Form(None),
        chunk_mode: Literal["concurrent", "sequential", "none"] = Form("concurrent"),
        num_step: int | None = Form(None),
        response_format: str = Form("wav"),
        request_timeout_s: float | None = Form(None),
    ):
        if not ref_audio and not ref_audio_base64:
            raise HTTPException(
                status_code=400,
                detail="Provide ref_audio file upload or ref_audio_base64",
            )
        if not ref_text or not ref_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Provide ref_text for clone requests; inferer-side ASR is disabled.",
            )

        ref_audio_b64 = await _read_reference_audio_b64(ref_audio, ref_audio_base64)
        cfg: Settings = request.app.state.cfg
        resolved_language = language or language_id
        normalized_text = normalize_tts_text_for_language(text, resolved_language)
        normalized_ref_text = normalize_tts_text_for_language(ref_text.strip(), resolved_language)
        infer_req = InferRequest(
            request_id=uuid.uuid4().hex,
            input=normalized_text,
            **_chunk_request_text(normalized_text, duration, chunk_mode, cfg),
            mode="clone",
            ref_audio_b64=ref_audio_b64,
            ref_text=normalized_ref_text,
            response_format=response_format,
            speed=speed or 1.0,
            duration=duration,
            language=resolved_language,
            chunk_mode=chunk_mode,
            num_step=num_step,
        )
        return await _audio_response(
            request,
            infer_req,
            response_format,
            request_timeout_s or cfg.request_timeout_s,
        )

    return app


async def _audio_response(
    request: Request,
    infer_req: InferRequest,
    response_format: str,
    timeout_s: float,
) -> Response:
    cfg: Settings = request.app.state.cfg
    client: InfererClient = request.app.state.infer_client
    try:
        result = await client.infer(infer_req, timeout_s)
        pcm_chunks = [_b64decode(chunk["pcm_b64"]) for chunk in result.get("chunks", [])]
        audio = pcm_chunks_to_format(pcm_chunks, response_format, cfg.sample_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        content=audio,
        media_type=media_type_for_format(response_format),
        headers={"X-Audio-Sample-Rate": str(cfg.sample_rate)},
    )


async def _read_reference_audio_b64(
    ref_audio: UploadFile | None,
    ref_audio_base64: str | None,
) -> str:
    if ref_audio_base64:
        try:
            payload = ref_audio_base64.split(",", 1)[1] if "," in ref_audio_base64 else ref_audio_base64
            data = base64.b64decode(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="ref_audio_base64 is not valid base64") from exc
    else:
        assert ref_audio is not None
        data = await ref_audio.read()
    return base64.b64encode(data).decode("ascii")


def _chunk_request_text(
    text: str,
    duration: float | None,
    chunk_mode: Literal["concurrent", "sequential", "none"],
    cfg: Settings,
) -> dict[str, list[str] | list[float | None]]:
    max_word_count = (
        _max_word_count_for_none_chunk_mode(cfg)
        if chunk_mode == "none"
        else cfg.text_chunk_words
    )
    chunks = split_text_by_word_count(
        text,
        max_word_count=max_word_count,
        soft_overflow_ratio=cfg.text_chunk_soft_overflow_ratio,
        same_sentence_penalty=cfg.text_chunk_same_sentence_penalty,
        sentence_boundary_penalty=cfg.text_chunk_sentence_boundary_penalty,
        fragment_boundary_penalty=cfg.text_chunk_fragment_boundary_penalty,
        short_underfill_ratio=cfg.text_chunk_short_underfill_ratio,
        short_underfill_penalty=cfg.text_chunk_short_underfill_penalty,
    )
    chunk_texts = [chunk["text"] for chunk in chunks]
    return {
        "chunks": chunk_texts,
        "chunk_durations": _split_duration(duration, chunks),
    }


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


def _max_word_count_for_none_chunk_mode(cfg: Settings) -> int:
    model_limit = _model_max_position_embeddings(cfg.model_id)
    if model_limit is None:
        return cfg.text_chunk_words
    tokens_per_word = max(1, int(cfg.cuda_graph_auto_width_tokens_per_word))
    reserved_tokens = max(512, int(model_limit * 0.05))
    usable_tokens = max(tokens_per_word, model_limit - reserved_tokens)
    return max(cfg.text_chunk_words, usable_tokens // tokens_per_word)


def _split_duration(
    duration: float | None,
    chunks: list[TextChunk],
) -> list[float | None]:
    if duration is None:
        return []
    if not chunks:
        return []
    if len(chunks) == 1:
        return [duration]

    weights = [max(1, int(chunk["word_count"])) for chunk in chunks]
    total_weight = sum(weights)
    durations = [duration * weight / total_weight for weight in weights]
    durations[-1] += duration - sum(durations)
    return durations


async def _sse_events(
    client: InfererClient,
    infer_req: InferRequest,
    timeout_s: float,
):
    try:
        async for msg in client.stream(infer_req, timeout_s):
            typ = msg.get("type")
            if typ == "accepted":
                yield _sse("speech.accepted", {"tasks": msg.get("tasks", 0)})
            elif typ == "chunk":
                payload = {
                    "object": "audio.chunk",
                    "request_id": msg.get("request_id"),
                    "task_id": msg.get("task_id"),
                    "seq": msg.get("seq"),
                    "part": msg.get("part"),
                    "parts": msg.get("parts"),
                    "encoding": "pcm16_base64",
                    "sample_rate": msg.get("sample_rate", 24000),
                    "audio": msg.get("pcm_b64", ""),
                }
                yield _sse("speech.audio.delta", payload)
            elif typ == "done":
                yield _sse("speech.audio.done", {"sample_rate": msg.get("sample_rate", 24000)})
                yield "data: [DONE]\n\n"
    except Exception as exc:
        yield _sse("error", {"error": f"{type(exc).__name__}: {exc}"})
        yield "data: [DONE]\n\n"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


app = create_app()


def _load_inferer_entries(cfg: Settings) -> list[dict[str, Any]]:
    if not cfg.inferers:
        return []
    try:
        payload = json.loads(cfg.inferers)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]
