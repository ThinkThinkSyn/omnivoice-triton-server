from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


OUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "test-artifacts"

SHORT_TEXTS = [
    "Hi.",
    "Okay, one moment.",
    "Thanks, that is done.",
    "Good morning. Please continue.",
    "No problem, try again.",
]
MEDIUM_TEXTS = [
    "This request checks batching with a medium sentence, a number like 2026, and natural punctuation.",
    "今天我们测试中等长度文本，里面包含 English words、数字 12345，还有正常的停顿。",
    "The scheduler should balance requests across multiple GPU inferers under sustained load.",
    "语音需要保持清晰稳定，同时参数会混合 speed、duration、language 和不同输出格式。",
]
LONG_TEXTS = [
    (
        "今天我们验证长文本 chunking 和 multi inferer routing。"
        "The text mixes Chinese and English, numbers like 2026.05, punctuation, and several semantic clauses. "
        "It should split into multiple chunks, preserve continuity, and still return valid audio. "
        "接下来继续加入更多内容，让请求明显长于普通短句，同时保持可控的目标时长。"
    ),
    (
        "A longer paragraph should exercise chunk packing without forcing every request into tiny batches. "
        "It includes English clauses, 中文短句，数字 12345, emoji-like descriptions, and normal punctuation. "
        "The generated audio should remain valid while the scheduler keeps GPU inferers busy."
    ),
]
DESIGNS = ["female, young adult", "male, middle-aged", "female, american accent", "male, low pitch"]
VOICES = ["auto", "alloy", "nova", "onyx", "sage"]
FORMATS = ["wav", "pcm", "pcm", "wav"]
SPEEDS = [0.85, 1.0, 1.15, 1.3]
LANGUAGES = [None, "en", None, None]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def valid_audio(fmt: str, body: bytes) -> bool:
    if fmt == "wav":
        return len(body) > 44 and body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    return len(body) > 0


def request_plan(index: int, ref_audio_b64: str | None, chunk_mode: str) -> dict[str, Any]:
    fmt = FORMATS[index % len(FORMATS)]
    speed = SPEEDS[index % len(SPEEDS)]
    language = LANGUAGES[index % len(LANGUAGES)]

    if index % 20 == 0 and ref_audio_b64 is not None:
        text = LONG_TEXTS[(index // 20) % len(LONG_TEXTS)] if index % 40 == 0 else MEDIUM_TEXTS[index % len(MEDIUM_TEXTS)]
        data: dict[str, Any] = {
            "text": text,
            "ref_audio_base64": ref_audio_b64,
            "ref_text": "Reference audio for clone smoke test.",
            "response_format": fmt,
            "speed": str(speed),
            "chunk_mode": chunk_mode,
            "request_timeout_s": "300",
        }
        if index % 3 != 0:
            data["duration"] = str(1.2 if len(text) > 180 else 0.45)
        if language:
            data["language"] = language
        return {"kind": "clone", "method": "form", "path": "/v1/audio/clone", "data": data, "format": fmt}

    if index % 10 == 0 or (index % 20 == 0 and ref_audio_b64 is None):
        text = LONG_TEXTS[(index // 10) % len(LONG_TEXTS)] if index % 30 == 0 else MEDIUM_TEXTS[index % len(MEDIUM_TEXTS)]
        data = {
            "text": text,
            "instruct": DESIGNS[index % len(DESIGNS)],
            "response_format": fmt,
            "speed": str(speed),
            "chunk_mode": chunk_mode,
            "request_timeout_s": "300",
        }
        if index % 4 != 0:
            data["duration"] = str(1.1 if len(text) > 180 else 0.4)
        if language:
            data["language"] = language
        kind = "design" if ref_audio_b64 is not None else "design_no_ref_audio"
        return {"kind": kind, "method": "form", "path": "/v1/audio/design", "data": data, "format": fmt}

    if index % 8 == 0:
        text = LONG_TEXTS[index % len(LONG_TEXTS)]
        duration = 1.0
    elif index % 3 == 0:
        text = MEDIUM_TEXTS[index % len(MEDIUM_TEXTS)]
        duration = 0.5
    else:
        text = f"{SHORT_TEXTS[index % len(SHORT_TEXTS)]} #{index}"
        duration = 0.25 + (index % 4) * 0.05

    payload: dict[str, Any] = {
        "model": "tts-1",
        "input": text,
        "voice": VOICES[index % len(VOICES)],
        "response_format": fmt,
        "speed": speed,
        "chunk_mode": chunk_mode,
        "request_timeout_s": 300,
    }
    if index % 6 != 0:
        payload["duration"] = duration
    if language:
        payload["language"] = language
    if index % 37 == 0:
        payload["CFG"] = 2.5
    return {"kind": "speech", "method": "json", "path": "/v1/audio/speech", "json": payload, "format": fmt}


async def sample_metrics(client: httpx.AsyncClient, stop: asyncio.Event, samples: list[dict[str, Any]]) -> None:
    while not stop.is_set():
        try:
            response = await client.get("/metrics", timeout=5.0)
            if response.status_code == 200:
                item = response.json()
                item["_ts"] = time.time()
                samples.append(item)
        except Exception as exc:
            samples.append({"_ts": time.time(), "error": repr(exc)})
        await asyncio.sleep(1.0)


async def send_one(
    client: httpx.AsyncClient,
    index: int,
    scheduled_at: float,
    ref_audio_b64: str | None,
    chunk_mode: str,
    result: dict[str, Any],
) -> None:
    await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
    plan = request_plan(index, ref_audio_b64, chunk_mode)
    start = time.perf_counter()
    try:
        if plan["method"] == "json":
            response = await client.post(plan["path"], json=plan["json"], timeout=300.0)
        else:
            response = await client.post(plan["path"], data=plan["data"], timeout=300.0)
        body = await response.aread()
        elapsed = time.perf_counter() - start
        result.update(
            {
                "index": index,
                "kind": plan["kind"],
                "status": response.status_code,
                "elapsed_s": elapsed,
                "bytes": len(body),
                "format": plan["format"],
                "valid_audio": response.status_code == 200 and valid_audio(plan["format"], body),
                "error": None if response.status_code == 200 else body[:500].decode("utf-8", "replace"),
            }
        )
    except Exception as exc:
        result.update(
            {
                "index": index,
                "kind": plan["kind"],
                "status": None,
                "elapsed_s": time.perf_counter() - start,
                "bytes": 0,
                "format": plan["format"],
                "valid_audio": False,
                "error": repr(exc),
            }
        )


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [item["elapsed_s"] for item in results if item.get("status") == 200]
    statuses: dict[str, int] = {}
    by_kind: dict[str, dict[str, Any]] = {}
    for item in results:
        statuses[str(item.get("status"))] = statuses.get(str(item.get("status")), 0) + 1
        kind = str(item.get("kind"))
        bucket = by_kind.setdefault(kind, {"total": 0, "success": 0, "statuses": {}, "latencies": []})
        bucket["total"] += 1
        bucket["statuses"][str(item.get("status"))] = bucket["statuses"].get(str(item.get("status")), 0) + 1
        if item.get("status") == 200:
            bucket["success"] += 1
            bucket["latencies"].append(item["elapsed_s"])

    for bucket in by_kind.values():
        values = bucket.pop("latencies")
        bucket["latency_s"] = {
            "mean": round(statistics.fmean(values), 4) if values else 0,
            "p95": round(percentile(values, 95), 4),
            "max": round(max(values), 4) if values else 0,
        }

    return {
        "success": len(latencies),
        "statuses": statuses,
        "by_kind": by_kind,
        "latency_s": {
            "min": round(min(latencies), 4) if latencies else 0,
            "mean": round(statistics.fmean(latencies), 4) if latencies else 0,
            "p50": round(percentile(latencies, 50), 4),
            "p90": round(percentile(latencies, 90), 4),
            "p95": round(percentile(latencies, 95), 4),
            "p99": round(percentile(latencies, 99), 4),
            "max": round(max(latencies), 4) if latencies else 0,
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    ref_audio_b64: str | None = None
    if args.ref_audio:
        ref_audio_b64 = "data:audio/wav;base64," + base64.b64encode(Path(args.ref_audio).read_bytes()).decode("ascii")

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", limits=limits, timeout=timeout) as client:
        before = (await client.get("/metrics")).json()
        stop = asyncio.Event()
        samples: list[dict[str, Any]] = []
        sampler = asyncio.create_task(sample_metrics(client, stop, samples))
        results: list[dict[str, Any]] = [{} for _ in range(args.total)]
        started_at = time.perf_counter()
        tasks = [
            asyncio.create_task(
                send_one(
                    client,
                    idx,
                    started_at + (idx / args.rate),
                    ref_audio_b64,
                    args.chunk_mode,
                    results[idx],
                )
            )
            for idx in range(args.total)
        ]
        await asyncio.gather(*tasks)
        completed_at = time.perf_counter()
        stop.set()
        await sampler
        after = (await client.get("/metrics")).json()

    summary = summarize_results(results)
    errors = [item for item in results if item.get("status") != 200]
    bad_outputs = [item for item in results if item.get("status") == 200 and not item.get("valid_audio")]
    bytes_total = sum(int(item.get("bytes") or 0) for item in results)
    audio_s = max(0.0, (bytes_total - summary["success"] * 44) / 48000.0)
    duration = completed_at - started_at
    before_tasks = int(before.get("total_tasks") or 0)
    after_tasks = int(after.get("total_tasks") or 0)
    before_errors = int(before.get("total_errors") or 0)
    after_errors = int(after.get("total_errors") or 0)
    result = {
        "total": args.total,
        "target_rate_rps": args.rate,
        "chunk_mode": args.chunk_mode,
        "clone_enabled": ref_audio_b64 is not None,
        "actual_wall_s": round(duration, 3),
        "actual_completion_rps": round(args.total / duration, 3) if duration else 0,
        **summary,
        "bytes_total": bytes_total,
        "estimated_audio_s": round(audio_s, 3),
        "estimated_rtf_wall_over_audio": round(duration / audio_s, 4) if audio_s else 0,
        "estimated_realtime_factor_audio_over_wall": round(audio_s / duration, 3) if duration else 0,
        "backend_delta": {
            "tasks": after_tasks - before_tasks,
            "errors": after_errors - before_errors,
        },
        "before_metrics": before,
        "after_metrics": after,
        "metric_samples": samples,
        "errors": errors[:30],
        "bad_outputs": bad_outputs[:30],
        "passed": len(errors) == 0 and len(bad_outputs) == 0 and (after_errors - before_errors) == 0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9194)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--chunk-mode", choices=["concurrent", "sequential", "none"], default="concurrent")
    parser.add_argument("--ref-audio", default="")
    parser.add_argument("--out", default=str(OUT_DIR / "mixed_1000_results.json"))
    args = parser.parse_args()
    result = asyncio.run(run(args))
    printable = {
        key: value
        for key, value in result.items()
        if key not in {"metric_samples", "before_metrics", "after_metrics"}
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
