from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn

from config import Settings
from logging_config import build_logging_config, configure_logging


ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if visible.strip() == "":
            return []
        return ids
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return []
    return [str(idx) for idx, line in enumerate(proc.stdout.splitlines()) if line.startswith("GPU ")]


def inferer_metrics_path(base_path: str, name: str) -> str:
    path = Path(base_path)
    suffix = path.suffix or ".shm"
    stem = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.name
    return str(path.with_name(f"{stem}_{name}{suffix}"))


def wait_for_ready(proc: subprocess.Popen, timeout_s: float) -> tuple[str, int]:
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"inferer exited early with code {proc.returncode}")
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        print(line, end="", flush=True)
        if line.startswith("OMNIVOICE_INFERER_READY"):
            parts = dict(item.split("=", 1) for item in line.strip().split()[1:])
            return parts["host"], int(parts["port"])
    raise TimeoutError("inferer did not become ready in time")


def drain_stdout(proc: subprocess.Popen) -> threading.Thread:
    assert proc.stdout is not None

    def _drain() -> None:
        for line in proc.stdout:
            print(line, end="", flush=True)

    thread = threading.Thread(target=_drain, name="omnivoice-inferer-log-drain", daemon=True)
    thread.start()
    return thread


def set_arg_env(cfg: Settings, attr: str, value, env_name: str) -> None:
    os.environ[env_name] = str(value)
    setattr(cfg, attr, value)


def argv_has_option(argv: list[str], option_names: set[str]) -> bool:
    for item in argv:
        option = item.split("=", 1)[0]
        if option in option_names:
            return True
    return False


def resolve_runtime_paths(cfg: Settings) -> None:
    run_id = cfg.log_run_id or time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(cfg.log_dir).expanduser() / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.log_run_id:
        cfg.log_run_id = run_id
        os.environ["OMNIVOICE_LOG_RUN_ID"] = run_id

    if not cfg.log_file:
        cfg.log_file = str(log_dir / "server.log")
        os.environ["OMNIVOICE_LOG_FILE"] = cfg.log_file

    if not cfg.pid_file:
        cfg.pid_file = str(log_dir / "server.pid")
        os.environ["OMNIVOICE_PID_FILE"] = cfg.pid_file

    os.environ["OMNIVOICE_LOG_DIR"] = cfg.log_dir

    if not cfg.metrics_shm_path:
        safe_run_id = cfg.log_run_id.replace("/", "_")
        cfg.metrics_shm_path = f"/dev/shm/omnivoice_metrics_{safe_run_id}_{cfg.port}.shm"
        os.environ["OMNIVOICE_METRICS_SHM_PATH"] = cfg.metrics_shm_path
    os.environ["OMNIVOICE_METRICS_SHM_SIZE"] = str(cfg.metrics_shm_size)
    if not cfg.clone_prompt_shared_cache_dir:
        safe_run_id = cfg.log_run_id.replace("/", "_")
        cfg.clone_prompt_shared_cache_dir = f"/dev/shm/omnivoice_clone_prompt_cache_{safe_run_id}_{cfg.port}"
        os.environ["OMNIVOICE_CLONE_PROMPT_SHARED_CACHE_DIR"] = cfg.clone_prompt_shared_cache_dir


def write_pid_file(path: str, pid: int) -> None:
    pid_path = Path(path).expanduser()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")


def cleanup_old_log_dirs(cfg: Settings) -> None:
    if cfg.log_retention_days <= 0:
        return
    root = Path(cfg.log_dir).expanduser()
    if not root.exists():
        return
    cutoff = time.time() - (cfg.log_retention_days * 24 * 60 * 60)
    for child in root.iterdir():
        if not child.is_dir() or child.name == cfg.log_run_id:
            continue
        try:
            run_ts = time.mktime(time.strptime(child.name, "%Y%m%d-%H%M%S"))
        except ValueError:
            continue
        if run_ts < cutoff:
            shutil.rmtree(child, ignore_errors=True)


def add_start_arguments(parser: argparse.ArgumentParser, cfg: Settings) -> None:
    parser.add_argument("--host", default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    parser.add_argument("--fastapi-workers", "--workers", dest="workers", type=int, default=cfg.workers)
    parser.add_argument("--infer-port", type=int, default=cfg.infer_port)
    parser.add_argument("--infer-host", default=cfg.infer_host)
    parser.add_argument("--gpu-inferer", type=int, default=cfg.gpu_inferer)
    parser.add_argument("--log-level", default=cfg.log_level)
    parser.add_argument("--log-dir", default=cfg.log_dir)
    parser.add_argument("--log-run-id", default=cfg.log_run_id)
    parser.add_argument("--log-file", default=cfg.log_file)
    parser.add_argument("--pid-file", default=cfg.pid_file)
    parser.add_argument("--log-retention-days", type=int, default=cfg.log_retention_days)
    parser.add_argument("--model-id", default=cfg.model_id)
    parser.add_argument("--runner-mode", choices=["official", "triton", "hybrid"], default=cfg.runner_mode)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=cfg.dtype)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--request-timeout-s", type=float, default=cfg.request_timeout_s)
    parser.add_argument("--infer-start-timeout-s", type=float, default=cfg.infer_start_timeout_s)
    parser.add_argument("--metrics-shm-path", default=cfg.metrics_shm_path)
    parser.add_argument("--metrics-shm-size", type=int, default=cfg.metrics_shm_size)
    parser.add_argument(
        "--metrics-snapshot-interval-s",
        type=float,
        default=cfg.metrics_snapshot_interval_s,
    )
    parser.add_argument(
        "--max-batch-size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=cfg.batch_size,
    )
    parser.add_argument(
        "--max-batch-latency",
        "--batch-wait-ms",
        dest="batch_wait_ms",
        type=int,
        default=cfg.batch_wait_ms,
        help="Maximum micro-batch wait in milliseconds",
    )
    parser.add_argument(
        "--cuda-stream-count",
        "--cuda-streams",
        dest="cuda_streams",
        type=int,
        default=cfg.cuda_streams,
    )
    parser.add_argument("--cuda-graph-min-width", type=int, default=cfg.cuda_graph_min_width)
    parser.add_argument("--cuda-graph-max-width", type=int, default=cfg.cuda_graph_max_width)
    parser.add_argument(
        "--cuda-graph-auto-width-tokens-per-word",
        type=int,
        default=cfg.cuda_graph_auto_width_tokens_per_word,
    )
    parser.add_argument("--cuda-graph-auto-max-width", type=int, default=cfg.cuda_graph_auto_max_width)
    parser.add_argument("--max-continuity-audio-tokens", type=int, default=cfg.max_continuity_audio_tokens)
    parser.add_argument("--max-continuity-text-words", type=int, default=cfg.max_continuity_text_words)
    parser.add_argument("--sample-rate", type=int, default=cfg.sample_rate)
    parser.add_argument("--max-sse-audio-b64-chars", type=int, default=cfg.max_sse_audio_b64_chars)
    parser.add_argument("--max-clone-audio-prompt-cache", type=int, default=cfg.max_clone_audio_prompt_cache)
    parser.add_argument("--num-step", type=int, default=cfg.num_step)
    parser.add_argument("--guidance-scale", type=float, default=cfg.guidance_scale)
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=cfg.denoise)
    parser.add_argument("--t-shift", type=float, default=cfg.t_shift)
    parser.add_argument("--position-temperature", type=float, default=cfg.position_temperature)
    parser.add_argument("--class-temperature", type=float, default=cfg.class_temperature)
    parser.add_argument("--layer-penalty-factor", type=float, default=cfg.layer_penalty_factor)
    parser.add_argument("--audio-chunk-duration", type=float, default=cfg.audio_chunk_duration)
    parser.add_argument("--audio-chunk-threshold", type=float, default=cfg.audio_chunk_threshold)
    parser.add_argument(
        "--postprocess-output",
        action=argparse.BooleanOptionalAction,
        default=cfg.postprocess_output,
    )
    parser.add_argument("--text-chunk-words", type=int, default=cfg.text_chunk_words)
    parser.add_argument(
        "--text-chunk-soft-overflow-ratio",
        type=float,
        default=cfg.text_chunk_soft_overflow_ratio,
    )
    parser.add_argument(
        "--text-chunk-same-sentence-penalty",
        type=int,
        default=cfg.text_chunk_same_sentence_penalty,
    )
    parser.add_argument(
        "--text-chunk-sentence-boundary-penalty",
        type=int,
        default=cfg.text_chunk_sentence_boundary_penalty,
    )
    parser.add_argument(
        "--text-chunk-fragment-boundary-penalty",
        type=int,
        default=cfg.text_chunk_fragment_boundary_penalty,
    )
    parser.add_argument(
        "--text-chunk-short-underfill-ratio",
        type=float,
        default=cfg.text_chunk_short_underfill_ratio,
    )
    parser.add_argument(
        "--text-chunk-short-underfill-penalty",
        type=int,
        default=cfg.text_chunk_short_underfill_penalty,
    )
    parser.add_argument("--default-voice-instructions", default=cfg.default_voice_instructions)


def start(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    cfg = Settings()
    parser = argparse.ArgumentParser(
        prog="omnivoice-triton-server start",
        description="Launch OmniVoice FastAPI server and GPU inferer processes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_start_arguments(parser, cfg)
    for action in parser._actions:
        if action.dest != "help" and action.help is None:
            action.help = "default: %(default)s"
    args = parser.parse_args(argv)

    set_arg_env(cfg, "host", args.host, "OMNIVOICE_HOST")
    set_arg_env(cfg, "port", args.port, "OMNIVOICE_PORT")
    workers_was_explicit = (
        argv_has_option(argv, {"--fastapi-workers", "--workers"}) or "OMNIVOICE_WORKERS" in os.environ
    )
    set_arg_env(cfg, "workers", args.workers, "OMNIVOICE_WORKERS")
    set_arg_env(cfg, "infer_host", args.infer_host, "OMNIVOICE_INFER_HOST")
    set_arg_env(cfg, "infer_port", args.infer_port, "OMNIVOICE_INFER_PORT")
    set_arg_env(cfg, "gpu_inferer", args.gpu_inferer, "OMNIVOICE_GPU_INFERER")
    set_arg_env(cfg, "log_level", args.log_level, "OMNIVOICE_LOG_LEVEL")
    set_arg_env(cfg, "log_dir", args.log_dir, "OMNIVOICE_LOG_DIR")
    set_arg_env(cfg, "log_run_id", args.log_run_id, "OMNIVOICE_LOG_RUN_ID")
    set_arg_env(cfg, "log_file", args.log_file, "OMNIVOICE_LOG_FILE")
    set_arg_env(cfg, "pid_file", args.pid_file, "OMNIVOICE_PID_FILE")
    set_arg_env(cfg, "log_retention_days", args.log_retention_days, "OMNIVOICE_LOG_RETENTION_DAYS")
    set_arg_env(cfg, "model_id", args.model_id, "OMNIVOICE_MODEL_ID")
    set_arg_env(cfg, "runner_mode", args.runner_mode, "OMNIVOICE_RUNNER_MODE")
    set_arg_env(cfg, "dtype", args.dtype, "OMNIVOICE_DTYPE")
    set_arg_env(cfg, "device", args.device, "OMNIVOICE_DEVICE")
    set_arg_env(cfg, "request_timeout_s", args.request_timeout_s, "OMNIVOICE_REQUEST_TIMEOUT_S")
    set_arg_env(
        cfg,
        "infer_start_timeout_s",
        args.infer_start_timeout_s,
        "OMNIVOICE_INFER_START_TIMEOUT_S",
    )
    set_arg_env(cfg, "metrics_shm_path", args.metrics_shm_path, "OMNIVOICE_METRICS_SHM_PATH")
    set_arg_env(cfg, "metrics_shm_size", args.metrics_shm_size, "OMNIVOICE_METRICS_SHM_SIZE")
    set_arg_env(
        cfg,
        "metrics_snapshot_interval_s",
        args.metrics_snapshot_interval_s,
        "OMNIVOICE_METRICS_SNAPSHOT_INTERVAL_S",
    )
    set_arg_env(cfg, "batch_size", args.batch_size, "OMNIVOICE_BATCH_SIZE")
    set_arg_env(cfg, "batch_wait_ms", args.batch_wait_ms, "OMNIVOICE_BATCH_WAIT_MS")
    set_arg_env(cfg, "cuda_streams", args.cuda_streams, "OMNIVOICE_CUDA_STREAMS")
    set_arg_env(
        cfg,
        "cuda_graph_min_width",
        args.cuda_graph_min_width,
        "OMNIVOICE_CUDA_GRAPH_MIN_WIDTH",
    )
    set_arg_env(
        cfg,
        "cuda_graph_max_width",
        args.cuda_graph_max_width,
        "OMNIVOICE_CUDA_GRAPH_MAX_WIDTH",
    )
    set_arg_env(
        cfg,
        "cuda_graph_auto_width_tokens_per_word",
        args.cuda_graph_auto_width_tokens_per_word,
        "OMNIVOICE_CUDA_GRAPH_AUTO_WIDTH_TOKENS_PER_WORD",
    )
    set_arg_env(
        cfg,
        "cuda_graph_auto_max_width",
        args.cuda_graph_auto_max_width,
        "OMNIVOICE_CUDA_GRAPH_AUTO_MAX_WIDTH",
    )
    set_arg_env(
        cfg,
        "max_continuity_audio_tokens",
        args.max_continuity_audio_tokens,
        "OMNIVOICE_MAX_CONTINUITY_AUDIO_TOKENS",
    )
    set_arg_env(
        cfg,
        "max_continuity_text_words",
        args.max_continuity_text_words,
        "OMNIVOICE_MAX_CONTINUITY_TEXT_WORDS",
    )
    set_arg_env(cfg, "sample_rate", args.sample_rate, "OMNIVOICE_SAMPLE_RATE")
    set_arg_env(
        cfg,
        "max_sse_audio_b64_chars",
        args.max_sse_audio_b64_chars,
        "OMNIVOICE_MAX_SSE_AUDIO_B64_CHARS",
    )
    set_arg_env(
        cfg,
        "max_clone_audio_prompt_cache",
        args.max_clone_audio_prompt_cache,
        "OMNIVOICE_MAX_CLONE_AUDIO_PROMPT_CACHE",
    )
    set_arg_env(cfg, "num_step", args.num_step, "OMNIVOICE_NUM_STEP")
    set_arg_env(cfg, "guidance_scale", args.guidance_scale, "OMNIVOICE_GUIDANCE_SCALE")
    set_arg_env(cfg, "denoise", args.denoise, "OMNIVOICE_DENOISE")
    set_arg_env(cfg, "t_shift", args.t_shift, "OMNIVOICE_T_SHIFT")
    set_arg_env(
        cfg,
        "position_temperature",
        args.position_temperature,
        "OMNIVOICE_POSITION_TEMPERATURE",
    )
    set_arg_env(
        cfg,
        "class_temperature",
        args.class_temperature,
        "OMNIVOICE_CLASS_TEMPERATURE",
    )
    set_arg_env(
        cfg,
        "layer_penalty_factor",
        args.layer_penalty_factor,
        "OMNIVOICE_LAYER_PENALTY_FACTOR",
    )
    set_arg_env(
        cfg,
        "audio_chunk_duration",
        args.audio_chunk_duration,
        "OMNIVOICE_AUDIO_CHUNK_DURATION",
    )
    set_arg_env(
        cfg,
        "audio_chunk_threshold",
        args.audio_chunk_threshold,
        "OMNIVOICE_AUDIO_CHUNK_THRESHOLD",
    )
    set_arg_env(cfg, "postprocess_output", args.postprocess_output, "OMNIVOICE_POSTPROCESS_OUTPUT")
    set_arg_env(cfg, "text_chunk_words", args.text_chunk_words, "OMNIVOICE_TEXT_CHUNK_WORDS")
    set_arg_env(
        cfg,
        "text_chunk_soft_overflow_ratio",
        args.text_chunk_soft_overflow_ratio,
        "OMNIVOICE_TEXT_CHUNK_SOFT_OVERFLOW_RATIO",
    )
    set_arg_env(
        cfg,
        "text_chunk_same_sentence_penalty",
        args.text_chunk_same_sentence_penalty,
        "OMNIVOICE_TEXT_CHUNK_SAME_SENTENCE_PENALTY",
    )
    set_arg_env(
        cfg,
        "text_chunk_sentence_boundary_penalty",
        args.text_chunk_sentence_boundary_penalty,
        "OMNIVOICE_TEXT_CHUNK_SENTENCE_BOUNDARY_PENALTY",
    )
    set_arg_env(
        cfg,
        "text_chunk_fragment_boundary_penalty",
        args.text_chunk_fragment_boundary_penalty,
        "OMNIVOICE_TEXT_CHUNK_FRAGMENT_BOUNDARY_PENALTY",
    )
    set_arg_env(
        cfg,
        "text_chunk_short_underfill_ratio",
        args.text_chunk_short_underfill_ratio,
        "OMNIVOICE_TEXT_CHUNK_SHORT_UNDERFILL_RATIO",
    )
    set_arg_env(
        cfg,
        "text_chunk_short_underfill_penalty",
        args.text_chunk_short_underfill_penalty,
        "OMNIVOICE_TEXT_CHUNK_SHORT_UNDERFILL_PENALTY",
    )
    set_arg_env(
        cfg,
        "default_voice_instructions",
        args.default_voice_instructions,
        "OMNIVOICE_DEFAULT_VOICE_INSTRUCTIONS",
    )

    resolve_runtime_paths(cfg)
    cleanup_old_log_dirs(cfg)
    write_pid_file(cfg.pid_file, os.getpid())
    configure_logging(cfg.log_level, cfg.log_file, cfg.log_retention_days)

    requested_gpu_inferers = cfg.gpu_inferer
    if requested_gpu_inferers <= 0:
        raise ValueError("gpu-inferer must be greater than 0")

    gpu_ids = visible_gpu_ids()
    gpu_inferers = min(requested_gpu_inferers, len(gpu_ids))
    if requested_gpu_inferers and gpu_inferers < requested_gpu_inferers:
        print(
            f"Clamping gpu_inferer from {requested_gpu_inferers} to {gpu_inferers} "
            f"visible GPU(s)",
            flush=True,
        )
    if gpu_inferers <= 0:
        raise ValueError("No inferer can be started: no visible GPU")
    if not workers_was_explicit:
        cfg.workers = gpu_inferers

    env = os.environ.copy()
    py_path = [SRC]
    if env.get("PYTHONPATH"):
        py_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_path)

    inferer_specs: list[dict] = []
    next_port = args.infer_port or cfg.infer_port or 0
    for idx in range(gpu_inferers):
        name = f"gpu{idx}"
        port = next_port or find_free_port()
        next_port = 0
        inferer_specs.append(
            {
                "name": name,
                "kind": "gpu",
                "host": cfg.infer_host,
                "port": port,
                "gpu_id": gpu_ids[idx],
                "metrics_shm_path": inferer_metrics_path(cfg.metrics_shm_path, name),
            }
        )
    if not inferer_specs:
        raise ValueError("No inferer endpoints configured")

    os.environ["OMNIVOICE_INFER_HOST"] = str(inferer_specs[0]["host"])
    os.environ["OMNIVOICE_INFER_PORT"] = str(inferer_specs[0]["port"])
    os.environ["OMNIVOICE_INFERERS"] = json.dumps(inferer_specs, separators=(",", ":"))

    infer_procs: list[subprocess.Popen] = []

    def start_inferer(spec: dict) -> subprocess.Popen:
        child_env = env.copy()
        child_env["OMNIVOICE_INFER_HOST"] = str(spec["host"])
        child_env["OMNIVOICE_INFER_PORT"] = str(spec["port"])
        child_env["OMNIVOICE_INFERER_NAME"] = str(spec["name"])
        child_env["OMNIVOICE_INFERER_KIND"] = str(spec["kind"])
        child_env["OMNIVOICE_METRICS_SHM_PATH"] = str(spec["metrics_shm_path"])
        child_env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_id"])
        child_env["OMNIVOICE_DEVICE"] = "cuda"
        cmd = [
            sys.executable,
            "-m",
            "inferer",
            "--host",
            str(spec["host"]),
            "--port",
            str(spec["port"]),
            "--log-level",
            cfg.log_level,
            "--log-file",
            cfg.log_file,
            "--inferer-name",
            str(spec["name"]),
            "--inferer-kind",
            str(spec["kind"]),
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
            cwd=str(ROOT),
            bufsize=1,
        )

    for spec in inferer_specs:
        proc = start_inferer(spec)
        infer_procs.append(proc)
        spec["pid"] = proc.pid

    def shutdown(*_):
        for proc in infer_procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in infer_procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for spec in inferer_specs:
            try:
                Path(str(spec["metrics_shm_path"])).unlink()
            except FileNotFoundError:
                pass
        if cfg.metrics_shm_path:
            try:
                Path(cfg.metrics_shm_path).unlink()
            except FileNotFoundError:
                pass
        if cfg.clone_prompt_shared_cache_dir:
            shutil.rmtree(cfg.clone_prompt_shared_cache_dir, ignore_errors=True)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        for proc in infer_procs:
            wait_for_ready(proc, cfg.infer_start_timeout_s)
        for proc in infer_procs:
            drain_stdout(proc)
        print(
            f"OMNIVOICE_SERVER_READY host={cfg.host} port={cfg.port} "
            f"workers={cfg.workers} inferers={len(inferer_specs)} "
            f"gpu_inferers={gpu_inferers} "
            f"log_file={cfg.log_file} pid_file={cfg.pid_file}",
            flush=True,
        )
        uvicorn.run(
            "app:app",
            host=cfg.host,
            port=cfg.port,
            workers=cfg.workers,
            log_level=cfg.log_level,
            log_config=build_logging_config(cfg.log_level, cfg.log_file, cfg.log_retention_days),
            timeout_keep_alive=120,
            backlog=2048,
            factory=False,
        )
    finally:
        shutdown()


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pid_file(path: str) -> int | None:
    try:
        raw = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def latest_pid_file(log_dir: str) -> str | None:
    root = Path(log_dir).expanduser()
    if not root.exists():
        return None
    candidates = [p for p in root.glob("*/server.pid") if p.is_file()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)


def pids_listening_on_port(port: int) -> set[int]:
    pids: set[int] = set()
    commands = [
        ["lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"],
        ["fuser", "-n", "tcp", str(port)],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except OSError:
            continue
        for item in proc.stdout.replace(",", " ").split():
            try:
                pids.add(int(item))
            except ValueError:
                pass
        if pids:
            break
    return pids


def terminate_pids(pids: set[int], timeout_s: float, kill_after_timeout: bool) -> int:
    live = {pid for pid in pids if pid_exists(pid)}
    if not live:
        return 0
    for pid in sorted(live):
        print(f"Stopping pid {pid}", flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        live = {pid for pid in live if pid_exists(pid)}
        if not live:
            return 0
        time.sleep(0.2)
    if kill_after_timeout:
        for pid in sorted(live):
            print(f"Killing pid {pid}", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return 0
    print(f"Timed out waiting for pid(s): {', '.join(str(pid) for pid in sorted(live))}", file=sys.stderr)
    return 1


def stop(argv: list[str] | None = None) -> int:
    cfg = Settings()
    parser = argparse.ArgumentParser(
        prog="omnivoice-triton-server stop",
        description="Stop an OmniVoice server started by this CLI or by systemd",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pid-file", action="append", default=[], help="PID file to stop")
    parser.add_argument("--log-dir", default=cfg.log_dir, help="Search latest <log-dir>/*/server.pid")
    parser.add_argument("--port", type=int, default=cfg.port, help="Stop process listening on this port")
    parser.add_argument("--no-port", action="store_true", help="Do not use port-based process discovery")
    parser.add_argument("--systemd", action="store_true", help="Run systemctl stop for the service")
    parser.add_argument("--service-name", default="omnivoice-server", help="systemd service name")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Graceful shutdown timeout")
    parser.add_argument(
        "--no-kill",
        action="store_true",
        help="Do not SIGKILL remaining processes after timeout",
    )
    args = parser.parse_args(argv)

    if args.systemd:
        service = args.service_name
        if not service.endswith(".service"):
            service = f"{service}.service"
        proc = subprocess.run(["systemctl", "stop", service], check=False)
        return int(proc.returncode)

    pid_files: list[str] = list(args.pid_file)
    if not pid_files:
        discovered = latest_pid_file(args.log_dir)
        if discovered:
            pid_files.append(discovered)

    pids: set[int] = set()
    for pid_file in pid_files:
        pid = read_pid_file(pid_file)
        if pid is not None:
            pids.add(pid)
        else:
            print(f"Ignoring missing or invalid pid file: {pid_file}", file=sys.stderr)

    if not args.no_port and args.port:
        pids.update(pids_listening_on_port(args.port))

    if not pids:
        print("No OmniVoice server process found", file=sys.stderr)
        return 1
    return terminate_pids(pids, args.timeout_s, not args.no_kill)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"start", "serve"}:
        start(argv[1:])
        return
    if argv and argv[0] == "stop":
        raise SystemExit(stop(argv[1:]))
    if argv and argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="omnivoice-triton-server",
            description="OmniVoice Triton Server command line",
        )
        subparsers = parser.add_subparsers(dest="command")
        add_start_arguments(subparsers.add_parser("start", help="Start the server"), Settings())
        subparsers.add_parser("stop", help="Stop the server")
        parser.print_help()
        return
    start(argv)


if __name__ == "__main__":
    main()
