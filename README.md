# OmniVoice Triton Server

OpenAI-compatible OmniVoice TTS server built for batched GPU inference. The
server runs FastAPI workers for request handling and one or more independent GPU
inferer processes for model execution.

中文说明: [docs/README.zh-CN.md](docs/README.zh-CN.md)
Request API and language list: [docs/request.md](docs/request.md)

## Quick Start

Install the package:

```bash
pip install omnivoice-triton-server
```

Start a one-GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-triton-server start \
  --model-id /path/to/OmniVoice
```

Start a two-GPU server:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
omnivoice-triton-server start \
  --port 9194 \
  --model-id /path/to/OmniVoice \
  --gpu-inferer 2 \
  --max-batch-size 16 \
  --max-batch-latency 250 \
  --cuda-stream-count 2 \
  --runner-mode hybrid \
  --default-num-step 32
```

Installing the package adds the `omnivoice-triton-server` console command. The
module entrypoint is also available:

```bash
python -m omnivoice-triton-server start --port 9194 --model-id /path/to/OmniVoice
```

Stop a foreground/background process by port or pid file:

```bash
omnivoice-triton-server stop --port 9194
omnivoice-triton-server stop --pid-file logs/20260520-212301/server.pid --no-port
```

Install or update a systemd service:

```bash
omnivoice-triton-server install-service \
  --cuda-visible-devices 0,1 \
  --python "$(command -v python)" \
  --service-name omnivoice-server \
  --working-dir "$PWD" \
  -- \
  --port 9194 \
  --model-id /path/to/OmniVoice \
  --gpu-inferer 2 \
  --max-batch-size 16
```

Stop a systemd deployment:

```bash
omnivoice-triton-server stop --systemd --service-name omnivoice-server
```

Use the CUDA device ids that are correct for your machine. `scripts/start_server.sh`
is only a POSIX shell convenience wrapper around the same module entrypoint.

For source-tree deployments, the repository also includes:

```bash
scripts/install_systemd_service.sh \
  --cuda-visible-devices 0,1 \
  --python /path/to/python \
  --service-name omnivoice-server \
  -- \
  --port 9194 \
  --model-id /path/to/OmniVoice \
  --gpu-inferer 2 \
  --max-batch-size 16
```

Arguments after `--` are passed directly to `omnivoice-triton-server start`.
The script writes `/etc/omnivoice/<service>.sh` and
`/etc/systemd/system/<service>.service`, then reloads, enables, and restarts
the unit unless `--no-enable` or `--no-start` is used.

## Requirements

- Python 3.12 or newer.
- PyTorch, Triton, Transformers, FastAPI, and the packages in
  `requirements.txt`.
- NVIDIA GPU for inference. The default runner mode is `hybrid`.
- OmniVoice model files available either from a Hugging Face model id or a local
  path passed with `--model-id` / `OMNIVOICE_MODEL_ID`.

## Origin

This repository combines three code lines into one deployable service:

- `omnivoice-server`: API server, request routing, socket IPC, batching, metrics,
  deployment scripts, and tests.
- `omnivoice-triton`: Triton/hybrid inference backend pieces and CUDA/Triton
  acceleration code.
- `k2-fsa/OmniVoice`: selected OmniVoice model/runtime code under
  `src/modeling`.

The code is kept in a single tree because scheduling, chunking, graph capture,
and model invocation need to be tuned together. This is not a clean upstream
mirror; it is a server-oriented integration.

## Main Optimizations

- Multi-GPU serving with `--gpu-inferer N`. Each GPU inferer is a separate
  process with its own model weights on one CUDA device.
- FastAPI worker count defaults to the effective GPU inferer count when
  `--fastapi-workers` is not specified.
- Worker-side preprocessing: validation, semantic text chunking, duration
  splitting, clone reference audio ingestion, response formatting, and SSE
  framing run outside the inferer.
- Async local TCP socket IPC between workers and inferers. Clone reference audio
  is sent through the socket; the server does not create temporary prompt audio
  files for normal requests.
- Shared-memory metrics snapshots, so `/metrics` does not block generation.
- Clone audio prompt LRU cache controlled by
  `--max-clone-audio-prompt-cache`.
- `chunk_mode` request control:
  - `concurrent`: default. Clone chunks share the same clone prompt. Auto/design
    generate chunk 0 first, then use that result as continuity prompt for the
    remaining chunks, which can run concurrently.
  - `sequential`: each chunk uses the previous generated chunk as continuity
    prompt.
  - `none`: still chunks text, but estimates a larger chunk size from the model
    context limit to avoid unnecessary splitting.
- Mixed-language semantic chunking with CJK/Thai/Hangul character counting,
  non-CJK token counting, punctuation-aware recursive splitting, and balanced
  packing near the configured word limit.
- Fixed-shape CUDA Graph prewarming with compact batch/width buckets, memory
  headroom checks, automatic effective batch/width fallback on smaller GPUs,
  graph hit/miss metrics, and graph-aware microbatch splitting for oversized
  batches.

CPU inferer code was removed. Scale this server with GPU inferer processes.

## Important Arguments

- `--model-id`: local model path or Hugging Face model id.
- `--gpu-inferer`: number of GPU inferer processes to launch. The launcher
  clamps this to the number of visible CUDA devices.
- `--fastapi-workers`: API worker count. Defaults to effective GPU inferer
  count when omitted.
- `--max-batch-size`, `--max-batch-latency`: scheduler batching controls.
- `--cuda-stream-count`: backend worker streams per inferer.
- `--cuda-graph-min-width`, `--cuda-graph-max-width`: graph width controls.
- `--cuda-graph-auto-width-tokens-per-word`,
  `--cuda-graph-auto-max-width`: context-limit estimation for `chunk_mode=none`.
- `--default-num-step`: server default generation step count. Default: `32`.
  Requests may override it with `num_step`.
- `--max-clone-audio-prompt-cache`: clone prompt LRU size. Default: `32`.
- `--max-continuity-audio-tokens`,
  `--max-continuity-text-words`: chunk continuity prompt limits.
- `--text-chunk-words` and `--text-chunk-*`: chunking and packing controls.
- `--log-dir`, `--log-run-id`, `--log-file`, `--pid-file`: runtime log layout.

All settings can also be set with `OMNIVOICE_*` environment variables.
Python defaults live in `src/config.py`; shell scripts do not define service
defaults.

## API

Speech endpoint:

```bash
curl -X POST http://127.0.0.1:9194/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "input": "Hello from OmniVoice.",
    "voice": "auto",
    "response_format": "wav",
    "speed": 1.0,
    "chunk_mode": "concurrent",
    "num_step": 32
  }' \
  --output speech.wav
```

Voice design endpoint:

```bash
curl -X POST http://127.0.0.1:9194/v1/audio/design \
  -F 'text=Hello from a designed voice.' \
  -F 'instruct=female, young adult, moderate pitch' \
  -F 'chunk_mode=concurrent' \
  -F 'response_format=wav' \
  --output design.wav
```

Voice clone endpoint:

```bash
curl -X POST http://127.0.0.1:9194/v1/audio/clone \
  -F 'text=Hello from a cloned voice.' \
  -F 'ref_audio=@ref.wav;type=audio/wav' \
  -F 'ref_text=Text spoken in the reference audio.' \
  -F 'chunk_mode=concurrent' \
  -F 'response_format=wav' \
  --output clone.wav
```

Supported response formats are `wav` and raw `pcm`.

## Benchmark

These numbers are a reference result for the hardware and launch configuration
below. They are not a hardware-independent promise.

Test configuration:

- Hardware used by this service: 2 x NVIDIA GeForce RTX 3080, 20 GiB each.
- Launch: `--gpu-inferer 2 --fastapi-workers 2 --runner-mode hybrid --dtype fp16
  --max-batch-size 16 --max-batch-latency 250 --cuda-stream-count 2
  --default-num-step 32`.
- Load generator: short text requests scheduled at 100 req/s.
- Audio quality smoke: auto, design, and clone outputs were checked by ASR on
  both short and long texts.

### Throughput

| Workload | Wall time | Completed req/s | Generated audio | Audio realtime | RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short speech/design, `num_step=16`, 1000 requests | 36.717 s | 27.235 | 785.980 s | 21.408x | 0.0467 |
| Short speech/design, `num_step=32`, 1000 requests | 62.998 s | 15.874 | 785.690 s | 12.472x | 0.0802 |

`Audio realtime` is generated audio duration divided by wall time. `RTF` is
wall time divided by generated audio duration.

### Scheduler Efficiency

| Workload | Client requests | Backend tasks | Backend batches | Tasks/backend batch | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Short speech/design, `num_step=16` | 1,000 | 1,000 | 83 | 12.048 | Same traffic distribution as the 32-step row. |
| Short speech/design, `num_step=32` | 1,000 | 1,000 | 67 | 14.925 | Same traffic distribution as the 16-step row. |

The useful batching signal is `Tasks/backend batch`: higher means the scheduler
kept the GPU inferers fed with larger model batches.

### CUDA Graph Behavior

Current two-GPU graph plan:

- Graph entries per inferer: 15.
- Captured shapes per inferer:
  `(2,8,128)`, `(2,8,160)`, `(2,8,256)`, `(2,8,512)`,
  `(2,8,640)`, `(8,8,64)`, `(8,8,128)`, `(8,8,160)`, `(8,8,256)`,
  `(16,8,128)`, `(16,8,160)`, `(16,8,256)`,
  `(32,8,64)`, `(32,8,128)`, `(32,8,160)`.
- Optional `(4,8,512)` graph capture was skipped by the memory headroom guard on
  this launch, so startup kept the model online instead of forcing an OOM.

The graph miss count is the number to watch when changing chunking, graph width
buckets, or `--max-batch-size`; sustained misses usually mean requests are
falling outside the prewarmed shape plan.
On lower-memory GPUs, check `requested_max_width`, `max_width`,
`requested_max_business_batch_size`, `max_business_batch_size`, and
`skipped_shapes` in `/metrics` to confirm whether startup reduced the graph
plan to fit available VRAM.

## Development Checks

```bash
python -m py_compile \
  src/app.py src/audio.py src/chunking.py src/config.py src/infer_client.py \
  src/inferer.py src/launcher.py src/protocol.py

PYTHONPATH=src python tests/test_chunking.py
python tests/test_api.py
python tests/load_1000_rps100.py --total 1000 --rate 100 --concurrency-limit 512
python tests/load_mixed_1000.py --total 1000 --rate 100 --ref-audio /path/to/ref.wav
```

Runtime artifacts, logs, generated audio, model weights, and local environment
files are ignored by `.gitignore`.

See `docs/DEPLOYMENT.md` for operational details.
