# OmniVoice Triton Server

OpenAI-compatible OmniVoice TTS server with worker-side preprocessing, batched
multi-GPU inferer processes, and a compact prewarmed CUDA Graph shape plan.

This repository is derived from an `omnivoice-server` integration and selected
runtime code from `omnivoice-triton`. It also folds in selected model/runtime
source from upstream `k2-fsa/OmniVoice` under `src/modeling`. The source is kept
in one server tree instead of a separate `vendor/` directory so the API server,
batch scheduler, model runtime, and Triton kernels can be optimized together.

## What This Adds

- OpenAI-compatible `/v1/audio/speech`, `/v1/audio/design`, and
  `/v1/audio/clone` endpoints.
- Multi-GPU inferer processes: `--gpu-inferer N` starts one independent model
  process per visible GPU, clamped to the number of visible GPUs.
- Worker-side text chunking, duration splitting, clone prompt preparation, audio
  formatting, and SSE framing so inferers stay focused on model execution.
- Async local socket IPC between FastAPI workers and inferers; clone reference
  audio is sent over the socket rather than through temporary files.
- Shared-memory metrics snapshots so `/metrics` does not block inferer work.
- Clone audio prompt cache with configurable LRU size.
- Request `chunk_mode`:
  - `concurrent`: default; chunks that can share context are batched together.
  - `sequential`: preserves previous-chunk audio continuity one chunk at a time.
  - `none`: still chunks, but uses the model context limit to make chunks as
    large as safely possible.
- Mixed-language semantic chunking with configurable packing penalties.
- Startup CUDA Graph prewarming with compact width/batch buckets, memory
  headroom checks, and graph-aware microbatch splitting for oversized batches.
- Triton/hybrid runner patches for fixed-shape graph replay, width caching, and
  reduced eager fallback.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0
export OMNIVOICE_MODEL_ID=k2-fsa/OmniVoice
scripts/start_server.sh
```

Common 2-GPU launch:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMNIVOICE_MODEL_ID=/path/to/OmniVoice \
scripts/start_server.sh \
  --port 9194 \
  --gpu-inferer 2 \
  --max-batch-size 16 \
  --max-batch-latency 250 \
  --cuda-stream-count 2 \
  --runner-mode hybrid \
  --num-step 32
```

If `--fastapi-workers` is not specified, the launcher defaults API worker count
to the effective GPU inferer count.

## Useful Arguments

- `--model-id`: local model path or Hugging Face model id.
- `--gpu-inferer`: number of GPU inferer processes to launch.
- `--max-batch-size`, `--max-batch-latency`: scheduler batching controls.
- `--cuda-stream-count`: backend worker streams per inferer.
- `--cuda-graph-min-width`, `--cuda-graph-max-width`: graph width controls.
- `--cuda-graph-auto-width-tokens-per-word`, `--cuda-graph-auto-max-width`:
  automatic graph width estimation for `chunk_mode=none`.
- `--num-step`: global diffusion/generation step count, default `32`.
- `--max-clone-audio-prompt-cache`: clone prompt LRU size, default `32`.
- `--max-continuity-audio-tokens`, `--max-continuity-text-words`: chunk
  continuity prompt limits.
- `--text-chunk-words` and `--text-chunk-*`: semantic chunking and packing
  controls.
- `--log-dir`, `--log-run-id`, `--log-file`, `--pid-file`: runtime log layout.

All `Settings` fields can also be set with `OMNIVOICE_*` environment variables.

## API Example

```bash
curl -X POST http://127.0.0.1:9194/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "input": "Hello from OmniVoice.",
    "voice": "auto",
    "response_format": "wav",
    "speed": 1.0,
    "chunk_mode": "concurrent"
  }' \
  --output speech.wav
```

## Benchmark

Latest local benchmark on the project test host:

- GPUs: 2 inferers on the last two visible NVIDIA GPUs.
- API workers: 2, matching `--gpu-inferer 2`.
- Runner: `hybrid`, dtype `fp16`, `--max-batch-size 16`,
  `--cuda-stream-count 2`, `--num-step 32`.
- Request rate target: 100 req/s, 1000 total requests.

Results:

| Traffic | Success | Completion | Estimated RTF | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Short mixed speech/design | 1000/1000 | 16.246 req/s | 0.0783 | 50.84 s |
| Mixed short/medium/long speech/design/clone | 1000/1000 | 4.046 req/s | 0.0933 | 228.60 s |

Audio quality smoke validation used ASR comparison over auto, design, and clone
requests for both short and long text; all 6 validation cases passed.

## Development Checks

```bash
python -m py_compile \
  src/app.py src/audio.py src/chunking.py src/config.py src/infer_client.py \
  src/inferer.py src/launcher.py src/protocol.py

python tests/test_chunking.py
python tests/test_api.py
python tests/load_1000_rps100.py --total 1000 --rate 100 --concurrency-limit 512
python tests/load_mixed_1000.py --total 1000 --rate 100 --ref-audio /path/to/ref.wav
```

Runtime artifacts, logs, generated audio, model weights, and local environment
files are ignored by `.gitignore`.

See `docs/DEPLOYMENT.md` for operational notes and endpoint details.
