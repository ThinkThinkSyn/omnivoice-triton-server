# OmniVoice Triton Server

OpenAI-compatible OmniVoice TTS server built for batched GPU inference. The
server runs FastAPI workers for request handling and one or more independent GPU
inferer processes for model execution.

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
  headroom checks, graph hit/miss metrics, and graph-aware microbatch splitting
  for oversized batches.

CPU inferer code was removed. Scale this server with GPU inferer processes.

## Requirements

- Python 3.12 or newer.
- PyTorch, Triton, Transformers, FastAPI, and the packages in
  `requirements.txt`.
- NVIDIA GPU for inference. The default runner mode is `hybrid`.
- OmniVoice model files available either from a Hugging Face model id or a local
  path passed with `--model-id` / `OMNIVOICE_MODEL_ID`.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0
export OMNIVOICE_MODEL_ID=k2-fsa/OmniVoice
scripts/start_server.sh
```

Two-GPU example:

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

`CUDA_VISIBLE_DEVICES` is a deployment choice. The benchmark below used
`CUDA_VISIBLE_DEVICES=6,7` on one 8-GPU test server because those two devices
were selected for that run; use the device ids that are correct on your machine.

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
- `--num-step`: global generation step count. Default: `32`.
- `--max-clone-audio-prompt-cache`: clone prompt LRU size. Default: `32`.
- `--max-continuity-audio-tokens`,
  `--max-continuity-text-words`: chunk continuity prompt limits.
- `--text-chunk-words` and `--text-chunk-*`: chunking and packing controls.
- `--log-dir`, `--log-run-id`, `--log-file`, `--pid-file`: runtime log layout.

All settings can also be set with `OMNIVOICE_*` environment variables.

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
    "chunk_mode": "concurrent"
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

These numbers are from one local test server and are included to describe the
observed configuration, not as a hardware-independent promise.

Hardware and launch configuration:

- GPU hardware: 2 x NVIDIA GeForce RTX 3080, 20 GiB each as reported by
  `nvidia-smi`.
- Test server had 8 visible RTX 3080 GPUs; this run selected GPU ids `6,7` with
  `CUDA_VISIBLE_DEVICES=6,7`.
- GPU inferers: `--gpu-inferer 2`.
- API workers: 2.
- Runner: `hybrid`.
- Dtype: `fp16`.
- Batch: `--max-batch-size 16`.
- CUDA streams: `--cuda-stream-count 2`.
- Generation steps: `--num-step 32`.
- Load shape: 1000 requests scheduled at 100 req/s.

Results:

| Traffic | Success | Completion | Estimated RTF | p95 latency | Backend tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short mixed speech/design | 1000/1000 | 16.246 req/s | 0.0783 | 50.84 s | 1000 |
| Mixed short/medium/long speech/design/clone | 1000/1000 | 4.046 req/s | 0.0933 | 228.60 s | 1733 |

The mixed test includes chunked long requests and clone/design paths, so backend
task count is higher than HTTP request count. Audio quality smoke validation
used ASR comparison over auto, design, and clone requests for both short and
long text; all 6 validation cases passed.

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
