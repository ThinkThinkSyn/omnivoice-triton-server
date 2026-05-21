# Deployment Notes

## Project Origin

This project combines an `omnivoice-server` API/service integration with
selected runtime and acceleration code from `omnivoice-triton`, plus selected
model/runtime source from upstream `k2-fsa/OmniVoice` under `src/modeling`.

The combined tree lets the server coordinate API workers, socket IPC, batching,
CUDA Graph shape planning, Triton kernels, and OmniVoice model calls without a
separate vendored package boundary.

## Runtime Topology

- FastAPI workers accept OpenAI-compatible requests and handle validation,
  text chunking, duration splitting, clone reference audio ingestion, response
  formatting, and SSE framing.
- GPU inferers are independent child processes. Each inferer owns one model copy
  on one visible GPU.
- FastAPI workers talk to inferers through local asyncio TCP sockets.
- `/metrics` reads shared-memory snapshots written by inferers, so metrics calls
  do not need to synchronize with active generation.
- Clone prompt cache metadata is shared across processes, and GPU inferers keep
  local prompt tensors for fast reuse.

CPU inferer support was intentionally removed. Scaling is through multiple GPU
inferer processes.

## Start

```bash
pip install omnivoice-triton-server

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

`scripts/start_server.sh` is a POSIX shell convenience wrapper around
`python -m omnivoice-triton-server start`. It is intentionally small:

- uses `python` unless `OMNIVOICE_PYTHON` is set,
- prepends this repository's `src/` to `PYTHONPATH`.

Service defaults live in `src/config.py` and can be overridden by CLI arguments
or `OMNIVOICE_*` environment variables. They are not defined in the shell
wrapper.

If `--fastapi-workers` is omitted, the launcher uses the effective GPU inferer
count as the worker count. If `--gpu-inferer` exceeds visible GPUs, it is clamped
to the number of visible GPUs. Startup fails if no GPU inferer can be started.

Stop commands:

```bash
omnivoice-triton-server stop --port 9194
omnivoice-triton-server stop --pid-file logs/<run-id>/server.pid --no-port
omnivoice-triton-server stop --systemd --service-name omnivoice-server
```

## Systemd Service

Use `omnivoice-triton-server install-service` to generate and register a Linux
systemd unit from a pip install. The command requires the CUDA device list and
accepts normal launcher arguments after `--`.

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
  --max-batch-size 16 \
  --max-batch-latency 250 \
  --cuda-stream-count 2 \
  --runner-mode hybrid \
  --default-num-step 32
```

The source tree also includes `scripts/install_systemd_service.sh`, which uses
the same service layout:

```bash
scripts/install_systemd_service.sh \
  --cuda-visible-devices 0,1 \
  --python /path/to/python \
  --service-name omnivoice-server \
  -- \
  --port 9194 \
  --model-id /path/to/OmniVoice \
  --gpu-inferer 2 \
  --max-batch-size 16 \
  --max-batch-latency 250 \
  --cuda-stream-count 2 \
  --runner-mode hybrid \
  --default-num-step 32
```

The installer writes:

- `/etc/omnivoice/<service>.sh`: wrapper with `CUDA_VISIBLE_DEVICES`,
  `PYTHONPATH`, extra `--env KEY=VALUE` values, and launcher arguments.
- `/etc/systemd/system/<service>.service`: systemd unit using the wrapper.

By default it runs `systemctl daemon-reload`, enables the service at boot, and
restarts it immediately. Use `--no-enable` or `--no-start` when staging a unit
without changing the active service.

## Important Arguments

- `--host`, `--port`
- `--fastapi-workers`
- `--model-id`, `--runner-mode`, `--dtype`, `--device`
- `--gpu-inferer`
- `--request-timeout-s`, `--infer-start-timeout-s`
- `--max-batch-size`, `--max-batch-latency`
- `--cuda-stream-count`
- `--cuda-graph-min-width`, `--cuda-graph-max-width`
- `--cuda-graph-auto-width-tokens-per-word`, `--cuda-graph-auto-max-width`
- `--default-num-step`, `--guidance-scale`, `--denoise/--no-denoise`, `--t-shift`
- `--position-temperature`, `--class-temperature`, `--layer-penalty-factor`
- `--audio-chunk-duration`, `--audio-chunk-threshold`
- `--postprocess-output/--no-postprocess-output`
- `--max-clone-audio-prompt-cache`
- `--max-continuity-audio-tokens`, `--max-continuity-text-words`
- `--text-chunk-words`
- `--text-chunk-soft-overflow-ratio`
- `--text-chunk-same-sentence-penalty`
- `--text-chunk-sentence-boundary-penalty`
- `--text-chunk-fragment-boundary-penalty`
- `--text-chunk-short-underfill-ratio`
- `--text-chunk-short-underfill-penalty`
- `--log-dir`, `--log-run-id`, `--log-file`, `--pid-file`,
  `--log-retention-days`

All settings are also available as `OMNIVOICE_*` environment variables.

## API

### Speech

```http
POST /v1/audio/speech
Content-Type: application/json
```

```json
{
  "model": "tts-1",
  "input": "Hello from OmniVoice.",
  "voice": "auto",
  "response_format": "wav",
  "speed": 1.0,
  "chunk_mode": "concurrent",
  "num_step": 32
}
```

Supported model ids:

- `omnivoice`
- `tts-1`
- `tts-1-hd`

Supported fields:

- `input`: required text
- `voice`: `auto`, OpenAI voice names, or `design:<instructions>`
- `speaker`: alias for `voice`
- `instructions`: explicit design prompt; forces design mode
- `response_format`: `wav` or raw `pcm`
- `speed`: `0.25` to `4.0`
- `duration`: optional target duration, `0.05` to `120.0`
- `language`: optional language hint
- `stream`: `true` for SSE
- `chunk_mode`: `concurrent`, `sequential`, or `none`
- `num_step`: optional per-request generation step count, `1` to `128`
- `request_timeout_s`: per-request timeout override

Unknown JSON fields are preserved under `extra_fields` for future plumbing and
returned in `X-OmniVoice-Extra-Fields` for non-streaming responses.

Server-controlled fields are rejected from client requests:

- `audio_chunk_duration`
- `audio_chunk_threshold`
- `batch_mode`
- `position_temperature`
- `postprocess_output`

### Voice Design

```bash
curl -X POST http://127.0.0.1:9194/v1/audio/design \
  -F 'text=Hello from a designed voice.' \
  -F 'instruct=female, young adult, moderate pitch' \
  -F 'chunk_mode=concurrent' \
  -F 'response_format=wav' \
  --output design.wav
```

### Voice Clone

```bash
curl -X POST http://127.0.0.1:9194/v1/audio/clone \
  -F 'text=Hello from a cloned voice.' \
  -F 'ref_audio=@ref.wav;type=audio/wav' \
  -F 'ref_text=Text spoken in the reference audio.' \
  -F 'chunk_mode=concurrent' \
  -F 'response_format=wav' \
  --output clone.wav
```

`/v1/audio/clone` also accepts `ref_audio_base64`.

## Chunking

Text chunking runs in API workers, not inferers. It uses a mixed-language word
counter:

- CJK, Japanese kana, Hangul, Thai, and similar character-level scripts count
  per character.
- Non-CJK spans split on whitespace and punctuation.
- Number groups and emoji can count as one word each.

The splitter recursively prefers paragraph, newline, sentence, semicolon/colon,
comma, and whitespace boundaries. It then scores candidate chunks to stay close
to the target word count while allowing bounded soft overflow when that preserves
a better semantic boundary or avoids very short fragments.

`chunk_mode` controls chunk execution:

- `concurrent`: default. Clone chunks share the same clone prompt. Auto/design
  generate the first chunk, use it as a continuity prompt, then run the remaining
  chunks concurrently.
- `sequential`: each chunk uses the previous chunk as continuity prompt.
- `none`: still chunks, but derives a larger max word count from the model
  context limit and then uses sequential execution.

## Batching And CUDA Graphs

The scheduler batches chunk jobs across requests. The current grouping key is
mode-focused so compatible requests with different speed, duration, language, or
voice prompt values can still batch when the model path supports per-item lists.

CUDA Graphs are captured at startup for a compact shape plan:

- business batch buckets are powers-of-two style high-value buckets up to the
  effective max batch,
- mandatory graph capture first tries the requested max width and max batch,
  then falls back through smaller effective widths and batch sizes when the GPU
  cannot keep the configured memory headroom,
- width buckets include the default short/medium widths and only add optional
  expensive wide shapes when memory headroom allows,
- runtime inputs pad upward to the nearest prewarmed shape,
- out-of-plan wide single-chunk batches can split into graph-coverable
  microbatches instead of forcing large eager fallback.

`/metrics` exposes graph entries, hits, misses, capture failures, skipped
shapes, requested/effective graph width and batch limits, memory snapshots,
batch counters, queue ages, and error counters.

## Metrics

Health and metrics:

```bash
curl http://127.0.0.1:9194/health
curl http://127.0.0.1:9194/metrics
```

Important fields:

- `pending_tasks`
- `queued_batches`, `queued_tasks`
- `running_batches`
- `total_batches`, `total_tasks`, `total_errors`
- `avg_batch_size`, `avg_batch_elapsed_s`, `avg_queue_wait_ms`
- `max_batch_size_seen`, `last_batch`
- `total_pcm_bytes`, `total_empty_audio_fallbacks`
- `cuda_graph_cache`

## Benchmarks

Latest local benchmark. These numbers are meant for capacity planning on this
hardware class and launch configuration.

- Hardware used by this service: 2 x NVIDIA GeForce RTX 3080, 20 GiB each.
- Launch: `--gpu-inferer 2 --fastapi-workers 2 --runner-mode hybrid --dtype fp16
  --max-batch-size 16 --max-batch-latency 250 --cuda-stream-count 2
  --default-num-step 32`.
- Load: short text requests at a 100 req/s target arrival rate.

### Throughput

| Workload | Wall time | Completed req/s | Generated audio | Audio realtime | RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short speech/design, `num_step=16`, 1000 requests | 36.717 s | 27.235 | 785.980 s | 21.408x | 0.0467 |
| Short speech/design, `num_step=32`, 1000 requests | 62.998 s | 15.874 | 785.690 s | 12.472x | 0.0802 |

### Scheduler Efficiency

| Workload | Client requests | Backend tasks | Backend batches | Tasks/backend batch | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Short speech/design, `num_step=16` | 1,000 | 1,000 | 83 | 12.048 | Same traffic distribution as the 32-step row. |
| Short speech/design, `num_step=32` | 1,000 | 1,000 | 67 | 14.925 | Same traffic distribution as the 16-step row. |

### CUDA Graph Behavior

- Graph entries per inferer: 15.
- Captured shapes per inferer:
  `(2,8,128)`, `(2,8,160)`, `(2,8,256)`, `(2,8,512)`,
  `(2,8,640)`, `(8,8,64)`, `(8,8,128)`, `(8,8,160)`, `(8,8,256)`,
  `(16,8,128)`, `(16,8,160)`, `(16,8,256)`,
  `(32,8,64)`, `(32,8,128)`, `(32,8,160)`.
- Optional `(4,8,512)` graph capture was skipped by the memory headroom guard on
  this launch, so startup kept the model online instead of forcing an OOM.

Audio quality smoke validation covered auto, design, and clone modes with short
and long text.

## Test Commands

```bash
PYTHONPATH=src python tests/test_chunking.py
python tests/test_api.py
python tests/load_1000_rps100.py \
  --total 1000 \
  --rate 100 \
  --concurrency-limit 512 \
  --out tmp/test-artifacts/load_1000_rps100_results.json

python tests/load_mixed_1000.py \
  --total 1000 \
  --rate 100 \
  --concurrency 512 \
  --chunk-mode concurrent \
  --ref-audio /path/to/ref.wav \
  --out tmp/test-artifacts/mixed_1000_results.json
```

Use `python -m py_compile src/*.py` for a quick syntax check.

Generated artifacts are written under ignored `tmp/`, `logs/`, and `run/`.
Model weights and exported media are ignored as well.

## Current Limitations

- Only `wav` and raw `pcm` response formats are implemented.
- SSE is chunk-level compatibility, not model-internal streaming.
- The socket protocol uses newline-delimited JSON with base64 audio payloads.
- Unknown `extra_fields` are preserved but not yet forwarded into
  `model.generate`.
