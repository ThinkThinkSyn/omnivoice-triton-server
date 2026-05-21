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
pip install -e .

CUDA_VISIBLE_DEVICES=0,1 \
python -m omnivoice_triton_server \
  --port 9194 \
  --model-id /path/to/OmniVoice \
  --gpu-inferer 2 \
  --max-batch-size 16 \
  --max-batch-latency 250 \
  --cuda-stream-count 2 \
  --runner-mode hybrid \
  --num-step 32
```

`scripts/start_server.sh` is a POSIX shell convenience wrapper around
`python -m omnivoice_triton_server`. It is intentionally small:

- uses `python` unless `OMNIVOICE_PYTHON` is set,
- prepends this repository's `src/` to `PYTHONPATH`.

Service defaults live in `src/config.py` and can be overridden by CLI arguments
or `OMNIVOICE_*` environment variables. They are not defined in the shell
wrapper.

If `--fastapi-workers` is omitted, the launcher uses the effective GPU inferer
count as the worker count. If `--gpu-inferer` exceeds visible GPUs, it is clamped
to the number of visible GPUs. Startup fails if no GPU inferer can be started.

## Systemd Service

Use `scripts/install_systemd_service.sh` to generate and register a Linux
systemd unit. The script requires the CUDA device list and accepts normal
launcher arguments after `--`.

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
  --num-step 32
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
- `--num-step`, `--guidance-scale`, `--denoise/--no-denoise`, `--t-shift`
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
  "chunk_mode": "concurrent"
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
- `request_timeout_s`: per-request timeout override

Unknown JSON fields are preserved under `extra_fields` for future plumbing and
returned in `X-OmniVoice-Extra-Fields` for non-streaming responses.

Server-controlled fields are rejected from client requests:

- `audio_chunk_duration`
- `audio_chunk_threshold`
- `batch_mode`
- `position_temperature`
- `num_step`
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
to the target word count while allowing a small soft overflow when that keeps a
better semantic boundary.

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
- width buckets include the default short/medium widths and only add expensive
  wide shapes when memory headroom allows,
- runtime inputs pad upward to the nearest prewarmed shape,
- out-of-plan wide single-chunk batches can split into graph-coverable
  microbatches instead of forcing large eager fallback.

`/metrics` exposes graph entries, hits, misses, capture failures, skipped
shapes, memory snapshots, batch counters, queue ages, and error counters.

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

Latest local benchmark on one project test host. These numbers are meant for
capacity planning on this hardware and launch configuration.

- Hardware used by this service: 2 x NVIDIA GeForce RTX 3080, 20 GiB each.
- Test host GPU inventory: 8 visible RTX 3080 GPUs.
- Devices selected for this run: `CUDA_VISIBLE_DEVICES=6,7`.
- Launch: `--gpu-inferer 2 --fastapi-workers 2 --runner-mode hybrid --dtype fp16
  --max-batch-size 16 --max-batch-latency 250 --cuda-stream-count 2
  --num-step 32`.
- Load: 1000 requests at a 100 req/s target arrival rate.

### Throughput And Latency

| Workload | Wall time | Completed req/s | Generated audio | Audio realtime | RTF | Mean latency | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short speech/design | 61.553 s | 16.246 | 786.100 s | 12.771x | 0.0783 | 27.2485 s | 26.6065 s | 50.8381 s | 54.6932 s |
| Mixed short/medium/long speech/design/clone | 247.159 s | 4.046 | 2,648.752 s | 10.717x | 0.0933 | 120.7916 s | 119.3613 s | 228.5990 s | 236.4849 s |

### Scheduler Efficiency

| Workload | Client requests | Backend tasks | Tasks/request | Backend batches | Tasks/backend batch | Backend task/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short speech/design | 1,000 | 1,000 | 1.000 | 68 | 14.706 | 16.246 |
| Mixed short/medium/long speech/design/clone | 1,000 | 1,733 | 1.733 | 67 | 25.866 | 7.011 |

### Mixed Workload Breakdown

| Kind | Requests | Mean latency | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| speech | 900 | 122.8774 s | 228.6261 s | 236.6119 s |
| design | 50 | 129.3640 s | 229.6973 s | 230.9176 s |
| clone | 50 | 74.6742 s | 143.6043 s | 145.6617 s |

### CUDA Graph Behavior

- Graph entries per inferer: 14.
- Captured shapes per inferer:
  `(2,8,128)`, `(2,8,160)`, `(2,8,256)`, `(2,8,512)`,
  `(8,8,64)`, `(8,8,128)`, `(8,8,160)`, `(8,8,256)`,
  `(16,8,128)`, `(16,8,160)`, `(16,8,256)`,
  `(32,8,64)`, `(32,8,128)`, `(32,8,160)`.
- Mixed 1000-request run graph delta: 11,520 hits and 32 misses after
  subtracting pre-run counters.
- Max backend batch seen by the two inferers: 49 and 46 tasks.

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
