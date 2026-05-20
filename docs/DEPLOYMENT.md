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

Latest local benchmark on one project test host. These numbers describe that
hardware and launch configuration; they are not meant to imply the same device
ids or throughput on other machines.

- GPU hardware: 2 x NVIDIA GeForce RTX 3080, 20 GiB each as reported by
  `nvidia-smi`.
- Test host GPU inventory: 8 visible RTX 3080 GPUs.
- Devices selected for this run: `CUDA_VISIBLE_DEVICES=6,7`.
- GPU inferers: `--gpu-inferer 2`.
- API workers: 2.
- Runner: `hybrid`.
- Dtype: `fp16`.
- Batch: `--max-batch-size 16`.
- CUDA streams: `--cuda-stream-count 2`.
- Generation steps: `--num-step 32`.
- Load: 1000 requests at a 100 req/s target arrival rate.

| Traffic | HTTP 200 | HTTP errors | Invalid audio | Backend errors | Wall time | Completion RPS | Audio seconds | RTF wall/audio | p50 | p95 | p99 | max | Bytes | Backend tasks | Backend batches | Avg batch size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short mixed speech/design | 1000 | 0 | 0 | 0 | 61.553 s | 16.246 | 786.100 s | 0.0783 | 26.6065 s | 50.8381 s | 54.6932 s | 55.8403 s | 37,743,800 | 1000 | 68 | 14.706 |
| Mixed short/medium/long speech/design/clone | 1000 | 0 | 0 | 0 | 247.159 s | 4.046 | 2,648.752 s | 0.0933 | 119.3613 s | 228.5990 s | 236.4849 s | 236.6119 s | 127,184,080 | 1733 | 67 | 25.866 |

Mixed benchmark per-kind breakdown:

| Kind | Count | Mean latency | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| clone | 50 | 74.6742 s | 143.6043 s | 145.6617 s |
| speech | 900 | 122.8774 s | 228.6261 s | 236.6119 s |
| design | 50 | 129.3640 s | 229.6973 s | 230.9176 s |

Both benchmarks recorded `0` HTTP failures, `0` invalid audio outputs, and
`0` backend errors. The mixed test includes chunked long requests and clone and
design paths, so backend task count is higher than HTTP request count. ASR
quality smoke validation covered auto, design, and clone modes with short and
long text; all 6 validation cases passed.

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
