# OmniVoice Triton Server 中文说明

这是一个面向部署的 OmniVoice TTS 服务，提供 OpenAI 兼容接口、FastAPI
工作进程、多个 GPU inferer 进程、Triton/hybrid 后端和固定形状 CUDA Graph
预热。

## 来源

本仓库把三条代码线合成到一个部署树里：

- `omnivoice-server`：API、路由、socket IPC、调度、metrics、部署脚本、测试。
- `omnivoice-triton`：Triton/hybrid 推理后端和 CUDA/Triton 加速代码。
- `k2-fsa/OmniVoice`：`src/modeling` 下选取的模型与运行时代码。

这样做的原因是 chunking、调度、graph capture 和模型调用需要一起调。

## 主要优化

- 多 GPU 推理：`--gpu-inferer N`
- API worker 数默认跟随 GPU inferer 数量
- worker 侧前后处理：文本切分、时长拆分、clone 参考音频处理、响应格式化、SSE
- socket IPC 使用本地 asyncio TCP
- `/metrics` 使用共享内存快照
- clone prompt LRU cache：`--max-clone-audio-prompt-cache`
- `chunk_mode`：
  - `concurrent`：默认；clone 共享相同 prompt，auto/design 先出第一个 chunk，再把它作为 continuity prompt 给后续 chunk
  - `sequential`：逐 chunk 串行 continuity
  - `none`：仍会 chunk，但尽量按模型上下文上限拼大 chunk
- 混合语言语义切分：中日韩泰等按字符计，非 CJK 按 token 计
- 预热固定 CUDA Graph 形状，带 memory headroom 检查和 microbatch 拆分

CPU inferer 已移除，扩容只靠 GPU inferer。

## 启动

```bash
python -m venv .venv
. .venv/bin/activate
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

`scripts/start_server.sh` 只是一个 shell 包装器，真正入口是
`python -m omnivoice_triton_server`。

## 重要参数

- `--model-id`
- `--gpu-inferer`
- `--fastapi-workers`
- `--max-batch-size` / `--max-batch-latency`
- `--cuda-stream-count`
- `--cuda-graph-min-width` / `--cuda-graph-max-width`
- `--cuda-graph-auto-width-tokens-per-word` / `--cuda-graph-auto-max-width`
- `--num-step`
- `--max-clone-audio-prompt-cache`
- `--text-chunk-*`

所有设置也可以用 `OMNIVOICE_*` 环境变量覆盖。默认值在 `src/config.py`。

## Benchmark

测试机器：

- 2 x NVIDIA GeForce RTX 3080, 20 GiB
- 8 张可见 GPU
- 本次选择 `CUDA_VISIBLE_DEVICES=6,7`

启动参数：

- `--gpu-inferer 2`
- `--fastapi-workers 2`
- `--runner-mode hybrid`
- `--dtype fp16`
- `--max-batch-size 16`
- `--cuda-stream-count 2`
- `--num-step 32`
- 1000 请求，目标速率 100 req/s

### 总表

| 流量 | HTTP 200 | HTTP 错误 | 非法音频 | 后端错误 | 总耗时 | 完成速率 | 音频秒数 | RTF(耗时/音频) | p50 | p95 | p99 | max | 字节数 | 后端任务 | 后端批次数 | 平均 batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 短文本混合 speech/design | 1000 | 0 | 0 | 0 | 61.553 s | 16.246 | 786.100 s | 0.0783 | 26.6065 s | 50.8381 s | 54.6932 s | 55.8403 s | 37,743,800 | 1000 | 68 | 14.706 |
| 混合短/中/长 speech/design/clone | 1000 | 0 | 0 | 0 | 247.159 s | 4.046 | 2,648.752 s | 0.0933 | 119.3613 s | 228.5990 s | 236.4849 s | 236.6119 s | 127,184,080 | 1733 | 67 | 25.866 |

### mixed 细分

| 类型 | 数量 | 平均延迟 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| clone | 50 | 74.6742 s | 143.6043 s | 145.6617 s |
| speech | 900 | 122.8774 s | 228.6261 s | 236.6119 s |
| design | 50 | 129.3640 s | 229.6973 s | 230.9176 s |

两组 benchmark 都是 0 HTTP 失败、0 非法音频、0 backend errors。mixed
测试里长文本会拆 chunk，所以 backend task 数大于 HTTP 请求数，这是正常的。

## 测试

```bash
PYTHONPATH=src python tests/test_chunking.py
python tests/test_api.py
python tests/load_1000_rps100.py --total 1000 --rate 100 --concurrency-limit 512
python tests/load_mixed_1000.py --total 1000 --rate 100 --ref-audio /path/to/ref.wav
```

