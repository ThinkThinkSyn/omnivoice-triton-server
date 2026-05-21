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

## systemd 服务

可以用 `scripts/install_systemd_service.sh` 生成并注册 systemd unit：

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

`--` 后面的参数会原样传给 `python -m omnivoice_triton_server`。脚本会写入
`/etc/omnivoice/<service>.sh` 和
`/etc/systemd/system/<service>.service`，默认执行 `daemon-reload`、开机启用并
重启服务；只想生成 unit 时可以加 `--no-enable` 或 `--no-start`。

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

这些数据用于估算这台机器的容量，不是跨硬件承诺。

测试配置：

- 服务使用硬件：2 x NVIDIA GeForce RTX 3080, 20 GiB
- 测试机器共有 8 张可见 RTX 3080，本次服务使用 `CUDA_VISIBLE_DEVICES=6,7`
- 启动参数：`--gpu-inferer 2 --fastapi-workers 2 --runner-mode hybrid
  --dtype fp16 --max-batch-size 16 --max-batch-latency 250
  --cuda-stream-count 2 --num-step 32`
- 压测：1000 请求，目标到达速率 100 req/s；因为目标速率高于服务能力，延迟里包含排队时间
- 音频质量 smoke：auto、design、clone 的长短文本都经过 ASR 对比

### 吞吐和延迟

| 负载 | 总耗时 | 完成 req/s | 生成音频时长 | 音频实时倍数 | RTF | 平均延迟 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 短文本 speech/design | 61.553 s | 16.246 | 786.100 s | 12.771x | 0.0783 | 27.2485 s | 26.6065 s | 50.8381 s | 54.6932 s |
| 混合短/中/长 speech/design/clone | 247.159 s | 4.046 | 2,648.752 s | 10.717x | 0.0933 | 120.7916 s | 119.3613 s | 228.5990 s | 236.4849 s |

`音频实时倍数 = 生成音频时长 / 总耗时`，`RTF = 总耗时 / 生成音频时长`。

### 调度效率

| 负载 | 客户端请求 | 后端任务 | 任务/请求 | 后端 batch | 任务/backend batch | 后端任务/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 短文本 speech/design | 1,000 | 1,000 | 1.000 | 68 | 14.706 | 16.246 |
| 混合短/中/长 speech/design/clone | 1,000 | 1,733 | 1.733 | 67 | 25.866 | 7.011 |

混合测试里长文本会拆 chunk，所以后端任务数大于客户端请求数。这里真正用来判断
batching 效率的是 `任务/backend batch`。

### mixed 类型细分

| 类型 | 请求数 | 平均延迟 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| speech | 900 | 122.8774 s | 228.6261 s | 236.6119 s |
| design | 50 | 129.3640 s | 229.6973 s | 230.9176 s |
| clone | 50 | 74.6742 s | 143.6043 s | 145.6617 s |

### CUDA Graph

- 每个 inferer 预热 14 个 graph。
- 每个 inferer 的形状：
  `(2,8,128)`, `(2,8,160)`, `(2,8,256)`, `(2,8,512)`,
  `(8,8,64)`, `(8,8,128)`, `(8,8,160)`, `(8,8,256)`,
  `(16,8,128)`, `(16,8,160)`, `(16,8,256)`,
  `(32,8,64)`, `(32,8,128)`, `(32,8,160)`。
- mixed 1000 请求期间，扣除压测前计数后 graph hits 为 11,520，misses 为 32。
- 两个 inferer 观测到的最大后端 batch 分别是 49 和 46 个任务。

## 测试

```bash
PYTHONPATH=src python tests/test_chunking.py
python tests/test_api.py
python tests/load_1000_rps100.py --total 1000 --rate 100 --concurrency-limit 512
python tests/load_mixed_1000.py --total 1000 --rate 100 --ref-audio /path/to/ref.wav
```
