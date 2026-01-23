# SGLang Integration

Alternative inference backend to vLLM with ~27% throughput improvement via RadixAttention prefix caching.

## Benchmark Results (2x H100)

| Metric | vLLM | SGLang | Improvement |
|--------|------|--------|-------------|
| Throughput | 300k tok/s | 382k tok/s | **+27%** |
| Latency | 1,705 ms | 1,341 ms | **-21%** |

## Architecture

The SGLang integration consists of three main components:

1. **Router** (`src/prime_rl/inference/server.py`): Main entrypoint that routes to vLLM or SGLang based on config
2. **SGLang Server** (`src/prime_rl/inference/sglang/server.py`): FastAPI proxy server with prime-rl extensions
3. **Worker Extensions** (`src/prime_rl/inference/sglang/worker/`): Weight broadcast implementations

The router parses the configuration and delegates to the appropriate backend.

## Quick Start

### Method 1: Via Config File

```toml
# config.toml
[inference]
backend = "sglang"

[inference.model]
name = "Qwen/Qwen2.5-7B-Instruct"
```

```bash
uv run inference @ config.toml
```

### Method 2: Via CLI Arguments

```bash
uv run inference @ config.toml --inference.backend sglang
```

### Method 3: In RL Training

```toml
# rl_config.toml
[inference]
backend = "sglang"
```

```bash
uv run rl @ rl_config.toml
```

## Run Benchmarks

```bash
# Setup separate environments (required due to dependency conflicts)
./scripts/setup_env.sh

# Run vLLM vs SGLang comparison
./scripts/run_benchmarks.sh
```

## API Endpoints

SGLang server implements all prime-rl required endpoints:

| Endpoint | Description | Status |
|----------|-------------|--------|
| `GET /health` | Health check | ✅ |
| `GET /v1/models` | List models | ✅ |
| `POST /v1/chat/completions` | Chat completions | ✅ |
| `POST /v1/chat/completions/tokens` | Pre-tokenized input | ✅ |
| `POST /update_weights` | Update weights from disk | ✅ |
| `POST /reload_weights` | Reset to original weights | ✅ |
| `POST /init_broadcaster` | Initialize NCCL broadcast | ✅ |
| `POST /v1/load_lora_adapter` | Load LoRA adapter | ✅ |
| `POST /v1/unload_lora_adapter` | Unload LoRA adapter | ✅ |

## Testing

### Integration Tests

```bash
# Test SGLang RL training (requires GPU)
uv run pytest tests/integration/test_rl_sglang.py -v
```

### Manual Testing

```bash
# 1. Start SGLang server
uv run inference @ configs/debug/infer.toml --inference.backend sglang

# 2. Test health
curl http://localhost:8000/health

# 3. Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

## Troubleshooting

### Dependency Conflicts

SGLang and vLLM have conflicting dependencies. Use separate virtual environments:

```bash
./scripts/setup_env.sh  # Creates venv_vllm and venv_sglang
```

### Performance Tuning

```toml
[inference]
backend = "sglang"
gpu_memory_utilization = 0.9

[inference.parallel]
tp = 2  # Tensor parallelism for multi-GPU
```

## Implementation Details

### Subprocess Architecture

SGLang runs in subprocess mode to avoid event loop conflicts:
1. Prime-RL FastAPI server starts on configured port (e.g., 8000)
2. SGLang subprocess starts on port+1000 (e.g., 9000)
3. FastAPI proxies requests to SGLang subprocess

### Weight Update Flow

```
Trainer → Orchestrator → Prime-RL FastAPI → SGLang subprocess → Model reload
```

## Known Limitations

1. **Dependency Conflicts**: Cannot install both vLLM 0.12.0 and SGLang >=0.4.4 in same environment
2. **LoRA Support**: Experimental (use `enable_lora = true` in config)
3. **Multi-API Server**: Not supported (use `dp` instead)
