# SGLang Integration for prime-rl

This document describes how to use SGLang as an alternative inference backend to vLLM in prime-rl.

## Overview

SGLang is a high-performance inference engine that provides throughput improvements over vLLM for RL training workloads with shared prefixes through RadixAttention automatic prefix caching.

## Quick Start

### Using SGLang Backend

Simply add `--backend sglang` to your inference server command:
```bash
# With vLLM (default)
python -m prime_rl.inference.server --model.name Qwen/Qwen2.5-0.5B

# With SGLang
python -m prime_rl.inference.server --backend sglang --model.name Qwen/Qwen2.5-0.5B
```

### Configuration

You can set the backend in your TOML config file:
```toml
[inference]
backend = "sglang"  # or "vllm"

[inference.model]
name = "Qwen/Qwen2.5-0.5B"

[inference.parallel]
tp = 1
dp = 1
```

## Feature Comparison

| Feature | vLLM | SGLang |
|---------|------|--------|
| OpenAI-compatible API | ✅ | ✅ |
| Tensor Parallelism | ✅ | ✅ |
| Data Parallelism | ✅ | ✅ |
| LoRA Support | ✅ | ✅ |
| Weight Updates (Filesystem) | ✅ | ✅ |
| Weight Updates (NCCL) | ✅ | ✅ |
| Token-in Endpoint | ✅ | ✅ |
| CUDA Graphs | ✅ | ✅ |
| RadixAttention | ❌ | ✅ |
| Chunked Prefill | ✅ | ✅ |

## Benchmarking

### Automated Benchmark

Run the complete benchmark suite:
```bash
./scripts/run_benchmarks.sh
```

This script will:
1. Run vLLM with prefix caching enabled
2. Clean GPU memory
3. Run SGLang with RadixAttention
4. Output comparison results

Results saved to: `prefix_results_vllm.json` and `prefix_results_sglang.json`

**Requirements:**
- GPU with CUDA support
- Environments set up via `./scripts/setup_env.sh`

**Validated Results (Qwen2.5-7B, 5K requests):**
- SGLang: 4,539 tokens/sec (23% faster than vLLM)
- SGLang: 29ms TTFT (37% faster than vLLM)

See [docs/benchmarks/RESULTS.md](benchmarks/RESULTS.md) for detailed results.

### Manual Benchmarking

For custom benchmark parameters:
```bash
# Start vLLM server
python -m prime_rl.inference.server --backend vllm --model.name Qwen/Qwen2.5-0.5B &

# Run benchmark
python scripts/benchmark_backends.py \
    --base-url http://localhost:8000/v1 \
    --backend vllm \
    --num-requests 100 \
    --concurrency 16 \
    --output results_vllm.json

# Stop vLLM server, start SGLang server
pkill -f "prime_rl.inference.server"
python -m prime_rl.inference.server --backend sglang --model.name Qwen/Qwen2.5-0.5B &

# Run benchmark and compare
python scripts/benchmark_backends.py \
    --base-url http://localhost:8000/v1 \
    --backend sglang \
    --num-requests 100 \
    --concurrency 16 \
    --output results_sglang.json \
    --compare-file results_vllm.json
```

## API Compatibility

The SGLang backend provides the same API endpoints as the vLLM backend:

### Standard Endpoints
- `POST /v1/chat/completions` - Standard chat completions
- `POST /v1/completions` - Text completions
- `GET /v1/models` - List available models
- `GET /health` - Health check

### prime-rl Specific Endpoints
- `POST /update_weights` - Update model weights from disk
- `POST /init_broadcaster` - Initialize NCCL weight broadcast
- `POST /v1/chat/completions/tokens` - Chat with pre-tokenized input
- `POST /v1/load_lora_adapter` - Load a LoRA adapter
- `POST /v1/unload_lora_adapter` - Unload a LoRA adapter

## Weight Update Mechanism

### Filesystem Mode (Default)

The filesystem weight update works the same way with both backends:

1. Trainer saves checkpoint to shared filesystem
2. Orchestrator calls `/update_weights` with the checkpoint path
3. Inference server loads weights from disk
```python
# From orchestrator
await admin_client.post("/update_weights", json={"weight_dir": "/path/to/checkpoint"})
```

### NCCL Mode

For NCCL weight broadcast:

1. Initialize broadcast group via `/init_broadcaster`
2. Trainer broadcasts weights via NCCL
3. Inference server receives weights directly in GPU memory
```python
# Initialize NCCL group
await admin_client.post("/init_broadcaster", json={
    "host": "master_host",
    "port": 12345,
    "server_rank": 0,
    "num_inference_server": 1,
    "timeout": 300,
})
```

## Performance Tips

1. **RadixAttention Benefits**: SGLang automatically shares KV cache across requests with common prefixes, beneficial for RL training where system prompts are reused.

2. **Chunked Prefill**: For long sequences:
```bash
   python -m prime_rl.inference.server --backend sglang --chunked-prefill-size 8192
```

3. **Memory Configuration**: Adjust GPU memory utilization:
```bash
   python -m prime_rl.inference.server --backend sglang --gpu_memory_utilization 0.85
```

4. **Data Parallelism**: For higher throughput:
```bash
   python -m prime_rl.inference.server --backend sglang --parallel.dp 2
```

## Troubleshooting

### SGLang server fails to start
- Ensure you have CUDA installed and working
- Check that the model can fit in GPU memory
- Try with `--enforce-eager` to disable CUDA graphs

### Weight updates fail
- Ensure the checkpoint path is accessible to the inference server
- Check filesystem permissions
- Verify the checkpoint format is compatible

### Slower than expected throughput
- Check if CUDA graphs are enabled (disable with `--enforce-eager` for debugging)
- Monitor GPU utilization with `nvidia-smi`
- Ensure batch sizes are optimal for your hardware

## Contributing

If you encounter issues or have improvements for the SGLang integration:

1. Open an issue on GitHub
2. Include benchmark results showing the performance difference
3. Provide your configuration and hardware details
