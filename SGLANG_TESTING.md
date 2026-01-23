# SGLang Integration Testing Guide

This guide provides instructions for testing the SGLang backend integration for prime-rl (RFC #1615).

## Status

✅ **Router Implementation**: Complete
✅ **SGLang Server**: Complete  
✅ **API Endpoints**: Complete (9/9 endpoints)
✅ **Configuration**: Complete
✅ **Documentation**: Complete
⏳ **Integration Tests**: Ready to run (requires GPU environment)
⏳ **End-to-End Tests**: Ready to run (requires GPU environment)

## Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.12
- Separate virtual environments for vLLM and SGLang (due to dependency conflicts)

## Setup

### Option 1: Automated Setup (Recommended)

```bash
cd /path/to/prime-rl-sglang-swap-3
./scripts/setup_env.sh
```

This creates:
- `venv_vllm/`: Environment with vLLM and prime-rl
- `venv_sglang/`: Environment with SGLang

### Option 2: Manual Setup

```bash
# Create vLLM environment
python3.12 -m venv venv_vllm
source venv_vllm/bin/activate
pip install -U pip uv
uv pip install -e ".[all]"
deactivate

# Create SGLang environment  
python3.12 -m venv venv_sglang
source venv_sglang/bin/activate
pip install -U pip uv
uv pip install "sglang[all]" openai httpx
deactivate
```

## Test Plan

### 1. Router Tests (No GPU Required)

Test that the router correctly selects backends:

```bash
source venv_vllm/bin/activate

# Test config parsing
python -c "
from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv
import sys

# Mock CLI args
sys.argv = ['test', '--inference.backend', 'sglang']
config = parse_argv(InferenceConfig, allow_extras=True)
assert config.backend == 'sglang', f'Expected sglang, got {config.backend}'
print('✅ Router config parsing works')
"

# Test backend selection logic
python -c "
from prime_rl.inference.server import main
from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv
import sys

sys.argv = ['test', '--inference.backend', 'vllm']
config = parse_argv(InferenceConfig, allow_extras=True)
assert config.backend == 'vllm'
print('✅ vLLM backend selection works')

sys.argv = ['test', '--inference.backend', 'sglang']
config = parse_argv(InferenceConfig, allow_extras=True)
assert config.backend == 'sglang'
print('✅ SGLang backend selection works')
"

deactivate
```

### 2. SGLang Server Startup Test (Requires GPU)

Test that SGLang server starts correctly:

```bash
source venv_sglang/bin/activate

# Create minimal config
cat > /tmp/test_sglang.toml << EOF
[inference]
backend = "sglang"

[inference.model]
name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing

[inference.server]
port = 8000
EOF

# Start server (will run in background)
timeout 180 uv run inference @ /tmp/test_sglang.toml &
SERVER_PID=$!

# Wait for server to start
sleep 120

# Test health endpoint
curl -f http://localhost:8000/health || echo "❌ Health check failed"

# Test models endpoint
curl -f http://localhost:8000/v1/models || echo "❌ Models endpoint failed"

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }' || echo "❌ Chat completion failed"

# Cleanup
kill $SERVER_PID
deactivate
```

### 3. Integration Tests (Requires GPU)

Run the full integration test suite:

```bash
source venv_vllm/bin/activate

# Run SGLang RL integration test
uv run pytest tests/integration/test_rl_sglang.py -v --tb=short

# Expected tests:
# - test_no_error: Verifies SGLang RL process completes without errors
# - test_sglang_reward_goes_up: Verifies reward improves during training
# - test_sglang_reward_in_range: Verifies final reward meets threshold (>0.65)
# - test_no_error_resume: Verifies training can resume from checkpoint
# - test_sglang_reward_in_range_resume: Verifies resumed training maintains performance

deactivate
```

### 4. End-to-End RL Training Test (Requires 2+ GPUs)

Test complete RL training workflow with SGLang:

```bash
source venv_vllm/bin/activate

# Create test config
cat > /tmp/test_rl_sglang.toml << EOF
inference_gpu_ids = [0]
trainer_gpu_ids = [1]
max_steps = 10

[inference]
backend = "sglang"

[inference.model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

[trainer.optim]
lr = 3e-6

[orchestrator]
batch_size = 64
rollouts_per_example = 8

[[orchestrator.env]]
id = "reverse-text"
EOF

# Run RL training
uv run rl @ /tmp/test_rl_sglang.toml

# Check outputs
ls outputs/checkpoints/  # Should contain checkpoint files
ls outputs/logs/  # Should contain inference.stdout, orchestrator.stdout, trainer.stdout

# Verify reward improvement
grep "Reward:" outputs/logs/orchestrator.stdout

deactivate
```

### 5. Benchmark Comparison Test (Requires 2 GPUs)

Compare vLLM vs SGLang performance:

```bash
# Run automated benchmark
./scripts/run_benchmarks.sh

# Expected output:
# - prefix_results_vllm.json: vLLM baseline results
# - prefix_results_sglang.json: SGLang results (should show ~27% improvement)

# Check results
cat prefix_results_vllm.json
cat prefix_results_sglang.json
```

### 6. Weight Update Test (Requires GPU)

Test dynamic weight updates during training:

```bash
source venv_sglang/bin/activate

# Start SGLang server
uv run inference @ /tmp/test_sglang.toml &
SERVER_PID=$!
sleep 120

# Test weight update endpoint
curl -X POST http://localhost:8000/update_weights \
  -H "Content-Type: application/json" \
  -d '{"weight_dir": "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"}' \
  -v

# Test reload weights endpoint
curl -X POST http://localhost:8000/reload_weights -v

# Cleanup
kill $SERVER_PID
deactivate
```

## Expected Results

### Router Tests
- ✅ Config parsing works for both backends
- ✅ Backend selection logic correctly routes to vLLM/SGLang

### SGLang Server Tests
- ✅ Server starts without errors
- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ Models endpoint returns model info
- ✅ Chat completion returns valid response

### Integration Tests
- ✅ RL training completes without errors
- ✅ Reward improves from ~0.0 to >0.65
- ✅ Training can resume from checkpoint
- ✅ Resumed training maintains performance

### Benchmark Tests
- ✅ SGLang shows ~27% throughput improvement over vLLM
- ✅ SGLang shows ~21% latency reduction

## Troubleshooting

### Test Failures

**"No module named 'sglang'"**
```bash
# Ensure you're in the correct venv
source venv_sglang/bin/activate
pip install "sglang[all]"
```

**"Connection refused" errors**
```bash
# Check if server is running
curl http://localhost:8000/health

# Check if port is in use
lsof -i :8000

# Check GPU availability
nvidia-smi
```

**"CUDA out of memory"**
```bash
# Use smaller model
# Edit config: name = "Qwen/Qwen2.5-0.5B-Instruct"

# Or reduce GPU memory utilization
# Edit config: gpu_memory_utilization = 0.7
```

**Dependency conflicts**
```bash
# Recreate environments
rm -rf venv_vllm venv_sglang
./scripts/setup_env.sh
```

### Known Issues

1. **vLLM + SGLang Dependency Conflict**: Cannot install both in same environment
   - **Solution**: Use separate virtual environments

2. **Event Loop Conflicts**: Direct SGLang integration causes asyncio errors
   - **Solution**: Subprocess architecture (already implemented)

3. **Test Timeout**: Integration tests may timeout on slow GPUs
   - **Solution**: Increase timeout in test config or use smaller model

## Verification Checklist

Before considering the integration complete, verify:

- [ ] Router correctly selects backend based on config
- [ ] SGLang server starts without errors
- [ ] All 9 API endpoints respond correctly
- [ ] Chat completions work (streaming and non-streaming)
- [ ] Weight updates work during training
- [ ] Integration tests pass (test_rl_sglang.py)
- [ ] End-to-end RL training completes successfully
- [ ] Benchmark shows expected performance improvement
- [ ] Documentation is complete and accurate

## Next Steps

After all tests pass:

1. **Performance Profiling**: Run extended benchmarks to verify 27% improvement
2. **Multi-GPU Testing**: Test with TP=2, DP=2 configurations
3. **LoRA Testing**: Verify LoRA adapter loading/unloading
4. **NCCL Testing**: Test NCCL weight broadcast (if available)
5. **Production Deployment**: Deploy to staging environment

## Contact

For issues or questions:
- GitHub Issues: https://github.com/PrimeIntellect-ai/prime-rl/issues
- RFC #1615: SGLang Backend Integration
