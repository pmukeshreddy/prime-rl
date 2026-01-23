#!/bin/bash
# =============================================================================
# prime-rl Backend Benchmark: vLLM vs SGLang (with streaming/TTFT)
# =============================================================================

set -e

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
REQUESTS="${REQUESTS:-200}"
PORT=8000

echo "=============================================="
echo "  prime-rl Backend Benchmark"
echo "  Model: $MODEL"
echo "  Requests: $REQUESTS per backend"
echo "=============================================="

# =============================================================================
# Nuclear Cleanup (same as run_benchmarks.sh)
# =============================================================================
echo -e "\n🧹 [1/4] Nuclear Cleanup..."
pkill -9 -f "prime_rl.inference" || true
pkill -9 -f vllm || true
pkill -9 -f sglang || true
pkill -9 -f python || true
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 || true
sleep 10

# =============================================================================
# vLLM Benchmark
# =============================================================================
echo -e "\n🚀 [2/4] prime-rl + vLLM..."
source venv_vllm/bin/activate

# Use both GPUs (TP=2) - NO prefix caching (matches original prime-rl)
# This gives SGLang advantage with RadixAttention
PYTHONPATH=src python3 -m prime_rl.inference.server \
    --backend vllm \
    --model.name "$MODEL" \
    --server.port $PORT \
    --model.trust-remote-code \
    --parallel.tp 2 \
    > /tmp/prime_rl_vllm.log 2>&1 &

PID=$!
echo "PID: $PID"
echo "Waiting for server..."

SERVER_READY=false
for i in {1..300}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ Server ready after ${i}s"
        SERVER_READY=true
        break
    fi
    if ! kill -0 $PID 2>/dev/null; then
        echo "❌ Server died!"; tail -50 /tmp/prime_rl_vllm.log; exit 1
    fi
    [ $((i % 30)) -eq 0 ] && echo "  Still waiting... ${i}s"
    sleep 1
done
[ "$SERVER_READY" = false ] && { echo "❌ Timeout"; exit 1; }

echo "Running benchmark ($REQUESTS requests, streaming)..."
python3 << BENCHMARK_EOF
import asyncio, json, time, sys
from openai import AsyncOpenAI

# IDENTICAL prompt for both vLLM and SGLang
SYSTEM_PROMPT = """You are an advanced AI research assistant specializing in reinforcement learning, machine learning optimization, and distributed systems.""" * 5
USER_MSG = "Explain RadixAttention and its benefits for LLM inference."
MAX_TOKENS = 256
TOTAL = $REQUESTS

async def bench():
    client = AsyncOpenAI(base_url='http://localhost:$PORT/v1', api_key='x')
    lats, ttfts, toks = [], [], 0
    sem = asyncio.Semaphore(32)
    completed = [0]
    start_time = time.perf_counter()
    
    def print_progress():
        pct = completed[0] * 100 // TOTAL
        bar = '█' * (pct // 2) + '░' * (50 - pct // 2)
        elapsed = time.perf_counter() - start_time
        rps = completed[0] / elapsed if elapsed > 0 else 0
        sys.stdout.write(f'\r  [{bar}] {completed[0]}/{TOTAL} ({pct}%) | {rps:.1f} req/s')
        sys.stdout.flush()
    
    async def req():
        nonlocal toks
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.chat.completions.create(
                    model='$MODEL',
                    messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':USER_MSG}],
                    max_tokens=MAX_TOKENS,
                    stream=True
                )
                ttft = None
                async for c in r:
                    if ttft is None: 
                        ttft = time.perf_counter() - t0
                    if c.choices[0].delta.content: 
                        toks += 1
                lats.append(time.perf_counter() - t0)
                if ttft: ttfts.append(ttft)
            except Exception as e:
                print(f"\nRequest error: {e}")
            finally:
                completed[0] += 1
                # Show progress every request for small batches, every 10 for large
                step = 1 if TOTAL <= 100 else 10
                if completed[0] % step == 0 or completed[0] == TOTAL:
                    print_progress()
    
    print_progress()  # Initial progress
    await asyncio.gather(*[req() for _ in range(TOTAL)])
    print()  # newline after progress bar
    
    res = {
        'backend': 'vllm',
        'total_tokens': toks,
        'output_tokens_per_sec': toks / sum(lats) * len(lats) if lats else 0,
        'avg_latency_ms': sum(lats) / len(lats) * 1000 if lats else 0,
        'avg_ttft_ms': sum(ttfts) / len(ttfts) * 1000 if ttfts else 0,
    }
    json.dump(res, open('prefix_results_vllm.json','w'))
    print(json.dumps(res, indent=2))

asyncio.run(bench())
BENCHMARK_EOF

kill $PID 2>/dev/null || true
deactivate

# =============================================================================
# Clear GPU (aggressive cleanup like run_benchmarks.sh)
# =============================================================================
echo -e "\n☢️  [3/4] Clearing VRAM..."
pkill -9 -f vllm || true
pkill -9 -f python || true
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 || true
sleep 20

# =============================================================================
# SGLang Benchmark
# =============================================================================
echo -e "\n🚀 [4/4] prime-rl + SGLang..."
source venv_sglang/bin/activate

# SGLang internal port (prime-rl server runs on PORT, SGLang runs on PORT+1000)
SGLANG_PORT=$((PORT + 1000))

# Use prime-rl SGLang wrapper with TP=2 (same as vLLM for fair comparison)
# Keep it simple - match native SGLang launch exactly
PYTHONPATH=src python3 -m prime_rl.inference.server \
    --backend sglang \
    --model.name "$MODEL" \
    --server.port $PORT \
    --model.trust-remote-code \
    --parallel.tp 2 \
    > /tmp/prime_rl_sglang.log 2>&1 &

PID=$!
echo "PID: $PID"
echo "Waiting for server..."

SERVER_READY=false
for i in {1..300}; do
    # Wait for SGLang's internal server to be ready (port 9000)
    if curl -s http://localhost:$SGLANG_PORT/health > /dev/null 2>&1; then
        echo "✅ Server ready after ${i}s"
        SERVER_READY=true
        break
    fi
    if ! kill -0 $PID 2>/dev/null; then
        echo "❌ Server died!"; tail -50 /tmp/prime_rl_sglang.log; exit 1
    fi
    [ $((i % 30)) -eq 0 ] && echo "  Still waiting... ${i}s"
    sleep 1
done
[ "$SERVER_READY" = false ] && { echo "❌ Timeout"; exit 1; }

echo "Running benchmark ($REQUESTS requests, streaming)..."
echo "Hitting SGLang directly on port $SGLANG_PORT (no proxy overhead)"
python3 << BENCHMARK_EOF
import asyncio, json, time, sys
from openai import AsyncOpenAI

# IDENTICAL prompt for both vLLM and SGLang (same as vLLM benchmark above)
SYSTEM_PROMPT = """You are an advanced AI research assistant specializing in reinforcement learning, machine learning optimization, and distributed systems.""" * 5
USER_MSG = "Explain RadixAttention and its benefits for LLM inference."
MAX_TOKENS = 256
TOTAL = $REQUESTS

async def bench():
    client = AsyncOpenAI(base_url='http://localhost:$SGLANG_PORT/v1', api_key='x')
    lats, ttfts, toks = [], [], 0
    sem = asyncio.Semaphore(32)
    completed = [0]
    start_time = time.perf_counter()
    
    def print_progress():
        pct = completed[0] * 100 // TOTAL
        bar = '█' * (pct // 2) + '░' * (50 - pct // 2)
        elapsed = time.perf_counter() - start_time
        rps = completed[0] / elapsed if elapsed > 0 else 0
        sys.stdout.write(f'\r  [{bar}] {completed[0]}/{TOTAL} ({pct}%) | {rps:.1f} req/s')
        sys.stdout.flush()
    
    async def req():
        nonlocal toks
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.chat.completions.create(
                    model='$MODEL',
                    messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':USER_MSG}],
                    max_tokens=MAX_TOKENS,
                    stream=True
                )
                ttft = None
                async for c in r:
                    if ttft is None: 
                        ttft = time.perf_counter() - t0
                    if c.choices[0].delta.content: 
                        toks += 1
                lats.append(time.perf_counter() - t0)
                if ttft: ttfts.append(ttft)
            except Exception as e:
                print(f"\nRequest error: {e}")
            finally:
                completed[0] += 1
                step = 1 if TOTAL <= 100 else 10
                if completed[0] % step == 0 or completed[0] == TOTAL:
                    print_progress()
    
    print_progress()
    # Simple asyncio.gather like native benchmark - no batching
    await asyncio.gather(*(req() for _ in range(TOTAL)))
    print()

    res = {
        'backend': 'sglang',
        'total_tokens': toks,
        'output_tokens_per_sec': toks / sum(lats) * len(lats) if lats else 0,
        'avg_latency_ms': sum(lats) / len(lats) * 1000 if lats else 0,
        'avg_ttft_ms': sum(ttfts) / len(ttfts) * 1000 if ttfts else 0,
    }
    json.dump(res, open('prefix_results_sglang.json','w'))
    print(json.dumps(res, indent=2))

asyncio.run(bench())
BENCHMARK_EOF

kill $PID 2>/dev/null || true
deactivate

# =============================================================================
# Results
# =============================================================================
echo -e "\n=============================================="
echo "  RESULTS"
echo "=============================================="
echo "vLLM:"
cat prefix_results_vllm.json
echo -e "\n"
echo "SGLang:"
cat prefix_results_sglang.json
echo -e "\n"

python3 -c "
import json
v = json.load(open('prefix_results_vllm.json'))
s = json.load(open('prefix_results_sglang.json'))
print('📈 SGLang vs vLLM (prime-rl integration):')
print(f'   Throughput: +{((s[\"output_tokens_per_sec\"]/v[\"output_tokens_per_sec\"])-1)*100:.1f}%')
print(f'   Latency:    -{((v[\"avg_latency_ms\"]-s[\"avg_latency_ms\"])/v[\"avg_latency_ms\"])*100:.1f}%')
print(f'   TTFT:       -{((v[\"avg_ttft_ms\"]-s[\"avg_ttft_ms\"])/v[\"avg_ttft_ms\"])*100:.1f}%')
"
echo "=============================================="
