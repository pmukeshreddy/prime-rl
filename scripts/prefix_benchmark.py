import asyncio, json, time, sys
from openai import AsyncOpenAI

# ~538 token equivalent system prompt as used in your winning run
SYSTEM_PROMPT = """You are an advanced AI research assistant specializing in reinforcement learning, machine learning optimization, and distributed systems. Your role is to help researchers and engineers solve complex technical problems.
## Your Capabilities
- Deep understanding of RL algorithms: PPO, GRPO, DPO, REINFORCE, A2C, SAC
- Expertise in distributed training: FSDP, DeepSpeed, Megatron-LM, tensor parallelism
- Knowledge of inference optimization: KV caching, speculative decoding, continuous batching
""" * 5 

async def benchmark(backend, model, num_requests=200, concurrency=32):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    latencies, ttfts, tokens = [], [], 0
    sem = asyncio.Semaphore(concurrency)

    async def run_req():
        nonlocal tokens
        async with sem:
            start = time.perf_counter()
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Explain RadixAttention benefits."}],
                    max_tokens=256, stream=True
                )
                ttft = None
                async for chunk in resp:
                    if ttft is None: ttft = time.perf_counter() - start
                    if chunk.choices[0].delta.content: tokens += 1
                latencies.append(time.perf_counter() - start)
                ttfts.append(ttft)
            except Exception as e:
                print(f"Request failed: {e}")

    start_t = time.perf_counter()
    await asyncio.gather(*(run_req() for _ in range(num_requests)))
    total_t = time.perf_counter() - start_t
    
    res = {
        "backend": backend,
        "output_tokens_per_sec": tokens / total_t,
        "avg_latency_ms": (sum(latencies)/len(latencies))*1000 if latencies else 0,
        "avg_ttft_ms": (sum(ttfts)/len(ttfts))*1000 if ttfts else 0
    }
    print(f"\n🚀 {backend.upper()} RESULTS:")
    print(json.dumps(res, indent=2))
    with open(f"prefix_results_{backend}.json", "w") as f: json.dump(res, f)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/prefix_benchmark.py [backend] [model]")
        sys.exit(1)
    asyncio.run(benchmark(sys.argv[1], sys.argv[2]))
