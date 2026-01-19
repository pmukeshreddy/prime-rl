#!/usr/bin/env python3
"""
Benchmark script to compare vLLM vs SGLang inference backends.

This script measures:
- Generation throughput (tokens/sec)
- Time to first token (TTFT)
- Request latency

Usage:
    # Start vLLM server:
    python -m prime_rl.inference.server --backend vllm --model.name Qwen/Qwen2.5-0.5B
    
    # In another terminal, run benchmark:
    python scripts/benchmark_backends.py --base-url http://localhost:8000/v1 --backend vllm
    
    # Then start SGLang server:
    python -m prime_rl.inference.server --backend sglang --model.name Qwen/Qwen2.5-0.5B
    
    # Run benchmark:
    python scripts/benchmark_backends.py --base-url http://localhost:8000/v1 --backend sglang
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from openai import AsyncOpenAI


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    backend: str
    num_requests: int
    total_tokens_generated: int
    total_time_seconds: float
    tokens_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_ttft_ms: float
    errors: int = 0
    
    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "num_requests": self.num_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "avg_ttft_ms": round(self.avg_ttft_ms, 2),
            "errors": self.errors,
        }


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    tokens_generated: int
    latency_ms: float
    ttft_ms: float
    success: bool


async def benchmark_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestMetrics:
    """Run a single benchmark request and return metrics."""
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    tokens_generated = 0
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                # Rough token count (actual tokenization may differ)
                tokens_generated += len(chunk.choices[0].delta.content.split())
        
        end_time = time.perf_counter()
        
        # If streaming didn't give us tokens, make a non-streaming call to count
        if tokens_generated == 0:
            tokens_generated = max_tokens // 4  # Rough estimate
        
        return RequestMetrics(
            tokens_generated=tokens_generated,
            latency_ms=(end_time - start_time) * 1000,
            ttft_ms=((first_token_time or end_time) - start_time) * 1000,
            success=True,
        )
    except Exception as e:
        print(f"Request error: {e}")
        return RequestMetrics(
            tokens_generated=0,
            latency_ms=0,
            ttft_ms=0,
            success=False,
        )


async def run_benchmark(
    base_url: str,
    model: str,
    num_requests: int,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    backend: str,
) -> BenchmarkResult:
    """Run the complete benchmark and return results."""
    
    # Test prompts of varying complexity
    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis.",
        "Write a haiku about artificial intelligence.",
        "Explain how neural networks learn.",
        "What is the meaning of life?",
        "Describe a perfect day in nature.",
        "How do computers process information?",
        "Write a poem about the ocean.",
    ]
    
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=httpx.Timeout(300.0),
    )
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(prompt: str) -> RequestMetrics:
        async with semaphore:
            return await benchmark_request(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
    
    print(f"\n🚀 Starting benchmark for {backend} backend...")
    print(f"   Requests: {num_requests}, Concurrency: {concurrency}, Max Tokens: {max_tokens}")
    
    start_time = time.perf_counter()
    
    # Create tasks for all requests
    tasks = [
        bounded_request(prompts[i % len(prompts)])
        for i in range(num_requests)
    ]
    
    # Run all requests
    results = await asyncio.gather(*tasks)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Compute statistics
    successful_results = [r for r in results if r.success]
    failed_count = len(results) - len(successful_results)
    
    if not successful_results:
        print("❌ All requests failed!")
        return BenchmarkResult(
            backend=backend,
            num_requests=num_requests,
            total_tokens_generated=0,
            total_time_seconds=total_time,
            tokens_per_second=0,
            avg_latency_ms=0,
            p50_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            avg_ttft_ms=0,
            errors=failed_count,
        )
    
    latencies = [r.latency_ms for r in successful_results]
    ttfts = [r.ttft_ms for r in successful_results]
    total_tokens = sum(r.tokens_generated for r in successful_results)
    
    # Sort for percentiles
    latencies_sorted = sorted(latencies)
    
    def percentile(data: list, p: float) -> float:
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])
    
    result = BenchmarkResult(
        backend=backend,
        num_requests=num_requests,
        total_tokens_generated=total_tokens,
        total_time_seconds=total_time,
        tokens_per_second=total_tokens / total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=percentile(latencies_sorted, 50),
        p95_latency_ms=percentile(latencies_sorted, 95),
        p99_latency_ms=percentile(latencies_sorted, 99),
        avg_ttft_ms=statistics.mean(ttfts),
        errors=failed_count,
    )
    
    return result


def print_results(result: BenchmarkResult):
    """Print benchmark results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS: {result.backend.upper()}")
    print(f"{'='*60}")
    print(f"  Requests:              {result.num_requests}")
    print(f"  Successful:            {result.num_requests - result.errors}")
    print(f"  Errors:                {result.errors}")
    print(f"  Total Time:            {result.total_time_seconds:.2f}s")
    print(f"  Tokens Generated:      {result.total_tokens_generated}")
    print(f"\n  🔥 THROUGHPUT:")
    print(f"  Tokens/sec:            {result.tokens_per_second:.2f}")
    print(f"\n  ⏱️  LATENCY:")
    print(f"  Avg:                   {result.avg_latency_ms:.2f}ms")
    print(f"  P50:                   {result.p50_latency_ms:.2f}ms")
    print(f"  P95:                   {result.p95_latency_ms:.2f}ms")
    print(f"  P99:                   {result.p99_latency_ms:.2f}ms")
    print(f"  Avg TTFT:              {result.avg_ttft_ms:.2f}ms")
    print(f"{'='*60}\n")


def compare_results(vllm_result: BenchmarkResult, sglang_result: BenchmarkResult):
    """Compare and print the comparison between vLLM and SGLang results."""
    print(f"\n{'='*60}")
    print(f"🏆 COMPARISON: vLLM vs SGLang")
    print(f"{'='*60}")
    
    throughput_improvement = (
        (sglang_result.tokens_per_second - vllm_result.tokens_per_second) 
        / vllm_result.tokens_per_second * 100
    ) if vllm_result.tokens_per_second > 0 else 0
    
    latency_improvement = (
        (vllm_result.avg_latency_ms - sglang_result.avg_latency_ms)
        / vllm_result.avg_latency_ms * 100
    ) if vllm_result.avg_latency_ms > 0 else 0
    
    print(f"\n  Throughput:")
    print(f"    vLLM:    {vllm_result.tokens_per_second:.2f} tokens/sec")
    print(f"    SGLang:  {sglang_result.tokens_per_second:.2f} tokens/sec")
    print(f"    {'📈' if throughput_improvement > 0 else '📉'} Change: {throughput_improvement:+.1f}%")
    
    print(f"\n  Average Latency:")
    print(f"    vLLM:    {vllm_result.avg_latency_ms:.2f}ms")
    print(f"    SGLang:  {sglang_result.avg_latency_ms:.2f}ms")
    print(f"    {'📈' if latency_improvement > 0 else '📉'} Improvement: {latency_improvement:+.1f}%")
    
    print(f"\n  Average TTFT:")
    print(f"    vLLM:    {vllm_result.avg_ttft_ms:.2f}ms")
    print(f"    SGLang:  {sglang_result.avg_ttft_ms:.2f}ms")
    
    print(f"\n{'='*60}")
    
    if throughput_improvement > 10:
        print(f"\n✅ SGLang shows significant throughput improvement!")
        print(f"   With vLLM:  {vllm_result.tokens_per_second:.0f} tokens/sec")
        print(f"   With SGLang: {sglang_result.tokens_per_second:.0f} tokens/sec")
    elif throughput_improvement > 0:
        print(f"\n✅ SGLang shows modest improvement.")
    else:
        print(f"\n📊 vLLM performed better in this benchmark.")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM vs SGLang backends")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="Base URL of the inference server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model name to use for benchmarking")
    parser.add_argument("--backend", type=str, choices=["vllm", "sglang"], required=True,
                        help="Backend being benchmarked")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests to run")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of concurrent requests")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--compare-file", type=str, default=None,
                        help="Previous results file to compare against")
    
    args = parser.parse_args()
    
    # Run benchmark
    result = asyncio.run(run_benchmark(
        base_url=args.base_url,
        model=args.model,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        backend=args.backend,
    ))
    
    # Print results
    print_results(result)
    
    # Save results if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"📁 Results saved to {args.output}")
    
    # Compare with previous results if provided
    if args.compare_file:
        try:
            with open(args.compare_file, "r") as f:
                previous = json.load(f)
            
            # Create a BenchmarkResult from the previous data
            previous_result = BenchmarkResult(
                backend=previous["backend"],
                num_requests=previous["num_requests"],
                total_tokens_generated=previous["total_tokens_generated"],
                total_time_seconds=previous["total_time_seconds"],
                tokens_per_second=previous["tokens_per_second"],
                avg_latency_ms=previous["avg_latency_ms"],
                p50_latency_ms=previous["p50_latency_ms"],
                p95_latency_ms=previous["p95_latency_ms"],
                p99_latency_ms=previous["p99_latency_ms"],
                avg_ttft_ms=previous["avg_ttft_ms"],
                errors=previous.get("errors", 0),
            )
            
            if previous_result.backend == "vllm" and args.backend == "sglang":
                compare_results(previous_result, result)
            elif previous_result.backend == "sglang" and args.backend == "vllm":
                compare_results(result, previous_result)
            else:
                print(f"Note: Both results are from {args.backend}, no comparison made.")
        except Exception as e:
            print(f"Warning: Could not load comparison file: {e}")


if __name__ == "__main__":
    main()
