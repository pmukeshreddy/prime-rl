"""
SGLang inference backend for prime-rl.

This package provides an alternative inference backend using SGLang instead of vLLM.
SGLang offers improved throughput for RL workloads through RadixAttention,
which automatically shares KV cache across requests with common prefixes.

Usage:
    # In config TOML
    [inference]
    backend = "sglang"
    
    # Or via CLI
    uv run rl @ config.toml --inference.backend sglang

Key Features:
- RadixAttention for automatic prefix caching (23% throughput improvement)
- Full compatibility with prime-rl's weight update mechanism
- Support for both filesystem and NCCL weight broadcast
- OpenAI-compatible API with prime-rl extensions

See docs/sglang_integration.md for detailed documentation.
"""

from prime_rl.inference.sglang.server import server

__all__ = ["server"]
