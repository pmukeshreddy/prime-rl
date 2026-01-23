"""
Inference server entrypoint that routes to vLLM or SGLang backends.

This module provides the main() entrypoint for the inference server command,
which routes to the appropriate backend (vLLM or SGLang) based on configuration.

Usage:
    # Use vLLM (default)
    uv run inference @ config.toml
    
    # Use SGLang for better throughput
    uv run inference @ config.toml --inference.backend sglang
    
    # Or set in config.toml
    [inference]
    backend = "sglang"
"""

import sys
from loguru import logger

from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv


def main():
    """
    Main entrypoint for the inference server.
    
    This function:
    1. Parses the inference configuration from CLI/config files
    2. Routes to the appropriate backend (vLLM or SGLang)
    3. Starts the inference server
    
    The backend is selected via the `backend` field in InferenceConfig:
    - "vllm" (default): Uses vLLM inference backend
    - "sglang": Uses SGLang inference backend with RadixAttention
    
    Any additional arguments not consumed by the config parser are passed
    directly to the backend server (e.g., vLLM-specific or SGLang-specific flags).
    """
    # Parse config from CLI args and config files, allowing extra args for backend
    config = parse_argv(InferenceConfig, allow_extras=True)
    remaining_args = config.get_unknown_args() if hasattr(config, 'get_unknown_args') else []
    
    logger.info(f"Starting inference server with backend: {config.backend}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Server: {config.server.host}:{config.server.port}")
    logger.info(f"Parallelism: {config.parallel}")
    
    # Route to the appropriate backend
    if config.backend == "vllm":
        logger.info("Routing to vLLM backend")
        from prime_rl.inference.vllm.server import server as vllm_server
        vllm_server(config, remaining_args)
        
    elif config.backend == "sglang":
        logger.info("Routing to SGLang backend")
        from prime_rl.inference.sglang.server import server as sglang_server
        sglang_server(config, remaining_args)
        
    else:
        logger.error(f"Unknown backend: {config.backend}")
        logger.error("Valid backends: vllm, sglang")
        sys.exit(1)


if __name__ == "__main__":
    main()
