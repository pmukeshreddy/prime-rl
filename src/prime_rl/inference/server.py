from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)

    # Route to appropriate backend based on config
    if config.server.backend == "sglang":
        # Import and launch SGLang server
        from prime_rl.inference.sglang.server import server
        server(config, sglang_args=config.get_unknown_args())
    else:
        # Import and launch vLLM server (default)
        from prime_rl.inference.vllm.server import server  # pyright: ignore
        server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
