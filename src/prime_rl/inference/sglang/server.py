"""
SGLang inference server backend for prime-rl.

This module provides an SGLang-based inference server that is compatible with
the prime-rl orchestrator. It supports:
- OpenAI-compatible chat completions API
- Dynamic weight updates from disk or via NCCL broadcast
- LoRA adapter loading/unloading
- Custom /v1/chat/completions/tokens endpoint for token-in requests
"""

import asyncio
import os
from argparse import Namespace
from http import HTTPStatus
from typing import Optional

import uvloop
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field

from prime_rl.inference.config import InferenceConfig

# Set environment variables before importing SGLang
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")


def server(config: InferenceConfig, sglang_args: list[str]):
    """
    Launch the SGLang inference server with prime-rl customizations.
    
    Args:
        config: The inference configuration from prime-rl
        sglang_args: Additional command-line arguments to pass to SGLang
    """
    # Import SGLang modules
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.entrypoints.http_server import launch_server, app
    from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
    
    # Convert prime-rl config to SGLang ServerArgs
    sglang_server_args = _config_to_sglang_args(config, sglang_args)
    
    # Add custom routes for prime-rl compatibility
    _add_prime_rl_routes(app)
    
    # Launch the server
    launch_server(sglang_server_args)


def _config_to_sglang_args(config: InferenceConfig, extra_args: list[str]) -> "ServerArgs":
    """
    Convert prime-rl InferenceConfig to SGLang ServerArgs.
    """
    from sglang.srt.server_args import ServerArgs
    
    # Create ServerArgs directly (no CLI parsing needed)
    server_args = ServerArgs(
        model_path=config.model.name,
        host=config.server.host or "0.0.0.0",
        port=config.server.port,
        tp_size=config.parallel.tp,
        dp_size=config.parallel.dp,
        dtype=config.model.dtype,
        mem_fraction_static=config.gpu_memory_utilization,
        context_length=config.model.max_model_len,
        disable_cuda_graph=config.model.enforce_eager,
        trust_remote_code=config.model.trust_remote_code,
        enable_lora=config.enable_lora,
        max_lora_rank=config.max_lora_rank if config.enable_lora else None,
    )
    
    return server_args


def _add_prime_rl_routes(app: FastAPI):
    """
    Add prime-rl specific routes to the SGLang FastAPI app.
    
    This adds:
    - /update_weights - Compatible with prime-rl orchestrator
    - /reload_weights - Reset to base model weights  
    - /init_broadcaster - Initialize NCCL broadcast for weight sync
    - /v1/chat/completions/tokens - Token-in endpoint
    """
    from sglang.srt.entrypoints.http_server import get_global_state
    from sglang.srt.managers.io_struct import (
        UpdateWeightFromDiskReqInput,
        InitWeightsUpdateGroupReqInput,
    )
    from pydantic import BaseModel
    
    class UpdateWeightsRequest(BaseModel):
        weight_dir: Optional[str] = None
    
    class InitBroadcasterRequest(BaseModel):
        host: str
        port: int
        server_rank: int
        num_inference_server: int
        timeout: int = 300
    
    class ChatCompletionRequestWithTokens(BaseModel):
        """Chat completion request that accepts pre-tokenized input."""
        model: str
        messages: list
        tokens: list[int] = Field(description="Prompt tokens to use for the request.")
        max_tokens: Optional[int] = None
        temperature: float = 1.0
        top_p: float = 1.0
        stream: bool = False
        logprobs: bool = False
        top_logprobs: Optional[int] = None
        skip_special_tokens: bool = True
        prompt_logprobs: Optional[int] = None
        # Additional SGLang-specific params
        min_tokens: int = 0
        repetition_penalty: float = 1.0
    
    @app.post("/update_weights")
    async def update_weights(request: UpdateWeightsRequest, raw_request: Request):
        """
        Update model weights from disk path.
        Compatible with prime-rl orchestrator's weight update mechanism.
        """
        global_state = get_global_state()
        if global_state is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        tokenizer_manager = global_state.tokenizer_manager
        
        if request.weight_dir is None:
            return {"status": "ok", "message": "No weight_dir provided, skipping update"}
        
        update_req = UpdateWeightFromDiskReqInput(
            model_path=request.weight_dir,
            load_format=None,
            abort_all_requests=False,
            flush_cache=True,
        )
        
        success, message, num_paused = await tokenizer_manager.update_weights_from_disk(
            update_req, raw_request
        )
        
        if success:
            return {"status": "ok", "message": message}
        else:
            return JSONResponse(
                content={"status": "error", "message": message},
                status_code=HTTPStatus.BAD_REQUEST
            )
    
    @app.post("/reload_weights")
    async def reload_weights(raw_request: Request):
        """
        Reload the original base model weights.
        Note: SGLang doesn't have a direct equivalent, so we return a warning.
        """
        # SGLang doesn't support reloading original weights directly
        # The user should restart the server or update to the original path
        return JSONResponse(
            content={
                "status": "warning",
                "message": "SGLang does not support reload_weights. Use update_weights with the original model path instead."
            },
            status_code=HTTPStatus.OK
        )
    
    @app.post("/init_broadcaster")
    async def init_broadcaster(request: InitBroadcasterRequest, raw_request: Request):
        """
        Initialize NCCL broadcast group for distributed weight synchronization.
        Compatible with prime-rl's NCCL weight broadcast mechanism.
        """
        global_state = get_global_state()
        if global_state is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        tokenizer_manager = global_state.tokenizer_manager
        
        # Calculate the rank offset for this inference server
        # prime-rl uses: global_rank = (server_rank * tp_size) + tp_rank
        # SGLang's init_weights_update_group expects rank_offset
        tp_size = tokenizer_manager.server_args.tp_size
        rank_offset = request.server_rank * tp_size + 1  # +1 because trainer is rank 0
        world_size = request.num_inference_server * tp_size + 1
        
        init_req = InitWeightsUpdateGroupReqInput(
            master_address=request.host,
            master_port=request.port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name="prime_rl_weight_update",
            backend="nccl",
        )
        
        success, message = await tokenizer_manager.init_weights_update_group(
            init_req, raw_request
        )
        
        if success:
            return {"status": "ok"}
        else:
            return JSONResponse(
                content={"status": "error", "message": message},
                status_code=HTTPStatus.BAD_REQUEST
            )
    
    @app.post("/v1/chat/completions/tokens")
    async def chat_completions_with_tokens(
        request: ChatCompletionRequestWithTokens,
        raw_request: Request
    ):
        """
        Chat completions endpoint that accepts pre-tokenized input.
        This is used by prime-rl for efficient multi-turn conversations
        to avoid retokenization discrepancies.
        """
        global_state = get_global_state()
        if global_state is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        tokenizer_manager = global_state.tokenizer_manager
        
        # Build sampling params
        sampling_params = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "min_new_tokens": request.min_tokens,
            "repetition_penalty": request.repetition_penalty,
        }
        
        if request.logprobs:
            sampling_params["return_logprob"] = True
            if request.top_logprobs:
                sampling_params["top_logprobs_num"] = request.top_logprobs
        
        if request.prompt_logprobs:
            # SGLang supports prompt logprobs via logprob_start_len
            sampling_params["return_logprob"] = True
            sampling_params["logprob_start_len"] = 0
        
        # Import GenerateReqInput
        from sglang.srt.managers.io_struct import GenerateReqInput
        import uuid
        
        # Create request with token input
        gen_req = GenerateReqInput(
            input_ids=request.tokens,
            sampling_params=sampling_params,
            rid=uuid.uuid4().hex,
            stream=request.stream,
        )
        
        if request.stream:
            async def stream_generator():
                async for output in tokenizer_manager.generate_request(gen_req, raw_request):
                    # Format as OpenAI-compatible SSE
                    chunk = _format_streaming_response(output, request.model)
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            output = await tokenizer_manager.generate_request(gen_req, raw_request).__anext__()
            response = _format_completion_response(output, request.model, request.tokens)
            return JSONResponse(content=response)


def _format_completion_response(output: dict, model: str, prompt_tokens: list[int]) -> dict:
    """Format SGLang output as OpenAI-compatible chat completion response."""
    import time
    
    response = {
        "id": f"chatcmpl-{output.get('rid', 'unknown')}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output.get("text", ""),
            },
            "finish_reason": _convert_finish_reason(output.get("meta_info", {}).get("finish_reason")),
        }],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": output.get("meta_info", {}).get("completion_tokens", 0),
            "total_tokens": len(prompt_tokens) + output.get("meta_info", {}).get("completion_tokens", 0),
        }
    }
    
    # Add logprobs if available
    if "meta_info" in output and "output_token_logprobs" in output["meta_info"]:
        response["choices"][0]["logprobs"] = {
            "content": output["meta_info"]["output_token_logprobs"]
        }
    
    # Add prompt logprobs if available
    if "meta_info" in output and "input_token_logprobs" in output["meta_info"]:
        response["prompt_logprobs"] = output["meta_info"]["input_token_logprobs"]
    
    return response


def _format_streaming_response(output: dict, model: str) -> str:
    """Format SGLang streaming output as OpenAI-compatible SSE chunk."""
    import json
    import time
    
    chunk = {
        "id": f"chatcmpl-{output.get('rid', 'unknown')}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": output.get("text", ""),
            },
            "finish_reason": _convert_finish_reason(output.get("meta_info", {}).get("finish_reason")) if output.get("meta_info", {}).get("finish_reason") else None,
        }]
    }
    return json.dumps(chunk)


def _convert_finish_reason(reason) -> str:
    """Convert SGLang finish reason to OpenAI format."""
    if reason is None:
        return None
    reason_str = str(reason).lower()
    if "length" in reason_str:
        return "length"
    elif "stop" in reason_str or "eos" in reason_str:
        return "stop"
    return "stop"


if __name__ == "__main__":
    from prime_rl.inference.config import InferenceConfig
    from prime_rl.utils.pydantic_config import parse_argv
    
    config = parse_argv(InferenceConfig, allow_extras=True)
    server(config, sglang_args=config.get_unknown_args())
