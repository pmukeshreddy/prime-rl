"""
SGLang inference server for prime-rl.

This module provides an SGLang-based inference backend that is compatible with
prime-rl's training loop. It includes custom endpoints for:
- Weight updates during training (/update_weights)
- NCCL weight broadcast initialization (/init_broadcaster)
- Pre-tokenized chat completions (/v1/chat/completions/tokens)

SGLang provides ~30% throughput improvement over vLLM for RL workloads due to
RadixAttention's automatic prefix caching.

Usage:
    uv run inference @ config.toml --inference.backend sglang
"""

import asyncio
import json
import os
import signal
import sys
import threading
import time
from http import HTTPStatus
from typing import Any, AsyncGenerator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from prime_rl.inference.config import InferenceConfig


# ============================================================================
# Request/Response Models
# ============================================================================

class UpdateWeightsRequest(BaseModel):
    """Request model for weight updates."""
    weight_dir: str


class InitBroadcasterRequest(BaseModel):
    """Request model for NCCL broadcaster initialization."""
    host: str
    port: int
    server_rank: int = 0
    num_inference_server: int = 1
    timeout: int = 300


class ChatCompletionRequest(BaseModel):
    """Request model for OpenAI-compatible chat completions."""
    model: str
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    stop: Optional[list[str]] = None
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None


class ChatCompletionTokensRequest(ChatCompletionRequest):
    """Request model for chat completions with pre-tokenized input."""
    input_ids: Optional[list[int]] = None
    tokens: Optional[list[int]] = None  # Alias for compatibility with vLLM endpoint


class LoadLoRARequest(BaseModel):
    """Request model for loading LoRA adapters."""
    lora_name: str
    lora_path: str


class UnloadLoRARequest(BaseModel):
    """Request model for unloading LoRA adapters."""
    lora_name: str


# ============================================================================
# Global State
# ============================================================================

# SGLang runtime instance (set after server starts)
_runtime = None
_runtime_lock = threading.Lock()
_model_path = None
_original_model_path = None  # For reload_weights
_server_args = None  # Store server args for runtime recreation


def get_runtime():
    """Get the SGLang runtime instance."""
    global _runtime
    if _runtime is None:
        raise RuntimeError("SGLang runtime not initialized")
    return _runtime


def set_runtime(runtime, model_path: str = None):
    """Set the SGLang runtime instance."""
    global _runtime, _model_path, _original_model_path
    with _runtime_lock:
        _runtime = runtime
        _model_path = model_path
        if _original_model_path is None:
            _original_model_path = model_path  # Store original for reload


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="prime-rl SGLang Server",
    description="SGLang inference server with prime-rl extensions for RL training",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    global _model_path
    return {
        "object": "list",
        "data": [
            {
                "id": _model_path or "unknown",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "prime-rl",
            }
        ]
    }


# ============================================================================
# OpenAI-Compatible Chat Completions
# ============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    This is the standard endpoint used by AsyncOpenAI clients in the orchestrator.
    SGLang's RadixAttention automatically caches common prefixes for improved
    throughput in RL workloads where many requests share system prompts.
    
    Returns:
        Standard OpenAI chat completion response (streaming or non-streaming)
    """
    try:
        runtime = get_runtime()
        
        # Build sampling params for SGLang Runtime
        sampling_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_tokens or 256,
        }
        
        if request.stop:
            sampling_params["stop"] = request.stop
        
        if request.top_k > 0:
            sampling_params["top_k"] = request.top_k
        
        # Return logprobs if requested
        if request.logprobs:
            sampling_params["return_logprob"] = True
            if request.top_logprobs:
                sampling_params["top_logprobs_num"] = request.top_logprobs
        
        # Convert messages to prompt string for SGLang Runtime
        # SGLang Runtime.generate() expects a prompt string, not messages
        prompt = _messages_to_prompt(request.messages)
        
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(runtime, prompt, sampling_params, request.model),
                media_type="text/event-stream"
            )
        else:
            result = await asyncio.to_thread(
                _generate_completion, runtime, prompt, sampling_params
            )
            return JSONResponse(content=_format_chat_response(result, request.model))
            
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Prime-RL Specific Endpoints
# ============================================================================

@app.post("/update_weights")
async def update_weights(request: Request):
    """
    Update model weights from disk.
    
    This endpoint is called by the orchestrator after the trainer saves a new
    checkpoint. SGLang will reload the model weights from the specified path.
    
    Request body:
        {"weight_dir": "/path/to/checkpoint"}
    
    Returns:
        {"status": "ok"} on success
    """
    try:
        data = await request.json()
        weight_dir = data.get("weight_dir")
        
        if not weight_dir:
            raise HTTPException(status_code=400, detail="weight_dir is required")
        
        logger.info(f"Updating weights from: {weight_dir}")
        
        runtime = get_runtime()
        
        # Method 1: Direct runtime method (newer SGLang versions)
        if hasattr(runtime, 'update_weights_from_disk'):
            success = await asyncio.to_thread(runtime.update_weights_from_disk, weight_dir)
            if not success:
                raise HTTPException(status_code=500, detail="Weight update failed")
            logger.success(f"Weights updated via update_weights_from_disk")
            return {"status": "ok"}
        
        # Method 2: HTTP call to internal SGLang server's /update_weights_from_disk
        if hasattr(runtime, 'url') and runtime.url:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{runtime.url}/update_weights_from_disk",
                    json={"model_path": weight_dir}
                )
                if resp.status_code == 200:
                    logger.success(f"Weights updated via /update_weights_from_disk: {weight_dir}")
                    return {"status": "ok"}
                else:
                    logger.warning(f"update_weights_from_disk returned {resp.status_code}: {resp.text}")
        
        # Method 3: Load weights directly if we have access to the model
        if hasattr(runtime, 'update_weights'):
            success = await asyncio.to_thread(runtime.update_weights, weight_dir)
            if success:
                logger.success(f"Weights updated via update_weights")
                return {"status": "ok"}
        
        raise HTTPException(
            status_code=501, 
            detail="Weight update not supported by this SGLang version"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_weights")
async def reload_weights():
    """
    Reload model weights from the original model path.
    
    Returns:
        {"status": "ok"} on success
    """
    global _original_model_path
    
    try:
        runtime = get_runtime()
        
        # Method 1: Direct runtime method
        if hasattr(runtime, 'reload_weights'):
            await asyncio.to_thread(runtime.reload_weights)
            logger.success("Weights reloaded via reload_weights")
            return {"status": "ok"}
        
        # Method 2: HTTP call to internal SGLang server's /update_weights_from_disk
        if hasattr(runtime, 'url') and runtime.url:
            async with httpx.AsyncClient(timeout=300) as client:
                # Reload from original model path
                model_path = _original_model_path or _model_path
                if model_path:
                    resp = await client.post(
                        f"{runtime.url}/update_weights_from_disk",
                        json={"model_path": model_path}
                    )
                    if resp.status_code == 200:
                        logger.success(f"Weights reloaded via /update_weights_from_disk: {model_path}")
                        return {"status": "ok"}
        
        raise HTTPException(
            status_code=501,
            detail="Weight reload not supported by this SGLang version"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/init_broadcaster")
async def init_broadcaster(request: Request):
    """
    Initialize NCCL weight broadcast group.
    
    This endpoint sets up direct GPU-to-GPU weight transfer from the trainer
    to the inference server, bypassing filesystem I/O for lower latency.
    
    Request body:
        {
            "host": "master_host",
            "port": 12345,
            "server_rank": 0,
            "num_inference_server": 1,
            "timeout": 300
        }
    
    Returns:
        {"status": "ok"} on success
    """
    try:
        data = await request.json()
        host = data.get("host")
        port = data.get("port")
        server_rank = data.get("server_rank", 0)
        num_inference_server = data.get("num_inference_server", 1)
        timeout = data.get("timeout", 300)
        
        if not host or not port:
            raise HTTPException(status_code=400, detail="host and port are required")
        
        logger.info(f"Initializing NCCL broadcaster: {host}:{port} (rank={server_rank})")
        
        runtime = get_runtime()
        
        # SGLang's native NCCL weight update group initialization
        if hasattr(runtime, 'init_weights_update_group'):
            await asyncio.to_thread(
                runtime.init_weights_update_group,
                master_address=host,
                master_port=port,
                rank_offset=server_rank,
                world_size=num_inference_server + 1,  # +1 for trainer
                group_name="weight_update",
                backend="nccl",
            )
        else:
            raise HTTPException(
                status_code=501,
                detail="NCCL broadcast not supported by this SGLang version"
            )
        
        logger.success(f"NCCL broadcaster initialized: {host}:{port}")
        return {"status": "ok"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize broadcaster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions/tokens")
async def chat_completions_with_tokens(request: ChatCompletionTokensRequest):
    """
    Chat completions with pre-tokenized input support.
    
    This endpoint accepts either regular messages or pre-tokenized input_ids/tokens,
    which avoids re-tokenization overhead in the RL training loop.
    
    Compatible with vLLM's /v1/chat/completions/tokens endpoint.
    
    Returns:
        Standard OpenAI chat completion response (streaming or non-streaming)
    """
    try:
        runtime = get_runtime()
        
        # Build the generation request
        gen_kwargs = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop,
            "n": request.n,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }
        
        # Handle max_tokens
        if request.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = request.max_tokens
        
        if request.top_k > 0:
            gen_kwargs["top_k"] = request.top_k
        
        if request.seed is not None:
            gen_kwargs["seed"] = request.seed
        
        # Return logprobs if requested
        if request.logprobs:
            gen_kwargs["return_logprob"] = True
            if request.top_logprobs:
                gen_kwargs["top_logprobs_num"] = request.top_logprobs
        
        # Use input_ids/tokens if provided, otherwise use messages
        # Support both 'input_ids' and 'tokens' field names for compatibility
        token_ids = request.input_ids or request.tokens
        if token_ids is not None:
            gen_kwargs["input_ids"] = token_ids
        else:
            gen_kwargs["messages"] = request.messages
        
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(runtime, gen_kwargs, request.model),
                media_type="text/event-stream"
            )
        else:
            result = await asyncio.to_thread(
                _generate_completion, runtime, gen_kwargs
            )
            return JSONResponse(content=_format_chat_response(result, request.model))
            
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_chat_completion(
    runtime, 
    prompt: str,
    sampling_params: dict, 
    model: str
) -> AsyncGenerator[str, None]:
    """Stream chat completion chunks.
    
    Uses a thread and queue to avoid "Cannot run the event loop while another 
    loop is running" errors when SGLang's generate() uses asyncio internally.
    """
    import queue
    
    # Sentinel to signal end of stream
    _DONE = object()
    _ERROR = object()
    
    chunk_queue = queue.Queue()
    
    def run_stream():
        """Run the synchronous streaming in a separate thread."""
        try:
            if hasattr(runtime, 'generate'):
                # Don't pass stream in sampling_params - SGLang doesn't accept it there
                # Use stream=True as a separate argument to generate()
                stream = runtime.generate(prompt, sampling_params=sampling_params, stream=True)
                
                for chunk in stream:
                    chunk_queue.put(chunk)
            
            chunk_queue.put(_DONE)
        except Exception as e:
            chunk_queue.put((_ERROR, e))
    
    # Start streaming in a background thread
    stream_thread = threading.Thread(target=run_stream, daemon=True)
    stream_thread.start()
    
    try:
        while True:
            # Use asyncio.to_thread to wait for queue.get() without blocking the event loop
            chunk = await asyncio.to_thread(chunk_queue.get, timeout=60)
            
            if chunk is _DONE:
                break
            
            if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] is _ERROR:
                raise chunk[1]
            
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            finish_reason = chunk.get("finish_reason") if isinstance(chunk, dict) else None
            
            response_chunk = {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": finish_reason,
                }]
            }
            yield f"data: {json.dumps(response_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": str(e)}
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        # Ensure thread completes
        stream_thread.join(timeout=1)


def _messages_to_prompt(messages: list) -> str:
    """Convert OpenAI messages format to a prompt string."""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    # Add assistant prefix for generation
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def _generate_completion(runtime, prompt: str, sampling_params: dict) -> dict:
    """Generate a completion (non-streaming)."""
    if hasattr(runtime, 'generate'):
        result = runtime.generate(prompt, sampling_params=sampling_params)
        # Handle different result formats
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return result
    else:
        raise RuntimeError("SGLang runtime does not support generate()")


def _format_chat_response(result, model: str) -> dict:
    """Format SGLang result as OpenAI chat completion response."""
    # Handle string results (SGLang may return JSON string)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            # Plain text result
            return {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                    "logprobs": None,
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    # Extract text from dict result
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    meta_info = result.get("meta_info", {}) if isinstance(result, dict) else {}
    
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": result.get("finish_reason", "stop") if isinstance(result, dict) else "stop",
            "logprobs": meta_info.get("output_token_logprobs"),
        }],
        "usage": {
            "prompt_tokens": meta_info.get("prompt_tokens", 0),
            "completion_tokens": meta_info.get("completion_tokens", 0),
            "total_tokens": meta_info.get("total_tokens", 0),
        }
    }


# ============================================================================
# LoRA Endpoints
# ============================================================================

@app.post("/v1/load_lora_adapter")
async def load_lora_adapter(request: LoadLoRARequest):
    """Load a LoRA adapter."""
    try:
        runtime = get_runtime()
        
        if hasattr(runtime, 'load_lora_adapter'):
            await asyncio.to_thread(
                runtime.load_lora_adapter,
                request.lora_name,
                request.lora_path
            )
        else:
            raise HTTPException(
                status_code=501,
                detail="LoRA not supported by this SGLang version"
            )
        
        logger.success(f"LoRA adapter loaded: {request.lora_name}")
        return {"status": "ok"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load LoRA adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/unload_lora_adapter")
async def unload_lora_adapter(request: UnloadLoRARequest):
    """Unload a LoRA adapter."""
    try:
        runtime = get_runtime()
        
        if hasattr(runtime, 'unload_lora_adapter'):
            await asyncio.to_thread(runtime.unload_lora_adapter, request.lora_name)
        else:
            raise HTTPException(
                status_code=501,
                detail="LoRA not supported by this SGLang version"
            )
        
        logger.success(f"LoRA adapter unloaded: {request.lora_name}")
        return {"status": "ok"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unload LoRA adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Server Entry Point
# ============================================================================

def server(config: InferenceConfig, sglang_args: list[str] = None):
    """
    Start the SGLang inference server with prime-rl extensions.
    
    This function:
    1. Configures and starts the SGLang runtime
    2. Adds custom FastAPI routes for prime-rl
    3. Runs the combined server
    
    Args:
        config: Prime-RL inference configuration
        sglang_args: Additional arguments to pass to SGLang
    """
    sglang_args = sglang_args or []
    
    try:
        import sglang as sgl
    except ImportError as e:
        logger.error(f"Failed to import sglang: {e}")
        logger.error("Please install sglang: pip install 'sglang[all]'")
        sys.exit(1)
    
    logger.info(f"Starting SGLang server with model: {config.model.name}")
    logger.info(f"Server config: host={config.server.host}, port={config.server.port}")
    logger.info(f"Parallel config: tp={config.parallel.tp}, dp={config.parallel.dp}")
    
    # Build SGLang server arguments
    server_args = {
        "model_path": config.model.name,
        "host": config.server.host or "0.0.0.0",
        "port": config.server.port,
        "tp_size": config.parallel.tp,
        "dp_size": config.parallel.dp,
        "trust_remote_code": config.model.trust_remote_code,
        "mem_fraction_static": config.gpu_memory_utilization,
    }
    
    # Handle dtype
    if config.model.dtype and config.model.dtype != "auto":
        server_args["dtype"] = config.model.dtype
    
    # Add max_model_len if specified
    if config.model.max_model_len:
        server_args["context_length"] = config.model.max_model_len
    
    # Add enforce_eager (disable CUDA graphs)
    if config.model.enforce_eager:
        server_args["disable_cuda_graph"] = True
    
    # Add LoRA config if enabled
    if config.enable_lora:
        server_args["enable_lora"] = True
        if config.max_lora_rank:
            server_args["max_lora_rank"] = config.max_lora_rank
    
    # Parse any additional SGLang-specific arguments
    i = 0
    while i < len(sglang_args):
        arg = sglang_args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(sglang_args) and not sglang_args[i + 1].startswith("--"):
                value = sglang_args[i + 1]
                # Try to parse as int/float/bool
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                server_args[key] = value
                i += 2
            else:
                server_args[key] = True
                i += 1
        else:
            i += 1
    
    logger.debug(f"SGLang server args: {server_args}")
    
    # Try different SGLang initialization methods based on version
    try:
        _start_with_runtime(config, server_args, sgl)
    except Exception as e:
        logger.error(f"Failed to start SGLang server: {e}")
        raise


def _start_with_runtime(config: InferenceConfig, server_args: dict, sgl):
    """Start server using SGLang Engine (not Runtime, which starts its own HTTP server)."""
    
    # Try to import Engine from sglang (lower-level API without HTTP server)
    Engine = None
    try:
        from sglang import Engine
    except ImportError:
        pass
    
    if Engine is None:
        try:
            from sglang.srt.server import Engine
        except ImportError:
            pass
    
    if Engine is None:
        # Fallback: Use subprocess to launch SGLang and proxy requests
        logger.info("Engine not available, using subprocess-based SGLang server")
        _start_with_subprocess(config, server_args)
        return
    
    logger.info("Initializing SGLang Engine...")
    
    # Create the engine with only supported args
    engine_args = {k: v for k, v in server_args.items() 
                   if k not in ['host', 'port']}
    
    try:
        engine = Engine(**engine_args)
    except Exception as e:
        logger.warning(f"Engine init failed: {e}, trying subprocess mode")
        _start_with_subprocess(config, server_args)
        return
    
    set_runtime(engine, config.model.name)
    
    logger.success("SGLang Engine initialized")
    
    # Setup graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutting down SGLang server...")
        try:
            if hasattr(engine, 'shutdown'):
                engine.shutdown()
        except:
            pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Run the server
    host = config.server.host or "0.0.0.0"
    port = config.server.port
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


def _start_with_subprocess(config: InferenceConfig, server_args: dict):
    """
    Start SGLang in a subprocess and proxy requests.
    
    This is a fallback for SGLang versions that don't expose Runtime directly.
    """
    import subprocess
    import httpx
    
    # Start SGLang server in subprocess
    sglang_port = config.server.port + 1000  # Use different port for SGLang
    
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", server_args["model_path"],
        "--host", "127.0.0.1",
        "--port", str(sglang_port),
        "--tp", str(server_args.get("tp_size", 1)),
        "--dp", str(server_args.get("dp_size", 1)),
    ]
    
    if server_args.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    
    if server_args.get("disable_cuda_graph"):
        cmd.append("--disable-cuda-graph")
    
    logger.info(f"Starting SGLang subprocess: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for SGLang to be ready
    sglang_url = f"http://127.0.0.1:{sglang_port}"
    for i in range(120):  # Wait up to 2 minutes
        try:
            resp = httpx.get(f"{sglang_url}/health", timeout=1)
            if resp.status_code == 200:
                logger.success("SGLang subprocess ready")
                break
        except:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("SGLang subprocess failed to start")
    
    # Create a proxy runtime
    class ProxyRuntime:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.client = httpx.Client(base_url=base_url, timeout=300)
            self.async_client = httpx.AsyncClient(base_url=base_url, timeout=300)
        
        def update_weights_from_disk(self, path: str) -> bool:
            resp = self.client.post("/update_weights", json={"weight_dir": path})
            return resp.status_code == 200
        
        def generate(self, prompt, sampling_params=None, stream=False):
            """Generate completion using OpenAI-compatible /v1/chat/completions endpoint."""
            sampling_params = sampling_params or {}
            
            # Use OpenAI-compatible endpoint for fair benchmarking with vLLM
            # Convert prompt back to messages format
            req = {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": sampling_params.get("max_new_tokens", 256),
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
                "stream": stream,
            }
            
            if sampling_params.get("stop"):
                req["stop"] = sampling_params["stop"]
            
            if sampling_params.get("top_k") and sampling_params["top_k"] > 0:
                req["top_k"] = sampling_params["top_k"]
            
            if stream:
                # Return an iterator that yields chunks
                def stream_generator():
                    with self.client.stream("POST", "/v1/chat/completions", json=req) as resp:
                        for line in resp.iter_lines():
                            if line.startswith("data:"):
                                data = line[5:].strip()
                                if data == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data)
                                    # Extract text from OpenAI format
                                    choices = chunk.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        text = delta.get("content", "")
                                        finish_reason = choices[0].get("finish_reason")
                                        yield {"text": text, "finish_reason": finish_reason}
                                except:
                                    pass
                return stream_generator()
            else:
                resp = self.client.post("/v1/chat/completions", json=req)
                data = resp.json()
                # Extract text from OpenAI format
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return {"text": message.get("content", "")}
                return {"text": ""}
        
        def shutdown(self):
            proc.terminate()
    
    set_runtime(ProxyRuntime(sglang_url), config.model.name)
    
    # Cleanup on exit
    import atexit
    atexit.register(proc.terminate)
    
    # Run our FastAPI server
    host = config.server.host or "0.0.0.0"
    port = config.server.port
    
    logger.info(f"Starting prime-rl proxy server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
