"""FastAPI server for DeepInfer."""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from deepinfer.config import Config
from deepinfer.engine import InferenceEngine
from deepinfer.gpu_utils import GPUDetector

logger = logging.getLogger(__name__)


# Request/Response Models
class CompletionRequest(BaseModel):
    """Completion request model."""
    model: Optional[str] = None
    prompt: str = Field(..., description="Input prompt")
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(default=-1, description="Top-k sampling")
    frequency_penalty: Optional[float] = Field(default=0.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=0.0, description="Presence penalty")
    repetition_penalty: Optional[float] = Field(default=1.0, description="Repetition penalty")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    n: Optional[int] = Field(default=1, ge=1, le=10, description="Number of completions")


class Message(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: Optional[str] = None
    messages: List[Message] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")


class CompletionChoice(BaseModel):
    """Completion choice model."""
    text: str
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """Completion response model."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


# Global engine instance
engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    global engine
    
    # Startup
    logger.info("Starting DeepInfer server...")
    
    # Initialize engine
    config = app.state.config
    engine = InferenceEngine(config=config)
    
    logger.info("DeepInfer server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DeepInfer server...")


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create FastAPI application."""
    if config is None:
        config = Config()
    
    app = FastAPI(
        title="DeepInfer",
        description="High-performance inference engine based on vLLM",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Store config in app state
    app.state.config = config
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        gpu_detector = GPUDetector()
        return {
            "status": "healthy",
            "model": config.model.name,
            "gpus": len(gpu_detector.gpus),
            "nvidia_5090": gpu_detector.has_nvidia_5090(),
        }
    
    # GPU info endpoint
    @app.get("/v1/gpu/info")
    async def gpu_info():
        """Get GPU information."""
        gpu_detector = GPUDetector()
        return {
            "gpus": [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "total_memory_gb": gpu.total_memory / 1024**3,
                    "free_memory_gb": gpu.free_memory / 1024**3,
                    "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
                    "driver_version": gpu.driver_version,
                    "cuda_version": gpu.cuda_version,
                    "is_nvidia_5090": gpu.is_nvidia_5090,
                    "is_nvidia_4090": gpu.is_nvidia_4090,
                }
                for gpu in gpu_detector.gpus
            ]
        }
    
    # Completions endpoint
    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        """Create text completion."""
        global engine
        
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        try:
            # Generate completion
            outputs = engine.generate(
                prompts=request.prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop,
            )
            
            # Format response
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            response = CompletionResponse(
                id=f"cmpl-{int(time.time())}",
                created=int(time.time()),
                model=config.model.name,
                choices=[
                    CompletionChoice(
                        text=generated_text,
                        index=0,
                        finish_reason=output.outputs[0].finish_reason,
                    )
                ],
                usage={
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Chat completions endpoint
    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        """Create chat completion."""
        global engine
        
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        try:
            # Convert messages to prompt
            prompt = ""
            for msg in request.messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n"
            prompt += "Assistant:"
            
            # Generate completion
            outputs = engine.generate(
                prompts=prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
            )
            
            # Format response
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": config.model.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text.strip(),
                        },
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Models endpoint
    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model.name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "deepinfer",
                }
            ]
        }
    
    return app


def main():
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepInfer Server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with command line arguments
    if args.model:
        config.model.name = args.model
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    
    # Create and run app
    app = create_app(config)
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()
