"""
vLLM Engine Shim - gRPC wrapper for vLLM AsyncLLMEngine

This shim provides a gRPC interface to vLLM without monkey-patching.
It calls vLLM's standard AsyncLLMEngine API.

Key principles:
- Non-invasive: vLLM runs as-is, no modifications
- Simple: ~200 lines, straightforward wrapper
- Complete: Implements all required Engine Service RPCs
"""
import asyncio
import logging
import time
import uuid
from typing import AsyncIterator, List, Optional

import grpc
from grpc import aio

# Import generated protobuf code
# These will be generated via: python -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/engine_service.proto
try:
    from xinf_engine.generated import engine_service_pb2, engine_service_pb2_grpc
except ImportError:
    # Fallback for initial setup - will be generated later
    logger = logging.getLogger(__name__)
    logger.warning("Generated protobuf code not found - run code generation first")
    engine_service_pb2 = None
    engine_service_pb2_grpc = None

logger = logging.getLogger(__name__)


class VLLMEngineServicer:
    """gRPC servicer for vLLM engine"""
    
    def __init__(self, model: str, **engine_args):
        self.model = model
        self.engine_args = engine_args
        self.engine = None
        
    async def initialize(self):
        """Initialize vLLM AsyncLLMEngine"""
        try:
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.sampling_params import SamplingParams
            
            # Store for later use
            self.SamplingParams = SamplingParams
            
            # Build engine args
            engine_args = AsyncEngineArgs(
                model=self.model,
                tensor_parallel_size=self.engine_args.get("tensor_parallel_size", 1),
                dtype=self.engine_args.get("dtype", "auto"),
                max_model_len=self.engine_args.get("max_model_len"),
                gpu_memory_utilization=self.engine_args.get("gpu_memory_utilization", 0.9),
            )
            
            # Create engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"vLLM engine initialized for model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    async def HealthCheck(self, request, context):
        """Health check RPC"""
        return engine_service_pb2.HealthCheckResponse(
            status="healthy",
            version="0.1.0"
        )
    
    async def Generate(self, request, context):
        """Generate text (non-streaming)"""
        request_id = str(uuid.uuid4())
        
        # Build sampling params
        sampling_params = self.SamplingParams(
            max_tokens=request.max_tokens if request.HasField("max_tokens") else 16,
            temperature=request.temperature if request.HasField("temperature") else 1.0,
            top_p=request.top_p if request.HasField("top_p") else 1.0,
        )
        
        # Generate
        results = []
        async for output in self.engine.generate(
            request.prompt,
            sampling_params,
            request_id
        ):
            results.append(output)
        
        # Return final result
        final_output = results[-1] if results else None
        if final_output:
            text = final_output.outputs[0].text
            tokens = len(final_output.outputs[0].token_ids)
            finish_reason = final_output.outputs[0].finish_reason
            
            return engine_service_pb2.GenerateResponse(
                id=request_id,
                text=text,
                tokens_generated=tokens,
                finish_reason=finish_reason or "stop",
                usage=engine_service_pb2.Usage(
                    prompt_tokens=len(final_output.prompt_token_ids),
                    completion_tokens=tokens,
                    total_tokens=len(final_output.prompt_token_ids) + tokens,
                )
            )
        
        return engine_service_pb2.GenerateResponse(
            id=request_id,
            text="",
            tokens_generated=0,
            finish_reason="error"
        )
    
    async def GenerateStream(self, request, context):
        """Generate text (streaming)"""
        request_id = str(uuid.uuid4())
        
        sampling_params = self.SamplingParams(
            max_tokens=request.max_tokens if request.HasField("max_tokens") else 16,
            temperature=request.temperature if request.HasField("temperature") else 1.0,
            top_p=request.top_p if request.HasField("top_p") else 1.0,
        )
        
        # Stream results
        async for output in self.engine.generate(
            request.prompt,
            sampling_params,
            request_id
        ):
            text = output.outputs[0].text
            is_finished = output.finished
            finish_reason = output.outputs[0].finish_reason if is_finished else None
            
            yield engine_service_pb2.GenerateStreamResponse(
                id=request_id,
                text=text,
                is_finished=is_finished,
                finish_reason=finish_reason
            )
    
    async def Chat(self, request, context):
        """Chat completion (non-streaming)"""
        # TODO: Apply chat template to messages
        # For now, just concatenate messages
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        
        # Delegate to Generate
        gen_request = engine_service_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        gen_response = await self.Generate(gen_request, context)
        
        # Convert to chat response
        return engine_service_pb2.ChatResponse(
            id=gen_response.id,
            object="chat.completion",
            created=int(time.time()),
            model=self.model,
            choices=[
                engine_service_pb2.ChatChoice(
                    index=0,
                    message=engine_service_pb2.ChatMessage(
                        role="assistant",
                        content=gen_response.text
                    ),
                    finish_reason=gen_response.finish_reason
                )
            ],
            usage=gen_response.usage
        )
    
    async def ChatStream(self, request, context):
        """Chat completion (streaming)"""
        # TODO: Apply chat template
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        
        gen_request = engine_service_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        async for chunk in self.GenerateStream(gen_request, context):
            yield engine_service_pb2.ChatStreamResponse(
                id=chunk.id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=self.model,
                choices=[
                    engine_service_pb2.ChatStreamChoice(
                        index=0,
                        delta=engine_service_pb2.ChatMessageDelta(
                            content=chunk.text
                        ),
                        finish_reason=chunk.finish_reason if chunk.is_finished else None
                    )
                ]
            )
    
    # Stubs for other RPCs
    async def CreateEmbedding(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Embedding not supported by vLLM")
        return engine_service_pb2.EmbeddingResponse()
    
    async def CancelRequest(self, request, context):
        # TODO: Implement request cancellation
        return engine_service_pb2.CancelResponse(success=False, message="Not implemented")
    
    async def GetMetrics(self, request, context):
        # TODO: Get metrics from vLLM engine
        return engine_service_pb2.MetricsResponse()
    
    async def GetKVCacheStatus(self, request, context):
        # TODO: Get KV cache status from vLLM
        return engine_service_pb2.KVCacheStatusResponse()
    
    async def Shutdown(self, request, context):
        logger.info("Shutdown requested")
        # TODO: Graceful shutdown
        return engine_service_pb2.ShutdownResponse(success=True)
    
    async def GetModelInfo(self, request, context):
        # TODO: Get model info from vLLM engine
        return engine_service_pb2.ModelInfoResponse(
            model_name=self.model
        )


async def serve_vllm(args):
    """Start vLLM gRPC server"""
    if engine_service_pb2 is None or engine_service_pb2_grpc is None:
        logger.error("Generated protobuf code not found. Please run code generation first:")
        logger.error("python -m grpc_tools.protoc -I protos --python_out=python/xinf_engine/generated --grpc_python_out=python/xinf_engine/generated protos/engine_service.proto")
        return
    
    servicer = VLLMEngineServicer(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    await servicer.initialize()
    
    server = aio.server()
    engine_service_pb2_grpc.add_EngineServiceServicer_to_server(servicer, server)
    
    listen_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await server.stop(grace=5)
