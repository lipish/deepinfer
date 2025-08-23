import asyncio
import os
import grpc
from typing import Dict, Optional

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams as VSampling

from engine.v1 import engine_pb2, engine_pb2_grpc


class VllmWorker(engine_pb2_grpc.EngineWorkerServicer):
    def __init__(self) -> None:
        self._engine: Optional[AsyncLLMEngine] = None
        self._sessions = set()
        self._next_session = 1
        self._next_seq = 1
        self._seq2req: Dict[str, str] = {}

    async def Health(self, request: engine_pb2.HealthRequest, context: grpc.aio.ServicerContext) -> engine_pb2.HealthResponse:
        return engine_pb2.HealthResponse(ok=True, worker_version="vllm-worker-0.1")

    async def _ensure_engine(self, model_id: str, device: str) -> None:
        if self._engine is not None:
            return
        # Minimal engine args; expand as needed via env vars
        tp = int(os.environ.get("TP_SIZE", "1"))
        dtype = os.environ.get("VLLM_DTYPE", "auto")
        args = EngineArgs(model=model_id, tensor_parallel_size=tp, dtype=dtype)
        self._engine = await AsyncLLMEngine.from_engine_args(args)

    async def CreateSession(self, request: engine_pb2.CreateSessionRequest, context: grpc.aio.ServicerContext) -> engine_pb2.CreateSessionResponse:
        try:
            model_id = request.model_id or os.environ.get("MODEL_ID", "facebook/opt-125m")
            device = request.device or os.environ.get("MODEL_DEVICE", "cuda:0")
            await self._ensure_engine(model_id, device)
        except Exception as e:
            return engine_pb2.CreateSessionResponse(ok=False, session_id="", error=str(e))
        sid = f"sess-{self._next_session}"; self._next_session += 1
        self._sessions.add(sid)
        return engine_pb2.CreateSessionResponse(ok=True, session_id=sid)

    async def StartSequence(self, request: engine_pb2.StartSequenceRequest, context: grpc.aio.ServicerContext) -> engine_pb2.StartSequenceResponse:
        if request.session_id not in self._sessions:
            return engine_pb2.StartSequenceResponse(ok=False, seq_id="", error="invalid session")
        seq_id = f"seq-{self._next_seq}"; self._next_seq += 1
        self._seq2req[seq_id] = ""  # placeholder; set in Prefill
        return engine_pb2.StartSequenceResponse(ok=True, seq_id=seq_id)

    async def Prefill(self, request: engine_pb2.PrefillRequest, context: grpc.aio.ServicerContext) -> engine_pb2.PrefillResponse:
        if request.session_id not in self._sessions:
            return engine_pb2.PrefillResponse(ok=False, consumed=0, error="invalid session")
        try:
            req_id = f"req-{request.seq_id}"
            sampling = VSampling(temperature=1.0)
            assert self._engine is not None
            await self._engine.add_request(request_id=req_id, prompt_token_ids=list(request.tokens), sampling_params=sampling)
            self._seq2req[request.seq_id] = req_id
            return engine_pb2.PrefillResponse(ok=True, consumed=len(request.tokens))
        except Exception as e:
            return engine_pb2.PrefillResponse(ok=False, consumed=0, error=str(e))

    async def Decode(self, request: engine_pb2.DecodeRequest, context: grpc.aio.ServicerContext) -> engine_pb2.DecodeResponse:
        rid = self._seq2req.get(request.seq_id)
        if not rid:
            return engine_pb2.DecodeResponse(ok=False, error="invalid seq")
        try:
            assert self._engine is not None
            # Only support sample_in_worker path; step until we see output for rid
            while True:
                outs = await self._engine.step()
                for out in outs:
                    if out.request_id == rid and out.outputs:
                        t = out.outputs[0].token_ids[-1]
                        return engine_pb2.DecodeResponse(ok=True, next_token=int(t))
                await asyncio.sleep(0)
        except Exception as e:
            return engine_pb2.DecodeResponse(ok=False, error=str(e))

    async def ReleaseSequence(self, request: engine_pb2.ReleaseSequenceRequest, context: grpc.aio.ServicerContext) -> engine_pb2.ReleaseSequenceResponse:
        rid = self._seq2req.pop(request.seq_id, None)
        try:
            if rid and self._engine is not None:
                await self._engine.abort_request(rid)
            return engine_pb2.ReleaseSequenceResponse(ok=True)
        except Exception as e:
            return engine_pb2.ReleaseSequenceResponse(ok=False, error=str(e))


async def serve() -> None:
    addr = os.environ.get("WORKER_ADDRESS", "127.0.0.1:50051")
    server = grpc.aio.server()
    engine_pb2_grpc.add_EngineWorkerServicer_to_server(VllmWorker(), server)
    server.add_insecure_port(addr)
    await server.start()
    print(f"[vllm-worker] listening on {addr}", flush=True)
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())

