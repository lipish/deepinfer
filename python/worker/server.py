import asyncio
import os
import grpc
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.v1 import engine_pb2, engine_pb2_grpc


ModelTuple = Tuple[AutoModelForCausalLM, AutoTokenizer]


class EngineWorker(engine_pb2_grpc.EngineWorkerServicer):
    def __init__(self) -> None:
        self._sessions = set()
        self._next_session = 1
        self._next_seq = 1
        self._seq_kv: Dict[str, List[int]] = {}
        self._model: Optional[ModelTuple] = None
        self._device = os.environ.get("MODEL_DEVICE", "cpu")
        self._model_id = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")

    async def Health(self, request: engine_pb2.HealthRequest, context: grpc.aio.ServicerContext) -> engine_pb2.HealthResponse:
        return engine_pb2.HealthResponse(ok=True, worker_version="py-worker-0.1")

    async def CreateSession(self, request: engine_pb2.CreateSessionRequest, context: grpc.aio.ServicerContext) -> engine_pb2.CreateSessionResponse:
        # lazily load model on first session
        try:
            if self._model is None:
                model_id = request.model_id or self._model_id
                tok = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                model = model.to(self._device)
                model.eval()
                self._model = (model, tok)
        except Exception as e:
            return engine_pb2.CreateSessionResponse(ok=False, session_id="", error=str(e))

        sid = f"sess-{self._next_session}"
        self._next_session += 1
        self._sessions.add(sid)
        return engine_pb2.CreateSessionResponse(ok=True, session_id=sid)

    async def StartSequence(self, request: engine_pb2.StartSequenceRequest, context: grpc.aio.ServicerContext) -> engine_pb2.StartSequenceResponse:
        if request.session_id not in self._sessions:
            return engine_pb2.StartSequenceResponse(ok=False, seq_id="", error="invalid session")
        seq_id = f"seq-{self._next_seq}"
        self._next_seq += 1
        self._seq_kv[seq_id] = []
        return engine_pb2.StartSequenceResponse(ok=True, seq_id=seq_id)

    async def Prefill(self, request: engine_pb2.PrefillRequest, context: grpc.aio.ServicerContext) -> engine_pb2.PrefillResponse:
        if request.seq_id not in self._seq_kv:
            return engine_pb2.PrefillResponse(ok=False, consumed=0, error="invalid seq")
        if self._model is None:
            return engine_pb2.PrefillResponse(ok=False, consumed=0, error="model not loaded")
        model, _ = self._model

        # In a real engine we'd compute logits and cache past_key_values.
        # For simplicity we don't expose PKV in the proto; we rely on the model's past caching implicitly via input ids.
        # Here we just record consumed tokens count; decode() will feed last token and use model.generate-like single step.
        kv = self._seq_kv[request.seq_id]
        kv.extend(request.tokens)

        consumed = len(request.tokens)
        resp = engine_pb2.PrefillResponse(ok=True, consumed=consumed)

        if request.return_last_logits:
            try:
                input_ids = torch.tensor([kv], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    out = model(input_ids=input_ids)
                    last_logits = out.logits[:, -1, :].float().cpu().numpy().flatten()
                # Truncate for bandwidth in this scaffold (not for production)
                resp.last_logits.extend(last_logits[:min(1000, last_logits.shape[0])].tolist())
            except Exception as e:
                return engine_pb2.PrefillResponse(ok=False, consumed=0, error=str(e))
        return resp

    async def Decode(self, request: engine_pb2.DecodeRequest, context: grpc.aio.ServicerContext) -> engine_pb2.DecodeResponse:
        if request.seq_id not in self._seq_kv:
            return engine_pb2.DecodeResponse(ok=False, error="invalid seq")
        if self._model is None:
            return engine_pb2.DecodeResponse(ok=False, error="model not loaded")
        model, _ = self._model

        try:
            # Feed the full prompt + last token; use model forward once and sample from last step
            kv = self._seq_kv[request.seq_id]
            input_ids = torch.tensor([kv], dtype=torch.long, device=self._device)
            with torch.no_grad():
                out = model(input_ids=input_ids)
                logits = out.logits[:, -1, :].float()
            if request.sample_in_worker:
                # Greedy for scaffold; you can add temperature/top-k/top-p later
                next_token = int(torch.argmax(logits, dim=-1).item())
                self._seq_kv[request.seq_id].append(next_token)
                return engine_pb2.DecodeResponse(ok=True, next_token=next_token)
            else:
                resp = engine_pb2.DecodeResponse(ok=True)
                resp.logits.extend(logits.cpu().numpy().flatten().tolist())
                return resp
        except Exception as e:
            return engine_pb2.DecodeResponse(ok=False, error=str(e))

    async def ReleaseSequence(self, request: engine_pb2.ReleaseSequenceRequest, context: grpc.aio.ServicerContext) -> engine_pb2.ReleaseSequenceResponse:
        self._seq_kv.pop(request.seq_id, None)
        return engine_pb2.ReleaseSequenceResponse(ok=True)


async def serve() -> None:
    addr = os.environ.get("WORKER_ADDRESS", "127.0.0.1:50051")
    server = grpc.aio.server()
    engine_pb2_grpc.add_EngineWorkerServicer_to_server(EngineWorker(), server)
    server.add_insecure_port(addr)
    await server.start()
    print(f"[py-worker] listening on {addr}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())

