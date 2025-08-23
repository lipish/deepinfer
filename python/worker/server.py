import asyncio
import os
import time
import grpc
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.v1 import engine_pb2, engine_pb2_grpc


ModelTuple = Tuple[AutoModelForCausalLM, AutoTokenizer]


def _pad_and_mask(batch_ids: List[List[int]], pad_id: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Pad variable-length sequences to a tensor and return attention mask and lengths."""
    if not batch_ids:
        raise ValueError("empty batch")
    lengths = [len(x) if len(x) > 0 else 1 for x in batch_ids]
    max_len = max(lengths)
    input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(batch_ids):
        if len(seq) == 0:
            continue
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    attn_mask = (input_ids != pad_id).long()
    return input_ids, attn_mask, lengths


class _MicroBatcher:
    """A very small decode micro-batcher that batches concurrent single-step Decode calls.

    Only used when sample_in_worker=true. Greedy sampling for now.
    """

    def __init__(self, get_seq_tokens, append_token, model: AutoModelForCausalLM, device: str, pad_id: int,
                 max_batch_size: int = 8, max_delay_ms: int = 5) -> None:
        self._get_seq_tokens = get_seq_tokens
        self._append_token = append_token
        self._model = model
        self._device = device
        self._pad_id = pad_id
        self._max_batch_size = max_batch_size
        self._max_delay = max_delay_ms / 1000.0
        self._q: asyncio.Queue[Tuple[asyncio.Future, str]] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def submit(self, seq_id: str) -> int:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._q.put((fut, seq_id))
        return await fut

    async def _run(self) -> None:
        while True:
            fut, seq_id = await self._q.get()
            batch: List[Tuple[asyncio.Future, str]] = [(fut, seq_id)]
            start = time.perf_counter()
            # try to gather more within delay window
            while len(batch) < self._max_batch_size:
                remaining = self._max_delay - (time.perf_counter() - start)
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._q.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            try:
                # Build batch inputs from current kvs
                seq_ids = [sid for (_, sid) in batch]
                seqs = [self._get_seq_tokens(sid) for sid in seq_ids]
                input_ids, attn_mask, lengths = _pad_and_mask(seqs, self._pad_id, self._device)
                with torch.no_grad():
                    out = self._model(input_ids=input_ids, attention_mask=attn_mask)
                    last_logits = out.logits.float()
                next_tokens: List[int] = []
                for i, L in enumerate(lengths):
                    idx = max(0, L - 1)
                    nxt = int(torch.argmax(last_logits[i, idx, :], dim=-1).item())
                    self._append_token(seq_ids[i], nxt)
                    next_tokens.append(nxt)
                for (f, _sid), t in zip(batch, next_tokens):
                    if not f.done():
                        f.set_result(t)
            except Exception as e:
                for f, _sid in batch:
                    if not f.done():
                        f.set_exception(e)


class EngineWorker(engine_pb2_grpc.EngineWorkerServicer):
    def __init__(self) -> None:
        self._sessions = set()
        self._next_session = 1
        self._next_seq = 1
        self._seq_kv: Dict[str, List[int]] = {}
        self._model: Optional[ModelTuple] = None
        self._device = os.environ.get("MODEL_DEVICE", "cpu")
        self._model_id = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")
        self._pad_id: int = 0
        self._mb: Optional[_MicroBatcher] = None

    async def Health(self, request: engine_pb2.HealthRequest, context: grpc.aio.ServicerContext) -> engine_pb2.HealthResponse:
        return engine_pb2.HealthResponse(ok=True, worker_version="py-worker-0.2-batch")

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
                # choose a pad id
                pad_id = tok.pad_token_id
                if pad_id is None:
                    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
                self._pad_id = int(pad_id)
                # start micro-batcher
                max_bs = int(os.environ.get("MB_MAX_BATCH", "8"))
                max_delay = int(os.environ.get("MB_DELAY_MS", "5"))
                self._mb = _MicroBatcher(
                    get_seq_tokens=lambda sid: list(self._seq_kv.get(sid, [])),
                    append_token=lambda sid, t: self._seq_kv.setdefault(sid, []).append(int(t)),
                    model=model,
                    device=self._device,
                    pad_id=self._pad_id,
                    max_batch_size=max_bs,
                    max_delay_ms=max_delay,
                )
                self._mb.start()
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

        # Record consumed tokens
        kv = self._seq_kv[request.seq_id]
        kv.extend(request.tokens)

        consumed = len(request.tokens)
        resp = engine_pb2.PrefillResponse(ok=True, consumed=consumed)

        if request.return_last_logits:
            try:
                input_ids, attn_mask, lengths = _pad_and_mask([kv], self._pad_id, self._device)
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                    last = out.logits[0, lengths[0] - 1, :].float().cpu().numpy().flatten()
                resp.last_logits.extend(last[: min(1000, last.shape[0])].tolist())
            except Exception as e:
                return engine_pb2.PrefillResponse(ok=False, consumed=0, error=str(e))
        return resp

    async def BatchPrefill(self, request: engine_pb2.BatchPrefillRequest, context: grpc.aio.ServicerContext) -> engine_pb2.BatchPrefillResponse:
        if self._model is None:
            return engine_pb2.BatchPrefillResponse(ok=False, error="model not loaded")
        model, _ = self._model

        seq_ids = list(request.seq_ids)
        if len(seq_ids) != len(request.tokens_list):
            return engine_pb2.BatchPrefillResponse(ok=False, error="length mismatch: seq_ids vs tokens_list")
        # Update kvs
        batch_inputs: List[List[int]] = []
        try:
            for i, sid in enumerate(seq_ids):
                if sid not in self._seq_kv:
                    return engine_pb2.BatchPrefillResponse(ok=False, error=f"invalid seq: {sid}")
                toks = list(request.tokens_list[i].tokens)
                self._seq_kv[sid].extend(toks)
                batch_inputs.append(self._seq_kv[sid])
        except Exception as e:
            return engine_pb2.BatchPrefillResponse(ok=False, error=str(e))

        consumed_list = [len(request.tokens_list[i].tokens) for i in range(len(seq_ids))]
        resp = engine_pb2.BatchPrefillResponse(ok=True)
        resp.consumed_list.extend(consumed_list)

        if request.return_last_logits:
            try:
                input_ids, attn_mask, lengths = _pad_and_mask(batch_inputs, self._pad_id, self._device)
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                    logits = out.logits.float().cpu().numpy()
                for i, L in enumerate(lengths):
                    last = logits[i, L - 1, :].flatten()
                    fl = engine_pb2.FloatList()
                    fl.values.extend(last[: min(1000, last.shape[0])].tolist())
                    resp.last_logits_list.append(fl)
            except Exception as e:
                return engine_pb2.BatchPrefillResponse(ok=False, error=str(e))
        return resp

    async def Decode(self, request: engine_pb2.DecodeRequest, context: grpc.aio.ServicerContext) -> engine_pb2.DecodeResponse:
        if request.seq_id not in self._seq_kv:
            return engine_pb2.DecodeResponse(ok=False, error="invalid seq")
        if self._model is None:
            return engine_pb2.DecodeResponse(ok=False, error="model not loaded")
        model, _ = self._model

        try:
            # If we sample in worker, try the micro-batcher to batch concurrent decodes
            if request.sample_in_worker and self._mb is not None:
                try:
                    next_token = await self._mb.submit(request.seq_id)
                    return engine_pb2.DecodeResponse(ok=True, next_token=int(next_token))
                except Exception:
                    # fall back to single decode
                    pass

            kv = self._seq_kv[request.seq_id]
            input_ids, attn_mask, lengths = _pad_and_mask([kv], self._pad_id, self._device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits[:, lengths[0] - 1, :].float()
            if request.sample_in_worker:
                next_token = int(torch.argmax(logits, dim=-1).item())
                self._seq_kv[request.seq_id].append(next_token)
                return engine_pb2.DecodeResponse(ok=True, next_token=next_token)
            else:
                resp = engine_pb2.DecodeResponse(ok=True)
                resp.logits.extend(logits.cpu().numpy().flatten().tolist())
                return resp
        except Exception as e:
            return engine_pb2.DecodeResponse(ok=False, error=str(e))

    async def BatchDecode(self, request: engine_pb2.BatchDecodeRequest, context: grpc.aio.ServicerContext) -> engine_pb2.BatchDecodeResponse:
        if self._model is None:
            return engine_pb2.BatchDecodeResponse(ok=False, error="model not loaded")
        model, _ = self._model

        seq_ids = list(request.seq_ids)
        if len(seq_ids) == 0:
            return engine_pb2.BatchDecodeResponse(ok=False, error="empty batch")
        try:
            seqs = []
            for sid in seq_ids:
                if sid not in self._seq_kv:
                    return engine_pb2.BatchDecodeResponse(ok=False, error=f"invalid seq: {sid}")
                seqs.append(self._seq_kv[sid])
            input_ids, attn_mask, lengths = _pad_and_mask(seqs, self._pad_id, self._device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                last = out.logits.float()
            resp = engine_pb2.BatchDecodeResponse(ok=True)
            if request.sample_in_worker:
                next_tokens: List[int] = []
                for i, L in enumerate(lengths):
                    idx = max(0, L - 1)
                    nxt = int(torch.argmax(last[i, idx, :], dim=-1).item())
                    self._seq_kv[seq_ids[i]].append(nxt)
                    next_tokens.append(nxt)
                resp.next_tokens.extend(next_tokens)
            else:
                for i, L in enumerate(lengths):
                    fl = engine_pb2.FloatList()
                    vals = last[i, L - 1, :].cpu().numpy().flatten()
                    fl.values.extend(vals[: min(1000, vals.shape[0])].tolist())
                    resp.logits_list.append(fl)
            return resp
        except Exception as e:
            return engine_pb2.BatchDecodeResponse(ok=False, error=str(e))

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

