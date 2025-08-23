# deepinfer (skeleton)

Rust API server + Python gRPC worker skeleton for a vLLM-like engine.

What you get:
- gRPC proto definition (EngineWorker)
- Rust workspace:
  - api-server: minimal HTTP API that calls the Python worker via gRPC
  - worker-proto: generated gRPC client types
- Python worker (grpc.aio) with placeholder logic (no real model)
- YAML config and dev scripts

Prerequisites:
- Rust toolchain (stable)
- Python 3.9+

Quickstart:
0) Fetch a tokenizer.json (GPT-2)
   bash scripts/fetch.tokenizer.json.sh

1) Start the Python worker (first run creates venv, installs torch/transformers, generates Python stubs)
   bash scripts/dev.run.worker.sh

2) Start the Rust API server
   bash scripts/dev.run.api.sh

3) Test (non-stream)
   curl http://127.0.0.1:8080/health
   curl -X POST http://127.0.0.1:8080/v1/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"hello vllm-like", "max_new_tokens":5, "temperature":1.0}'

4) Test (SSE stream)
   curl -N -H 'Accept: text/event-stream' -H 'Content-Type: application/json' \
        -X POST http://127.0.0.1:8080/v1/generate \
        -d '{"prompt":"hello vllm-like", "max_new_tokens":5, "stream":true, "sampling":{"temperature":1.0, "top_p":0.9}}'

Notes:
- API uses Rust tokenizers to encode/decode with tokenizer.json (GPT-2 BPE).
- Python worker runs sshleifer/tiny-gpt2 on CPU with caching past_key_values and sampling.
- This is still a scaffold suitable for correctness and E2E plumbing. Performance optimizations and batching are to be added.

