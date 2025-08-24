# TODO

This file tracks next actions and how to validate streaming output for the new OpenAI-compatible API.

## Why current streaming may appear blocked locally
- On first run, the Python worker loads a Hugging Face model (default: `sshleifer/tiny-gpt2`).
- If the model is not cached locally, it must be downloaded from the internet before any SSE tokens are produced.
- During this initial download, the `/v1/chat/completions` streaming response will not emit events yet, so it looks like it is “stuck”.

## Options to make streaming test work immediately

A) Use a local model directory (recommended if you already have one)
- Provide a local path for `MODEL_ID` that contains model files (e.g., `config.json`, `model.safetensors`, etc.).
- Example usage when starting the worker:
  ```bash
  PYTHONPATH=python/worker \
  WORKER_ADDRESS=127.0.0.1:50051 \
  MODEL_ID=/absolute/path/to/local-model \
  MODEL_DEVICE=cpu \
  .venv/bin/python -m python.worker.server
  ```

B) Warm up by downloading a tiny model (internet required)
- Pre-download the default tiny model to cache, then run streaming test.
- Example warm-up:
  ```bash
  .venv/bin/python - <<'PY'
  from transformers import AutoModelForCausalLM, AutoTokenizer
  m = "sshleifer/tiny-gpt2"
  AutoTokenizer.from_pretrained(m)
  AutoModelForCausalLM.from_pretrained(m)
  print("downloaded")
  PY
  ```

C) Use vLLM on a GPU host
- If you have a GPU/Linux machine with vLLM, run the vLLM worker instead (keeps PagedAttention, etc.).
- Example usage:
  ```bash
  # Install vLLM in a proper environment
  pip install "vllm>=0.5.0" torch --extra-index-url https://download.pytorch.org/whl/cu121

  # Start the vLLM-based worker
  WORKER_ADDRESS=0.0.0.0:50051 \
  MODEL_ID=meta-llama/Llama-3-8B \
  TP_SIZE=1 \
  python -m python.worker.vllm_worker
  ```

## End-to-end streaming test
- Start Python worker (choose A/B/C above). For HF worker (CPU):
  ```bash
  PYTHONPATH=python/worker \
  WORKER_ADDRESS=127.0.0.1:50051 \
  MODEL_DEVICE=cpu \
  .venv/bin/python -m python.worker.server
  ```

- Start Rust API server:
  ```bash
  RUST_LOG=info BATCH_MAX=8 BATCH_DELAY_MS=10 cargo run -p api-server
  ```

- Health check:
  ```bash
  curl -s http://127.0.0.1:8080/health | jq .
  ```

- OpenAI-compatible streaming request:
  ```bash
  curl -N -s -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":8,"stream":true}'
  ```

## Notes / next steps
- Sampling parameters are not fully wired into the workers yet (temperature/top_p/etc.). We can map `SamplingParams` in proto to HF/vLLM implementations next.
- The chat prompt is built with a simple role-prefix template; for best quality, add model-specific chat templates.
- Function/tool calls and advanced OpenAI response fields are not implemented yet; we can extend the route after sampling is in place.

