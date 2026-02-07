# API Reference

DeepInfer provides an OpenAI-compatible REST API for text generation.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, authentication is optional. To enable API key authentication:

```yaml
# config.yaml
server:
  api_key: "your-secret-key"
```

Include in requests:
```bash
curl -H "Authorization: Bearer your-secret-key" ...
```

## Endpoints

### Health Check

Check server status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "gpus": 1,
  "nvidia_5090": true
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

### GPU Information

Get detailed GPU information.

**Endpoint**: `GET /v1/gpu/info`

**Response**:
```json
{
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 5090",
      "total_memory_gb": 32.0,
      "free_memory_gb": 28.5,
      "compute_capability": "8.9",
      "driver_version": "550.54.14",
      "cuda_version": "12.1",
      "is_nvidia_5090": true,
      "is_nvidia_4090": false
    }
  ]
}
```

**Example**:
```bash
curl http://localhost:8000/v1/gpu/info
```

---

### List Models

List available models.

**Endpoint**: `GET /v1/models`

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "created": 1704067200,
      "owned_by": "deepinfer"
    }
  ]
}
```

**Example**:
```bash
curl http://localhost:8000/v1/models
```

---

### Create Completion

Generate text completion.

**Endpoint**: `POST /v1/completions`

**Request Body**:
```json
{
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": -1,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0,
  "stop": null,
  "stream": false,
  "n": 1
}
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | required | Input prompt |
| max_tokens | integer | 512 | Maximum tokens to generate |
| temperature | float | 0.7 | Sampling temperature (0-2) |
| top_p | float | 0.95 | Nucleus sampling probability |
| top_k | integer | -1 | Top-k sampling (-1 = disabled) |
| frequency_penalty | float | 0.0 | Frequency penalty (0-2) |
| presence_penalty | float | 0.0 | Presence penalty (0-2) |
| repetition_penalty | float | 1.0 | Repetition penalty |
| stop | array | null | Stop sequences |
| stream | boolean | false | Stream response |
| n | integer | 1 | Number of completions |

**Response**:
```json
{
  "id": "cmpl-1704067200",
  "object": "text_completion",
  "created": 1704067200,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "text": " bright and full of possibilities...",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 20,
    "total_tokens": 25
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

### Create Chat Completion

Generate chat completion.

**Endpoint**: `POST /v1/chat/completions`

**Request Body**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": false,
  "stop": null
}
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| messages | array | required | List of messages |
| max_tokens | integer | 512 | Maximum tokens to generate |
| temperature | float | 0.7 | Sampling temperature |
| top_p | float | 0.95 | Nucleus sampling probability |
| stream | boolean | false | Stream response |
| stop | array | null | Stop sequences |

**Message Format**:
```json
{
  "role": "system|user|assistant",
  "content": "message content"
}
```

**Response**:
```json
{
  "id": "chatcmpl-1704067200",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

---

## Streaming

For streaming responses, set `stream: true` in the request.

**Example**:
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

**Streaming Response** (Server-Sent Events):
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704067200,"model":"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704067200,"model":"...","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704067200,"model":"...","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

...

data: [DONE]
```

## Error Responses

**400 Bad Request**:
```json
{
  "detail": "Invalid request parameters"
}
```

**503 Service Unavailable**:
```json
{
  "detail": "Engine not initialized"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Error message"
}
```

## Rate Limiting

Currently not implemented. Can be added using middleware.

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Chat completion
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

## OpenAI SDK Compatibility

DeepInfer is compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # or your API key if configured
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```
