# deepinfer 测试指南

## 测试环境

| 项目 | 配置 |
|------|------|
| 服务器 | 10.21.11.166 |
| GPU | 2x NVIDIA RTX 5090 (Blackwell SM 12.0) |
| CUDA | 12.8 |
| 驱动 | 570.211.01 |
| vLLM | v0.11.0 (Docker) |

## 快速测试

### 1. 健康检查

```bash
curl http://localhost:8082/health
# 期望: {"status":"healthy"}
```

### 2. 部署模型

```bash
curl -X POST http://localhost:8082/v1/deployments \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "backend": "docker",
    "docker_image": "vllm/vllm-openai:v0.11.0",
    "model_path": "/home/ai/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct",
    "device": "0,1",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.8
  }'
# 期望: {"status":"accepted","message":"Model launch request accepted...","model":"Qwen2.5-1.5B-Instruct"}
```

### 3. 等待引擎就绪

```bash
# 检查 etcd 中的引擎状态
etcdctl get --prefix /engines | grep -o '"status":"[^"]*"'
# 期望: "status":"running"

# 检查 Docker 容器
docker ps | grep deepinfer-engine
```

### 4. 聊天测试

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'
# 期望: {"id":"chatcmpl-...","choices":[{"message":{"content":"..."}}],...}
```

## 功能测试

### 单 GPU 部署

```bash
# 部署
curl -X POST http://localhost:8082/v1/deployments \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-single-gpu",
    "backend": "docker",
    "docker_image": "vllm/vllm-openai:v0.11.0",
    "model_path": "/path/to/model",
    "device": "0",
    "tensor_parallel_size": 1
  }'

# 验证 GPU 使用
nvidia-smi --query-gpu=index,memory.used --format=csv
```

### 多 GPU 张量并行

```bash
# 部署（2 GPU）
curl -X POST http://localhost:8082/v1/deployments \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-multi-gpu",
    "backend": "docker",
    "docker_image": "vllm/vllm-openai:v0.11.0",
    "model_path": "/path/to/model",
    "device": "0,1",
    "tensor_parallel_size": 2
  }'

# 验证双卡显存使用（应该对称）
nvidia-smi --query-gpu=index,memory.used --format=csv
# 期望: 两张卡显存使用量相近
```

### 长文本生成

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Write a 500 word essay about AI."}],
    "max_tokens": 1000
  }'
```

### 多轮对话

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "My name is Alice."},
      {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
      {"role": "user", "content": "What is my name?"}
    ],
    "max_tokens": 50
  }'
# 期望: 模型应该记住用户名字是 Alice
```

## 性能测试

### 吞吐量测试

```bash
# 安装 hey（HTTP 负载测试工具）
go install github.com/rakyll/hey@latest

# 并发测试
hey -n 100 -c 10 \
  -m POST \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}' \
  http://localhost:8082/v1/chat/completions
```

### 延迟测试

```bash
# 首 Token 延迟 (TTFT)
time curl -s -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}'
```

## 状态检查

### etcd 数据

```bash
# 列出所有引擎
etcdctl get --prefix /engines

# 格式化显示
etcdctl get --prefix /engines --print-value-only | python3 -m json.tool

# 删除失败的引擎
etcdctl del /engines/{engine_id}
```

### Docker 容器

```bash
# 查看运行中的容器
docker ps --filter "name=deepinfer"

# 查看容器日志
docker logs deepinfer-engine-{id} --tail 100

# 进入容器调试
docker exec -it deepinfer-engine-{id} bash
```

### 服务日志

```bash
# Gateway 日志
tail -f /tmp/gateway.log

# Worker Agent 日志
tail -f /tmp/worker.log
```

## 错误排查

### 引擎启动失败

```bash
# 检查 etcd 中的错误信息
etcdctl get --prefix /engines --print-value-only | grep error_message

# 检查 Docker 容器日志
docker logs deepinfer-engine-{id}
```

### 推理请求失败

```bash
# 检查引擎状态
etcdctl get --prefix /engines --print-value-only | grep status

# 直接测试 vLLM 容器
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/model","messages":[{"role":"user","content":"test"}]}'
```

### GPU 问题

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## 清理测试环境

```bash
# 停止所有 deepinfer 容器
docker stop $(docker ps -q --filter "name=deepinfer")
docker rm $(docker ps -aq --filter "name=deepinfer")

# 清理 etcd 数据
etcdctl del --prefix /engines

# 停止服务
pkill -f "deepinfer serve"
pkill -f "deepinfer worker"
```

## 测试检查清单

- [ ] Gateway 健康检查通过
- [ ] etcd 连接正常
- [ ] Worker Agent 启动成功
- [ ] 单 GPU 部署正常
- [ ] 多 GPU 张量并行正常
- [ ] 聊天请求返回正确
- [ ] GPU 显存使用正常
- [ ] 错误处理正确
