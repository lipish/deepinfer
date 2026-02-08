# deepinfer 架构设计

## 概述

deepinfer 是一个 Rust-Native 推理编排平台，采用声明式调度架构，通过 Docker 容器运行 vLLM 推理引擎。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (curl/SDK)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Gateway (Axum HTTP)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ /v1/deploy  │  │ /v1/chat/   │  │ /health, /v1/models     │  │
│  │   ments     │  │ completions │  │                         │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────┘  │
│         │                │                                       │
│         ▼                ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Router (KV-aware)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MetaStore (etcd)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  /engines/{id} → RunningEngine { status, endpoint, ... } │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Watch
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Worker Agent (node-1)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Reconciler  │  │  Heartbeat  │  │    Engine Launcher      │  │
│  │   (Watch)   │  │   Sender    │  │  (Docker Container)     │  │
│  └──────┬──────┘  └─────────────┘  └───────────┬─────────────┘  │
└─────────┼──────────────────────────────────────┼────────────────┘
          │                                      │
          ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Docker vLLM Container                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  vllm/vllm-openai:v0.11.0                                │    │
│  │  • OpenAI-compatible API (HTTP :8000)                    │    │
│  │  • Tensor Parallelism (multi-GPU)                        │    │
│  │  • RTX 5090 / Blackwell SM 12.0 Support                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Gateway (deepinfer-gateway)

API 网关，基于 Axum 构建，提供 OpenAI 兼容接口。

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/deployments` | POST | 部署新模型 |
| `/v1/chat/completions` | POST | 聊天补全 |

**职责：**
- 接收客户端请求
- 路由到对应的后端引擎
- HTTP/gRPC 协议转换

### 2. MetaStore (deepinfer-meta)

基于 etcd 的分布式元数据存储。

**数据结构：**
```
/engines/{engine_id} → RunningEngine {
    engine_id: UUID,
    config: EngineConfig,
    status: Pending | Starting | Running | Failed,
    endpoint: { address, port, protocol },
    node_id: String,
    device_indices: [0, 1],
    container_id: String,
}
```

**操作：**
- `put(key, value)` - 写入
- `get(key)` - 读取
- `list(prefix)` - 列表
- `watch(prefix)` - 监听变更

### 3. Worker Agent (deepinfer-agent)

运行在每个节点上的代理，负责引擎生命周期管理。

**子组件：**

| 组件 | 职责 |
|------|------|
| Reconciler | Watch etcd，执行调度意图 |
| Engine Launcher | 启动/停止 Docker 容器 |
| Heartbeat | 定期上报节点状态 |
| Health Check | 检查引擎健康状态 |

### 4. Engine Launcher

Docker 容器生命周期管理。

**启动流程：**
```
1. 检测 Pending 状态的引擎
2. 构建 docker run 命令
3. 启动容器（挂载模型、分配 GPU）
4. 等待健康检查通过
5. 更新状态为 Running
```

**Docker 命令示例：**
```bash
docker run -d \
  --name deepinfer-engine-{id} \
  --gpus "device=0,1" \
  -v /path/to/model:/model \
  -p 8000:8000 \
  vllm/vllm-openai:v0.11.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8
```

### 5. Router (deepinfer-router)

请求路由器，支持 KV-cache 感知的会话亲和性。

**路由策略：**
- 会话亲和性：相同 session_id 路由到同一引擎
- 最少连接：选择负载最低的引擎

### 6. Scheduler (deepinfer-scheduler)

资源调度器（当前为简化实现）。

**调度策略：**
- `idle_first` - 优先选择空闲节点

## 数据流

### 模型部署流程

```
1. Client → POST /v1/deployments
2. Gateway → 创建 RunningEngine (status: Pending)
3. Gateway → 写入 etcd /engines/{id}
4. Worker Agent → Watch 检测到新引擎
5. Reconciler → 领取任务，更新 status: Starting
6. Engine Launcher → docker run ...
7. Engine Launcher → 等待健康检查
8. Reconciler → 更新 status: Running, endpoint: {...}
```

### 推理请求流程

```
1. Client → POST /v1/chat/completions
2. Gateway → 查询 etcd 获取 Running 引擎
3. Router → 选择目标引擎
4. Gateway → HTTP 转发到 Docker vLLM
5. vLLM → 执行推理
6. Gateway → 返回响应给 Client
```

## 技术选型

| 层次 | 技术 | 说明 |
|------|------|------|
| 控制平面 | Rust | 高性能、类型安全 |
| Web 框架 | Axum | 异步、模块化 |
| 元数据存储 | etcd | 分布式一致性 |
| 推理引擎 | vLLM | OpenAI 兼容 API |
| 容器化 | Docker | GPU 支持 (nvidia-docker) |
| 序列化 | serde/JSON | 配置和数据交换 |

## 设计原则

1. **声明式调度** - Scheduler 写入意图，Agent 异步执行
2. **非侵入式引擎** - vLLM 在 Docker 中独立运行
3. **统一硬件抽象** - NVML FFI 检测 GPU 设备
4. **水平扩展** - 多 Worker Agent 分布式部署

## Crate 依赖关系

```
deepinfer-cli
    ├── deepinfer-gateway
    │   ├── deepinfer-router
    │   ├── deepinfer-meta
    │   └── deepinfer-common
    ├── deepinfer-agent
    │   ├── deepinfer-meta
    │   ├── deepinfer-device
    │   └── deepinfer-common
    └── deepinfer-scheduler
        └── deepinfer-common
```
