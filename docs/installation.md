# deepinfer 安装配置指南

## 系统要求

| 组件 | 最低要求 |
|------|----------|
| 操作系统 | Ubuntu 20.04+ / CentOS 8+ |
| Rust | 1.70+ |
| Docker | 20.10+ |
| nvidia-docker | 2.0+ |
| CUDA | 12.0+ |
| etcd | 3.5+ |

## 安装步骤

### 1. 安装 Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustc --version  # 验证安装
```

### 2. 安装 Docker 和 nvidia-docker

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装 nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 3. 安装 etcd

```bash
# 方式一：apt 安装
sudo apt-get install -y etcd

# 方式二：手动下载
ETCD_VER=v3.5.12
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz
tar xzf etcd-${ETCD_VER}-linux-amd64.tar.gz
sudo cp etcd-${ETCD_VER}-linux-amd64/etcd* /usr/local/bin/

# 验证
etcd --version
```

### 4. 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  libssl-dev \
  pkg-config \
  protobuf-compiler
```

### 5. 编译 deepinfer

```bash
git clone https://github.com/lipish/deepinfer.git
cd deepinfer
cargo build --release
```

### 6. 下载 vLLM Docker 镜像

```bash
docker pull vllm/vllm-openai:v0.11.0
```

### 7. 下载模型

```bash
# 使用 modelscope（国内推荐）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"

# 或使用 huggingface
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```

## 配置

### 主配置文件

`configs/default.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8082
max_connections = 1000

[scheduler]
reconcile_interval_secs = 5
placement_strategy = "idle_first"

[agent]
heartbeat_interval_secs = 10
reconcile_interval_secs = 5
health_check_interval_secs = 30

[storage]
backend = "etcd"
etcd_endpoints = ["127.0.0.1:2379"]
```

### 配置项说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `server.host` | 0.0.0.0 | Gateway 监听地址 |
| `server.port` | 8082 | Gateway 监听端口 |
| `storage.backend` | etcd | 存储后端类型 |
| `storage.etcd_endpoints` | ["127.0.0.1:2379"] | etcd 地址列表 |

## 启动服务

### 1. 启动 etcd

```bash
# 使用提供的脚本
./start-etcd.sh

# 或手动启动
etcd \
  --name deepinfer-etcd \
  --data-dir ./data/etcd \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379

# 验证
etcdctl endpoint health
```

### 2. 启动 Gateway

```bash
RUST_LOG=info ./target/release/deepinfer serve -c configs/default.toml
```

### 3. 启动 Worker Agent

```bash
RUST_LOG=info ./target/release/deepinfer worker -n node-1
```

## 部署模型

### API 方式

```bash
curl -X POST http://localhost:8082/v1/deployments \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "backend": "docker",
    "docker_image": "vllm/vllm-openai:v0.11.0",
    "model_path": "/home/user/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct",
    "device": "0",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.8
  }'
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| model | 是 | 模型名称（用于 API 调用） |
| backend | 是 | 后端类型，固定为 "docker" |
| docker_image | 是 | Docker 镜像 |
| model_path | 是 | 模型文件路径 |
| device | 否 | GPU 设备，如 "0" 或 "0,1" |
| tensor_parallel_size | 否 | 张量并行数，默认 1 |
| gpu_memory_utilization | 否 | GPU 显存利用率，默认 0.9 |

## 多 GPU 配置

### 双卡张量并行

```bash
curl -X POST http://localhost:8082/v1/deployments \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "backend": "docker",
    "docker_image": "vllm/vllm-openai:v0.11.0",
    "model_path": "/path/to/model",
    "device": "0,1",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.8
  }'
```

## 网络配置

### 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| Gateway | 8082 | HTTP API |
| etcd | 2379 | 客户端端口 |
| etcd | 2380 | 集群通信端口 |
| vLLM | 8000+ | 推理服务（容器内） |

### 防火墙配置

```bash
# 开放必要端口
sudo ufw allow 8082/tcp  # Gateway
sudo ufw allow 2379/tcp  # etcd
```

## 日志配置

通过环境变量 `RUST_LOG` 控制日志级别：

```bash
# 详细日志
RUST_LOG=debug ./target/release/deepinfer serve

# 仅错误日志
RUST_LOG=error ./target/release/deepinfer serve

# 指定模块日志级别
RUST_LOG=deepinfer_agent=debug,deepinfer_gateway=info ./target/release/deepinfer serve
```

## 常见问题

### 1. etcd 连接失败

```
Error: failed to connect to etcd
```

**解决：** 确保 etcd 正在运行，检查 `etcd_endpoints` 配置。

### 2. Docker GPU 不可用

```
Error: could not select device driver
```

**解决：** 安装 nvidia-docker2 并重启 Docker 服务。

### 3. 模型路径错误

```
Error: model path not found
```

**解决：** 确保 `model_path` 是模型文件的绝对路径。

### 4. 端口被占用

```
Error: Address already in use
```

**解决：** 修改 `configs/default.toml` 中的端口配置，或停止占用端口的进程。
