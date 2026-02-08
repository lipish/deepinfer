use deepinfer_common::types::{RunningEngine, EndpointInfo, EngineBackend};
use anyhow::{Result, anyhow};
use std::process::{Child, Command, Stdio};
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::{info, error};
use uuid::Uuid;

/// Represents a running engine process or container
enum EngineHandle {
    Process(Child),
    Container(String), // container ID
}

/// Engine launcher manages engine processes and containers
pub struct EngineLauncher {
    handles: Mutex<HashMap<Uuid, EngineHandle>>,
}

impl EngineLauncher {
    pub fn new() -> Self {
        Self {
            handles: Mutex::new(HashMap::new()),
        }
    }
    
    /// Launch an engine (either native process or docker container)
    pub async fn launch_engine(&self, engine: &RunningEngine) -> Result<EndpointInfo> {
        match engine.config.backend {
            EngineBackend::Docker => self.launch_docker_engine(engine).await,
            EngineBackend::Native => self.launch_native_engine(engine).await,
        }
    }
    
    /// Launch engine as Docker container
    async fn launch_docker_engine(&self, engine: &RunningEngine) -> Result<EndpointInfo> {
        let docker_image = engine.config.docker_image.as_ref()
            .ok_or_else(|| anyhow!("docker_image is required for docker backend"))?;
        
        info!("Launching Docker engine {} for model: {} using image: {}", 
              engine.engine_id, engine.config.model_name, docker_image);
        
        // Allocate port
        let port = 8000 + engine.device_indices.first().unwrap_or(&0);
        
        // Build device string for --gpus
        let device_str = if !engine.device_indices.is_empty() {
            engine.device_indices.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        } else {
            "0".to_string()
        };
        
        // Container name
        let container_name = engine.config.container_name.clone()
            .unwrap_or_else(|| format!("deepinfer-engine-{}", &engine.engine_id.to_string()[..8]));
        
        // Determine model path (for volume mount)
        let model_path = engine.config.model_path.clone()
            .unwrap_or_else(|| engine.config.model_name.clone());
        
        // Build docker command
        let mut cmd = Command::new("docker");
        cmd.arg("run")
            .arg("-d")  // detached
            .arg("--name").arg(&container_name)
            .arg("--gpus").arg(format!("\"device={}\"", device_str))
            .arg("-v").arg(format!("{}:/model", model_path))
            .arg("-p").arg(format!("{}:8000", port))
            .arg(docker_image)
            .arg("--model").arg("/model")
            .arg("--tensor-parallel-size").arg(engine.config.tensor_parallel_size.to_string())
            .arg("--gpu-memory-utilization").arg(engine.config.gpu_memory_utilization.to_string());
        
        // Optional args
        if let Some(max_len) = engine.config.max_model_len {
            cmd.arg("--max-model-len").arg(max_len.to_string());
        }
        
        if let Some(dtype) = &engine.config.dtype {
            cmd.arg("--dtype").arg(dtype);
        }
        
        info!("Docker command: {:?}", cmd);
        
        let output = cmd.output()
            .map_err(|e| anyhow!("Failed to run docker: {}", e))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Docker run failed: {}", stderr));
        }
        
        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        info!("Docker container started: {} ({})", container_name, &container_id[..12]);
        
        self.handles.lock().unwrap().insert(engine.engine_id, EngineHandle::Container(container_id.clone()));
        
        // Wait for container to be ready (vLLM may take a while to initialize)
        self.wait_for_docker_ready(port as u16, 180).await?;
        
        Ok(EndpointInfo {
            address: "localhost".to_string(),
            port: port as u16,
            protocol: "http".to_string(),  // Docker vLLM uses HTTP (OpenAI-compatible)
        })
    }
    
    /// Wait for Docker vLLM to be ready
    async fn wait_for_docker_ready(&self, port: u16, timeout_secs: u64) -> Result<()> {
        let client = reqwest::Client::new();
        // Use /v1/models endpoint as health check (more reliable for vLLM)
        let url = format!("http://localhost:{}/v1/models", port);
        
        let start = std::time::Instant::now();
        loop {
            if start.elapsed().as_secs() > timeout_secs {
                return Err(anyhow!("Timeout waiting for Docker engine to be ready after {}s", timeout_secs));
            }
            
            match client.get(&url).timeout(std::time::Duration::from_secs(5)).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("Docker engine ready on port {}", port);
                    return Ok(());
                }
                Ok(resp) => {
                    info!("Health check returned status: {}, retrying...", resp.status());
                    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                }
                Err(e) => {
                    info!("Health check failed: {}, retrying...", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                }
            }
        }
    }
    
    /// Launch engine as native Python process
    async fn launch_native_engine(&self, engine: &RunningEngine) -> Result<EndpointInfo> {
        info!("Launching native engine {} for model: {}", engine.engine_id, engine.config.model_name);
        
        // Allocate port dynamically (simple approach: use 50051 + device_index)
        let port = 50051 + engine.device_indices.first().unwrap_or(&0);
        
        // Determine device string
        let device_str = if !engine.device_indices.is_empty() {
            format!("cuda:{}", engine.device_indices[0])
        } else {
            "cuda:0".to_string()
        };
        
        // Get project root
        let project_root = std::env::current_dir()?;
        let python_path = project_root.join("python");
        
        // Build command to start Python engine shim
        let mut cmd = Command::new("python3");
        cmd.current_dir(&project_root)
            .env("PYTHONPATH", python_path.to_str().unwrap())
            .env("VLLM_USE_MODELSCOPE", "True")
            .arg("python/deepinfer_engine/server.py")
            .arg("--engine")
            .arg(&engine.config.engine_type)
            .arg("--model")
            .arg(&engine.config.model_name)
            .arg("--port")
            .arg(port.to_string())
            .arg("--device")
            .arg(&device_str)
            .arg("--tensor-parallel-size")
            .arg(engine.config.tensor_parallel_size.to_string())
            .arg("--gpu-memory-utilization")
            .arg(engine.config.gpu_memory_utilization.to_string());
        
        if let Some(dtype) = &engine.config.dtype {
            cmd.arg("--dtype").arg(dtype);
        }
        
        if let Some(max_len) = engine.config.max_model_len {
            cmd.arg("--max-model-len").arg(max_len.to_string());
        }
        
        // Create log file for engine output
        let log_path = format!("/tmp/engine-{}.log", engine.engine_id);
        let log_file = std::fs::File::create(&log_path)
            .map_err(|e| anyhow!("Failed to create log file: {}", e))?;
        cmd.stdout(Stdio::from(log_file.try_clone().unwrap()))
            .stderr(Stdio::from(log_file));
        info!("Engine logs will be written to: {}", log_path);
        
        let child = cmd.spawn()
            .map_err(|e| anyhow!("Failed to spawn engine process: {}", e))?;
        
        let pid = child.id();
        info!("Engine {} spawned with PID {}", engine.engine_id, pid);
        
        self.handles.lock().unwrap().insert(engine.engine_id, EngineHandle::Process(child));
        
        // Wait a bit for engine to start
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        Ok(EndpointInfo {
            address: "localhost".to_string(),
            port: port as u16,
            protocol: "grpc".to_string(),
        })
    }
    
    /// Stop an engine (process or container)
    pub async fn stop(&self, engine_id: Uuid) -> Result<()> {
        info!("Stopping engine: {}", engine_id);
        
        let mut handles = self.handles.lock().unwrap();
        if let Some(handle) = handles.remove(&engine_id) {
            match handle {
                EngineHandle::Process(mut child) => {
                    match child.kill() {
                        Ok(_) => {
                            let _ = child.wait();
                            info!("Engine process stopped: {}", engine_id);
                        }
                        Err(e) => {
                            error!("Failed to kill engine process {}: {}", engine_id, e);
                        }
                    }
                }
                EngineHandle::Container(container_id) => {
                    let output = Command::new("docker")
                        .args(["stop", &container_id])
                        .output();
                    
                    match output {
                        Ok(o) if o.status.success() => {
                            info!("Docker container stopped: {}", &container_id[..12]);
                            // Remove container
                            let _ = Command::new("docker")
                                .args(["rm", &container_id])
                                .output();
                        }
                        Ok(o) => {
                            error!("Failed to stop container: {}", String::from_utf8_lossy(&o.stderr));
                        }
                        Err(e) => {
                            error!("Failed to run docker stop: {}", e);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}

impl Default for EngineLauncher {
    fn default() -> Self {
        Self::new()
    }
}
