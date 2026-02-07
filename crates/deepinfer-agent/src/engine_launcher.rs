use deepinfer_common::types::{EngineConfig, RunningEngine, EngineStatus, EndpointInfo};
use anyhow::{Result, anyhow};
use std::process::{Child, Command, Stdio};
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::{info, error};
use uuid::Uuid;

/// Engine launcher manages engine processes
pub struct EngineLauncher {
    processes: Mutex<HashMap<Uuid, Child>>,
}

impl EngineLauncher {
    pub fn new() -> Self {
        Self {
            processes: Mutex::new(HashMap::new()),
        }
    }
    
    /// Launch an engine process
    pub async fn launch_engine(&self, engine: &RunningEngine) -> Result<EndpointInfo> {
        info!("Launching engine {} for model: {}", engine.engine_id, engine.config.model_name);
        
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
        
        self.processes.lock().unwrap().insert(engine.engine_id, child);
        
        // Wait a bit for engine to start
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        Ok(EndpointInfo {
            address: "localhost".to_string(),
            port: port as u16,
            protocol: "grpc".to_string(),
        })
    }
    
    /// Stop an engine process
    pub async fn stop(&self, engine_id: Uuid) -> Result<()> {
        info!("Stopping engine: {}", engine_id);
        
        let mut processes = self.processes.lock().unwrap();
        if let Some(mut child) = processes.remove(&engine_id) {
            match child.kill() {
                Ok(_) => {
                    let _ = child.wait();
                    info!("Engine stopped: {}", engine_id);
                }
                Err(e) => {
                    error!("Failed to kill engine {}: {}", engine_id, e);
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
