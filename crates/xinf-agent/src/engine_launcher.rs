use xinf_common::{EngineConfig, RunningEngine, EngineStatus};
use anyhow::Result;
use std::process::{Child, Command};
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
    pub async fn launch(&self, config: &EngineConfig) -> Result<RunningEngine> {
        info!("Launching engine for model: {}", config.model_name);
        
        let engine_id = Uuid::new_v4();
        
        // Build command to start Python engine shim
        let mut cmd = Command::new("python3");
        cmd.arg("-m")
            .arg("xinf_engine.server")
            .arg("--engine")
            .arg(&config.engine_type)
            .arg("--model")
            .arg(&config.model_name)
            .arg("--port")
            .arg("50051"); // TODO: Dynamic port allocation
        
        if config.tensor_parallel_size > 1 {
            cmd.arg("--tensor-parallel-size")
                .arg(config.tensor_parallel_size.to_string());
        }
        
        // TODO: Add more config arguments
        
        let child = cmd.spawn()?;
        let pid = child.id();
        
        self.processes.lock().unwrap().insert(engine_id, child);
        
        Ok(RunningEngine {
            engine_id,
            config: config.clone(),
            status: EngineStatus::Starting,
            endpoint: None,
            node_id: String::new(), // Will be set by caller
            device_indices: vec![],
            pid: Some(pid),
            started_at: Some(chrono::Utc::now()),
            error_message: None,
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
