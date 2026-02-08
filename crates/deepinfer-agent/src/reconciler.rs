use crate::engine_launcher::EngineLauncher;
use deepinfer_meta::MetaStore;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::info;

/// Reconciler ensures desired state matches actual state
pub struct Reconciler {
    node_id: String,
    store: Arc<dyn MetaStore>,
    launcher: Arc<EngineLauncher>,
}

impl Reconciler {
    pub fn new(
        node_id: String,
        store: Arc<dyn MetaStore>,
        launcher: Arc<EngineLauncher>,
    ) -> Self {
        Self {
            node_id,
            store,
            launcher,
        }
    }
    
    /// Main reconciliation loop with watch support
    pub async fn run(&self) {
        info!("Starting reconciler for node: {}", self.node_id);
        
        // Start watch on /engines prefix
        let mut watch_rx = match self.store.watch("/engines").await {
            Ok(rx) => rx,
            Err(e) => {
                tracing::error!("Failed to start watch: {}", e);
                return;
            }
        };
        
        // Spawn periodic reconciliation task
        let store = self.store.clone();
        let launcher = self.launcher.clone();
        let node_id = self.node_id.clone();
        
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(30));
            loop {
                ticker.tick().await;
                if let Err(e) = Self::reconcile_once(&node_id, &store, &launcher).await {
                    tracing::error!("Periodic reconciliation error: {}", e);
                }
            }
        });
        
        // Watch-based reconciliation
        loop {
            match watch_rx.recv().await {
                Ok(value) => {
                    tracing::info!("MetaStore change detected, reconciling...");
                    if let Err(e) = Self::reconcile_once(&self.node_id, &self.store, &self.launcher).await {
                        tracing::error!("Watch reconciliation error: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("Watch error: {}", e);
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    
                    // Try to re-establish watch
                    match self.store.watch("/engines").await {
                        Ok(rx) => watch_rx = rx,
                        Err(e) => {
                            tracing::error!("Failed to re-establish watch: {}", e);
                            return;
                        }
                    }
                }
            }
        }
    }
    
    async fn reconcile_once(
        node_id: &str,
        store: &Arc<dyn MetaStore>,
        launcher: &Arc<EngineLauncher>,
    ) -> anyhow::Result<()> {
        use deepinfer_common::types::{RunningEngine, EngineStatus, RestartPolicy};
        
        // List all engines from MetaStore
        let engines = store.list("/engines").await?;
        
        for (key, value) in engines {
            let engine: RunningEngine = serde_json::from_slice(&value)?;
            
            // Process pending engines with no assigned node
            if engine.status == EngineStatus::Pending && engine.node_id.is_empty() {
                tracing::info!("Found pending engine: {} for model {}", engine.engine_id, engine.config.model_name);
                
                // Assign to this node and start the engine
                let mut updated_engine = engine.clone();
                updated_engine.node_id = node_id.to_string();
                updated_engine.status = EngineStatus::Starting;
                
                // Update MetaStore
                let updated_value = serde_json::to_vec(&updated_engine)?;
                store.put(&key, updated_value).await?;
                
                // Launch the engine
                let launcher = launcher.clone();
                let store = store.clone();
                let key = key.clone();
                let engine = updated_engine.clone();
                
                tokio::spawn(async move {
                    match launcher.launch_engine(&engine).await {
                        Ok(endpoint) => {
                            tracing::info!("Engine {} started successfully at {}", engine.engine_id, endpoint.address);
                            
                            // Update status to Running
                            let mut final_engine = engine.clone();
                            final_engine.status = EngineStatus::Running;
                            final_engine.endpoint = Some(endpoint);
                            final_engine.started_at = Some(chrono::Utc::now());
                            final_engine.error_message = None;
                            
                            let final_value = serde_json::to_vec(&final_engine).unwrap();
                            let _ = store.put(&key, final_value).await;
                        }
                        Err(e) => {
                            tracing::error!("Failed to launch engine {}: {}", engine.engine_id, e);
                            
                            // Update status to Failed
                            let mut failed_engine = engine.clone();
                            failed_engine.status = EngineStatus::Failed;
                            failed_engine.error_message = Some(e.to_string());
                            
                            let failed_value = serde_json::to_vec(&failed_engine).unwrap();
                            let _ = store.put(&key, failed_value).await;
                        }
                    }
                });
            }
            
            // Handle failed engines - attempt restart based on policy
            if engine.status == EngineStatus::Failed && engine.node_id == node_id {
                let should_restart = match engine.config.restart_policy {
                    RestartPolicy::Always => true,
                    RestartPolicy::OnFailure => true,
                    RestartPolicy::Never => false,
                };
                
                let max_restarts = engine.config.max_restarts;
                let within_limit = max_restarts == 0 || engine.restart_count < max_restarts;
                
                if should_restart && within_limit {
                    tracing::info!(
                        "Restarting failed engine {} (attempt {}/{})", 
                        engine.engine_id, 
                        engine.restart_count + 1,
                        if max_restarts == 0 { "unlimited".to_string() } else { max_restarts.to_string() }
                    );
                    
                    // Reset engine status to pending for restart
                    let mut restart_engine = engine.clone();
                    restart_engine.status = EngineStatus::Pending;
                    restart_engine.node_id = String::new(); // Clear node assignment
                    restart_engine.endpoint = None;
                    restart_engine.error_message = None;
                    restart_engine.restart_count += 1;
                    
                    let restart_value = serde_json::to_vec(&restart_engine)?;
                    store.put(&key, restart_value).await?;
                } else if !within_limit {
                    tracing::warn!(
                        "Engine {} exceeded max restart attempts ({}), not restarting",
                        engine.engine_id,
                        max_restarts
                    );
                }
            }
        }
        
        Ok(())
    }
}
