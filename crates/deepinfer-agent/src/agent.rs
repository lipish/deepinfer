use crate::engine_launcher::EngineLauncher;
use crate::reconciler::Reconciler;
use crate::heartbeat::HeartbeatSender;
use deepinfer_meta::MetaStore;
use std::sync::Arc;
use tracing::info;

/// Worker agent main struct
pub struct WorkerAgent {
    node_id: String,
    store: Arc<dyn MetaStore>,
    launcher: Arc<EngineLauncher>,
    reconciler: Arc<Reconciler>,
    heartbeat: Arc<HeartbeatSender>,
}

impl WorkerAgent {
    pub fn new(
        node_id: String,
        store: Arc<dyn MetaStore>,
    ) -> Self {
        let launcher = Arc::new(EngineLauncher::new());
        let reconciler = Arc::new(Reconciler::new(
            node_id.clone(),
            store.clone(),
            launcher.clone(),
        ));
        let heartbeat = Arc::new(HeartbeatSender::new(
            node_id.clone(),
            store.clone(),
        ));
        
        Self {
            node_id,
            store,
            launcher,
            reconciler,
            heartbeat,
        }
    }
    
    /// Start the agent's main loop
    pub async fn run(&self) {
        info!("Starting worker agent: {}", self.node_id);
        
        // Start heartbeat loop
        let heartbeat = self.heartbeat.clone();
        tokio::spawn(async move {
            heartbeat.run().await;
        });
        
        // Start reconciler
        self.reconciler.run().await;
    }
}
