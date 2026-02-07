use deepinfer_device::DeviceDiscovery;
use deepinfer_meta::MetaStore;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{info, error};

/// Heartbeat sender reports device status to MetaStore
pub struct HeartbeatSender {
    node_id: String,
    store: Arc<dyn MetaStore>,
}

impl HeartbeatSender {
    pub fn new(node_id: String, store: Arc<dyn MetaStore>) -> Self {
        Self { node_id, store }
    }
    
    pub async fn run(&self) {
        info!("Starting heartbeat for node: {}", self.node_id);
        
        let mut ticker = interval(Duration::from_secs(10));
        
        loop {
            ticker.tick().await;
            
            if let Err(e) = self.send_heartbeat().await {
                error!("Heartbeat error: {}", e);
            }
        }
    }
    
    async fn send_heartbeat(&self) -> anyhow::Result<()> {
        let device_info = DeviceDiscovery::discover()?;
        
        let key = format!("/nodes/{}", self.node_id);
        let value = serde_json::to_vec(&device_info)?;
        
        self.store.put(&key, value).await?;
        
        Ok(())
    }
}
