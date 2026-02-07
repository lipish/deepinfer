use crate::state::ClusterSnapshot;
use xinf_meta::MetaStore;
use xinf_device::NodeDeviceInfo;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tracing::{info, warn};

/// Sync cluster state from MetaStore to local snapshot
pub struct StateSynchronizer {
    store: Arc<dyn MetaStore>,
    snapshot: Arc<ClusterSnapshot>,
}

impl StateSynchronizer {
    pub fn new(store: Arc<dyn MetaStore>, snapshot: Arc<ClusterSnapshot>) -> Self {
        Self { store, snapshot }
    }
    
    /// Start watching MetaStore for node updates
    pub async fn start_watch(&self) {
        info!("Starting state synchronizer");
        
        // TODO: Implement watch loop
        // Watch for changes in /nodes/* prefix
        // Parse NodeDeviceInfo from events
        // Update local snapshot
        
        match self.store.watch("/nodes/").await {
            Ok(mut stream) => {
                while let Some(event) = stream.next().await {
                    match event {
                        xinf_meta::WatchEvent::Put { key, value } => {
                            if let Ok(info) = serde_json::from_slice::<NodeDeviceInfo>(&value) {
                                let node_id = key.strip_prefix("/nodes/").unwrap_or(&key);
                                info!("Updating node in snapshot: {}", node_id);
                                self.snapshot.update_node(node_id.to_string(), info);
                            }
                        }
                        xinf_meta::WatchEvent::Delete { key } => {
                            let node_id = key.strip_prefix("/nodes/").unwrap_or(&key);
                            info!("Removing node from snapshot: {}", node_id);
                            self.snapshot.remove_node(node_id);
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to start watch: {}", e);
            }
        }
    }
}
