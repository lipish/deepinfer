use crate::state::ClusterSnapshot;
use deepinfer_meta::MetaStore;
use deepinfer_device::NodeDeviceInfo;
use std::sync::Arc;
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
        
        // Watch for changes in /nodes/* prefix
        match self.store.watch("/nodes/").await {
            Ok(mut rx) => {
                loop {
                    match rx.recv().await {
                        Ok(value) => {
                            // Parse the value as NodeDeviceInfo
                            if let Ok(info) = serde_json::from_slice::<NodeDeviceInfo>(&value) {
                                info!("Updating node in snapshot: {:?}", info);
                                // The node_id should be part of NodeDeviceInfo
                                // For now, we'll use a placeholder - this should be improved
                                // to extract node_id from the key in the watch event
                                self.snapshot.update_node("unknown".to_string(), info);
                            }
                        }
                        Err(e) => {
                            warn!("Watch channel error: {}", e);
                            break;
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
