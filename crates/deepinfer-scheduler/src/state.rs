use dashmap::DashMap;
use deepinfer_device::NodeDeviceInfo;
use std::sync::Arc;

/// Cluster snapshot maintained in memory via DashMap
pub struct ClusterSnapshot {
    nodes: Arc<DashMap<String, NodeDeviceInfo>>,
}

impl ClusterSnapshot {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(DashMap::new()),
        }
    }
    
    pub fn update_node(&self, node_id: String, info: NodeDeviceInfo) {
        self.nodes.insert(node_id, info);
    }
    
    pub fn remove_node(&self, node_id: &str) {
        self.nodes.remove(node_id);
    }
    
    pub fn get_node(&self, node_id: &str) -> Option<NodeDeviceInfo> {
        self.nodes.get(node_id).map(|v| v.clone())
    }
    
    pub fn get_all_nodes(&self) -> Vec<(String, NodeDeviceInfo)> {
        self.nodes.iter().map(|entry| {
            (entry.key().clone(), entry.value().clone())
        }).collect()
    }
}

impl Default for ClusterSnapshot {
    fn default() -> Self {
        Self::new()
    }
}
