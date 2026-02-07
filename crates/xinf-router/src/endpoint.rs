use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use anyhow::{anyhow, Result};

#[derive(Clone)]
pub struct EndpointInfo {
    pub address: String,
    pub model_name: String,
    pub active_connections: Arc<AtomicU64>,
}

/// Endpoint manager
pub struct EndpointManager {
    endpoints: Arc<DashMap<String, Vec<EndpointInfo>>>,
    session_map: Arc<DashMap<String, String>>,
}

impl EndpointManager {
    pub fn new() -> Self {
        Self {
            endpoints: Arc::new(DashMap::new()),
            session_map: Arc::new(DashMap::new()),
        }
    }
    
    pub fn register_endpoint(&self, model_name: String, address: String) {
        let info = EndpointInfo {
            address: address.clone(),
            model_name: model_name.clone(),
            active_connections: Arc::new(AtomicU64::new(0)),
        };
        
        self.endpoints.entry(model_name).or_insert_with(Vec::new).push(info);
    }
    
    pub async fn get_least_loaded(&self, model_name: &str) -> Result<String> {
        let endpoints = self.endpoints.get(model_name)
            .ok_or_else(|| anyhow!("No endpoints for model: {}", model_name))?;
        
        if endpoints.is_empty() {
            return Err(anyhow!("No endpoints available"));
        }
        
        // Find endpoint with least connections
        let best = endpoints.iter()
            .min_by_key(|e| e.active_connections.load(Ordering::Relaxed))
            .unwrap();
        
        Ok(best.address.clone())
    }
    
    pub async fn get_by_session(&self, session_id: &str) -> Option<String> {
        self.session_map.get(session_id).map(|v| v.clone())
    }
    
    pub fn track_session(&self, session_id: String, endpoint: String) {
        self.session_map.insert(session_id, endpoint);
    }
}

impl Default for EndpointManager {
    fn default() -> Self {
        Self::new()
    }
}
