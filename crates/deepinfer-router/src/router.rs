use crate::endpoint::EndpointManager;
use deepinfer_common::ExecutionContext;
use anyhow::Result;
use std::sync::Arc;
use tracing::debug;

/// KV-cache aware router with session affinity
pub struct KvAwareRouter {
    endpoint_mgr: Arc<EndpointManager>,
}

impl KvAwareRouter {
    pub fn new(endpoint_mgr: Arc<EndpointManager>) -> Self {
        Self { endpoint_mgr }
    }
    
    /// Route a request to an appropriate endpoint
    pub async fn route(
        &self,
        model_name: &str,
        ctx: &ExecutionContext,
    ) -> Result<String> {
        // Session affinity: if session_id is present, try to route to same endpoint
        if let Some(session_id) = &ctx.session_id {
            if let Some(endpoint) = self.endpoint_mgr.get_by_session(session_id).await {
                debug!("Routing to endpoint via session affinity: {}", endpoint);
                return Ok(endpoint);
            }
        }
        
        // Otherwise, use least-connections
        self.endpoint_mgr.get_least_loaded(model_name).await
    }
}
