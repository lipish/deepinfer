/// Engine health checker

pub struct HealthChecker;

impl HealthChecker {
    pub fn new() -> Self {
        Self
    }
    
    /// Check if engine is healthy via gRPC health check
    pub async fn check_health(&self, _endpoint: &str) -> bool {
        // TODO: Implement gRPC health check call
        true
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}
