use crate::types::NodeDeviceInfo;
use crate::discovery::DeviceDiscovery;
use std::time::Duration;
use tokio::time;

/// Periodic device monitor
pub struct DeviceMonitor {
    interval: Duration,
}

impl DeviceMonitor {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            interval: Duration::from_secs(interval_secs),
        }
    }

    /// Start monitoring loop
    pub async fn run<F>(self, mut callback: F)
    where
        F: FnMut(NodeDeviceInfo) + Send,
    {
        let mut interval = time::interval(self.interval);
        loop {
            interval.tick().await;
            
            match DeviceDiscovery::discover() {
                Ok(info) => callback(info),
                Err(e) => {
                    tracing::error!("Device discovery failed: {}", e);
                }
            }
        }
    }
}
