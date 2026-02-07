use crate::backends::{nvidia, fallback};
use crate::types::NodeDeviceInfo;
use tracing::{info, warn};

/// Device discovery interface
pub struct DeviceDiscovery;

impl DeviceDiscovery {
    /// Discover all devices on the current node
    pub fn discover() -> anyhow::Result<NodeDeviceInfo> {
        info!("Starting device discovery");
        
        // Try NVIDIA first
        match nvidia::discover_nvidia_devices() {
            Ok(mut node_info) if !node_info.devices.is_empty() => {
                info!("Discovered {} NVIDIA GPU(s)", node_info.devices.len());
                
                // Check for Blackwell/RTX 5090
                for device in &node_info.devices {
                    if device.is_blackwell() {
                        info!(
                            "Detected Blackwell architecture GPU: {} (SM {}.{})",
                            device.name,
                            device.compute_capability.unwrap().0,
                            device.compute_capability.unwrap().1
                        );
                    }
                    if device.is_rtx_5090() {
                        info!("Detected RTX 5090 GPU!");
                    }
                }
                
                node_info.timestamp = chrono::Utc::now();
                return Ok(node_info);
            }
            Ok(_) => {
                warn!("NVIDIA backend returned no devices");
            }
            Err(e) => {
                warn!("NVIDIA device discovery failed: {}", e);
            }
        }

        // TODO: Try other backends (ROCm, XPU, etc.)
        
        // Fallback to CPU
        info!("Falling back to CPU-only mode");
        fallback::discover_cpu_fallback()
    }
}
