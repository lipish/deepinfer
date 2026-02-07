/// Fallback CPU-only backend
use crate::types::{AcceleratorType, DataType, DeviceInfo, NodeDeviceInfo};
use anyhow::Result;

pub fn discover_cpu_fallback() -> Result<NodeDeviceInfo> {
    let cpu_device = DeviceInfo {
        index: 0,
        accelerator_type: AcceleratorType::Cpu,
        name: "CPU".to_string(),
        compute_capability: None,
        total_memory_mb: get_total_memory_mb(),
        free_memory_mb: get_free_memory_mb(),
        utilization_percent: 0.0,
        temperature_celsius: None,
        power_usage_watts: None,
        supported_dtypes: vec![
            DataType::FP32,
            DataType::FP16,
            DataType::BF16,
            DataType::INT8,
        ],
    };

    Ok(NodeDeviceInfo {
        node_id: gethostname(),
        hostname: gethostname(),
        devices: vec![cpu_device],
        timestamp: chrono::Utc::now(),
    })
}

fn get_total_memory_mb() -> u64 {
    // Try to get from /proc/meminfo on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb_val) = kb.parse::<u64>() {
                            return kb_val / 1024; // Convert to MB
                        }
                    }
                }
            }
        }
    }
    
    // Default fallback
    8192 // 8GB default
}

fn get_free_memory_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb_val) = kb.parse::<u64>() {
                            return kb_val / 1024;
                        }
                    }
                }
            }
        }
    }
    
    get_total_memory_mb() / 2 // Assume half is free
}

fn gethostname() -> String {
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string())
}
