use serde::{Deserialize, Serialize};

/// Accelerator type enumeration supporting multiple vendors
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorType {
    Cuda,
    Rocm,
    Xpu,
    Mps,
    AscendNpu,
    CambriconMlu,
    EnflameGcu,
    MooreThreadsMusa,
    HygonDcu,
    KunlunxinXpu,
    Cpu,
}

impl std::fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AcceleratorType::Cuda => "cuda",
            AcceleratorType::Rocm => "rocm",
            AcceleratorType::Xpu => "xpu",
            AcceleratorType::Mps => "mps",
            AcceleratorType::AscendNpu => "ascend_npu",
            AcceleratorType::CambriconMlu => "cambricon_mlu",
            AcceleratorType::EnflameGcu => "enflame_gcu",
            AcceleratorType::MooreThreadsMusa => "moore_threads_musa",
            AcceleratorType::HygonDcu => "hygon_dcu",
            AcceleratorType::KunlunxinXpu => "kunlunxin_xpu",
            AcceleratorType::Cpu => "cpu",
        };
        write!(f, "{}", s)
    }
}

/// Supported data types for computation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
    FP8E4M3,
    FP8E5M2,
    FP4,
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub index: u32,
    pub accelerator_type: AcceleratorType,
    pub name: String,
    pub compute_capability: Option<(u32, u32)>,
    pub total_memory_mb: u64,
    pub free_memory_mb: u64,
    pub utilization_percent: f32,
    pub temperature_celsius: Option<f32>,
    pub power_usage_watts: Option<f32>,
    pub supported_dtypes: Vec<DataType>,
}

/// Node device information (all devices on a node)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDeviceInfo {
    pub node_id: String,
    pub hostname: String,
    pub devices: Vec<DeviceInfo>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl DeviceInfo {
    /// Check if device supports Blackwell architecture (SM 10.0 / compute capability 12.0)
    pub fn is_blackwell(&self) -> bool {
        self.accelerator_type == AcceleratorType::Cuda
            && self.compute_capability.map_or(false, |(major, _)| major >= 10)
    }

    /// Check if device is RTX 5090 or similar Blackwell-based GPU
    pub fn is_rtx_5090(&self) -> bool {
        self.is_blackwell() && self.name.to_lowercase().contains("5090")
    }
}
