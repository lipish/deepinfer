/// NVIDIA GPU backend using NVML FFI
use crate::types::{AcceleratorType, DataType, DeviceInfo, NodeDeviceInfo};
use anyhow::{anyhow, Result};
use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;
use std::ffi::{c_char, c_int, c_uint, c_ulonglong};
use tracing::{debug, warn};

// NVML return codes
const NVML_SUCCESS: c_int = 0;

// NVML types
type NvmlDevice = *mut std::ffi::c_void;
type NvmlReturn = c_int;

// NVML function types
type NvmlInit = unsafe extern "C" fn() -> NvmlReturn;
type NvmlShutdown = unsafe extern "C" fn() -> NvmlReturn;
type NvmlDeviceGetCount = unsafe extern "C" fn(*mut c_uint) -> NvmlReturn;
type NvmlDeviceGetHandleByIndex = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn;
type NvmlDeviceGetName = unsafe extern "C" fn(NvmlDevice, *mut c_char, c_uint) -> NvmlReturn;
type NvmlDeviceGetMemoryInfo = unsafe extern "C" fn(NvmlDevice, *mut MemoryInfo) -> NvmlReturn;
type NvmlDeviceGetUtilizationRates = unsafe extern "C" fn(NvmlDevice, *mut Utilization) -> NvmlReturn;
type NvmlDeviceGetTemperature = unsafe extern "C" fn(NvmlDevice, c_int, *mut c_uint) -> NvmlReturn;
type NvmlDeviceGetPowerUsage = unsafe extern "C" fn(NvmlDevice, *mut c_uint) -> NvmlReturn;
type NvmlDeviceGetCudaComputeCapability =
    unsafe extern "C" fn(NvmlDevice, *mut c_int, *mut c_int) -> NvmlReturn;

#[repr(C)]
struct MemoryInfo {
    total: c_ulonglong,
    free: c_ulonglong,
    used: c_ulonglong,
}

#[repr(C)]
struct Utilization {
    gpu: c_uint,
    memory: c_uint,
}

static NVML_LIB: OnceCell<Option<Library>> = OnceCell::new();

/// Try to dynamically load NVML library
fn load_nvml() -> Result<&'static Library> {
    NVML_LIB.get_or_init(|| {
        // Try common library names
        for lib_name in &["libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"] {
            if let Ok(lib) = unsafe { Library::new(lib_name) } {
                debug!("Loaded NVML library: {}", lib_name);
                return Some(lib);
            }
        }
        warn!("Failed to load NVML library");
        None
    }).as_ref().ok_or_else(|| anyhow!("NVML library not available"))
}

/// Discover NVIDIA devices using NVML
pub fn discover_nvidia_devices() -> Result<NodeDeviceInfo> {
    let lib = load_nvml()?;
    
    unsafe {
        let nvml_init: Symbol<NvmlInit> = lib.get(b"nvmlInit_v2")?;
        let nvml_shutdown: Symbol<NvmlShutdown> = lib.get(b"nvmlShutdown")?;
        let nvml_device_get_count: Symbol<NvmlDeviceGetCount> = lib.get(b"nvmlDeviceGetCount_v2")?;
        let nvml_device_get_handle: Symbol<NvmlDeviceGetHandleByIndex> = 
            lib.get(b"nvmlDeviceGetHandleByIndex_v2")?;
        let nvml_device_get_name: Symbol<NvmlDeviceGetName> = lib.get(b"nvmlDeviceGetName")?;
        let nvml_device_get_memory: Symbol<NvmlDeviceGetMemoryInfo> = 
            lib.get(b"nvmlDeviceGetMemoryInfo")?;
        let nvml_device_get_util: Symbol<NvmlDeviceGetUtilizationRates> = 
            lib.get(b"nvmlDeviceGetUtilizationRates")?;
        let nvml_device_get_temp: Symbol<NvmlDeviceGetTemperature> = 
            lib.get(b"nvmlDeviceGetTemperature")?;
        let nvml_device_get_power: Symbol<NvmlDeviceGetPowerUsage> = 
            lib.get(b"nvmlDeviceGetPowerUsage")?;
        let nvml_device_get_cc: Symbol<NvmlDeviceGetCudaComputeCapability> = 
            lib.get(b"nvmlDeviceGetCudaComputeCapability")?;

        // Initialize NVML
        let ret = nvml_init();
        if ret != NVML_SUCCESS {
            return Err(anyhow!("nvmlInit failed with code {}", ret));
        }

        let mut device_count: c_uint = 0;
        let ret = nvml_device_get_count(&mut device_count);
        if ret != NVML_SUCCESS {
            nvml_shutdown();
            return Err(anyhow!("nvmlDeviceGetCount failed with code {}", ret));
        }

        let mut devices = Vec::new();
        
        for i in 0..device_count {
            let mut handle: NvmlDevice = std::ptr::null_mut();
            if nvml_device_get_handle(i, &mut handle) != NVML_SUCCESS {
                continue;
            }

            // Get device name
            let mut name_buf = vec![0u8; 256];
            let _ = nvml_device_get_name(handle, name_buf.as_mut_ptr() as *mut c_char, 256);
            let name = String::from_utf8_lossy(&name_buf)
                .trim_end_matches('\0')
                .to_string();

            // Get memory info
            let mut mem_info = MemoryInfo {
                total: 0,
                free: 0,
                used: 0,
            };
            let _ = nvml_device_get_memory(handle, &mut mem_info);

            // Get utilization
            let mut util = Utilization { gpu: 0, memory: 0 };
            let _ = nvml_device_get_util(handle, &mut util);

            // Get temperature
            let mut temp: c_uint = 0;
            let has_temp = nvml_device_get_temp(handle, 0, &mut temp) == NVML_SUCCESS;

            // Get power
            let mut power: c_uint = 0;
            let has_power = nvml_device_get_power(handle, &mut power) == NVML_SUCCESS;

            // Get compute capability
            let mut cc_major: c_int = 0;
            let mut cc_minor: c_int = 0;
            let has_cc = nvml_device_get_cc(handle, &mut cc_major, &mut cc_minor) == NVML_SUCCESS;

            // Determine supported data types based on compute capability
            let supported_dtypes = if has_cc {
                get_supported_dtypes(cc_major as u32, cc_minor as u32)
            } else {
                vec![DataType::FP32, DataType::FP16, DataType::INT8]
            };

            devices.push(DeviceInfo {
                index: i,
                accelerator_type: AcceleratorType::Cuda,
                name,
                compute_capability: if has_cc {
                    Some((cc_major as u32, cc_minor as u32))
                } else {
                    None
                },
                total_memory_mb: (mem_info.total / 1024 / 1024) as u64,
                free_memory_mb: (mem_info.free / 1024 / 1024) as u64,
                utilization_percent: util.gpu as f32,
                temperature_celsius: if has_temp { Some(temp as f32) } else { None },
                power_usage_watts: if has_power { Some(power as f32 / 1000.0) } else { None },
                supported_dtypes,
            });
        }

        nvml_shutdown();

        Ok(NodeDeviceInfo {
            node_id: gethostname(),
            hostname: gethostname(),
            devices,
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Get supported data types based on compute capability
fn get_supported_dtypes(major: u32, _minor: u32) -> Vec<DataType> {
    let mut dtypes = vec![DataType::FP32, DataType::FP16, DataType::INT8];
    
    // BF16 support (Ampere and later, SM 8.0+)
    if major >= 8 {
        dtypes.push(DataType::BF16);
    }
    
    // FP8 support (Hopper and later, SM 9.0+)
    if major >= 9 {
        dtypes.push(DataType::FP8E4M3);
        dtypes.push(DataType::FP8E5M2);
    }
    
    // Enhanced FP8 and FP4 support (Blackwell, SM 10.0+)
    if major >= 10 {
        dtypes.push(DataType::FP4);
        dtypes.push(DataType::INT4);
    }
    
    dtypes
}

/// Get hostname
fn gethostname() -> String {
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string())
}
