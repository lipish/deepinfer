use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use deepinfer_common::types::{EngineConfig, RunningEngine, EngineStatus, EngineBackend};
use deepinfer_meta::MetaStore;

#[derive(Debug, Deserialize)]
pub struct LaunchRequest {
    pub model: String,
    pub engine: Option<String>,
    pub device: Option<String>,
    #[serde(default)]
    pub tensor_parallel_size: Option<u32>,
    #[serde(default)]
    pub gpu_memory_utilization: Option<f32>,
    /// Backend type: "native" or "docker"
    #[serde(default)]
    pub backend: Option<String>,
    /// Docker image (required if backend is "docker")
    pub docker_image: Option<String>,
    /// Model path (for volume mount in docker mode)
    pub model_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LaunchResponse {
    pub status: String,
    pub message: String,
    pub model: String,
}

pub async fn launch_model(
    State((store, _router)): State<(Arc<dyn MetaStore>, Arc<deepinfer_router::KvAwareRouter>)>,
    Json(req): Json<LaunchRequest>,
) -> impl IntoResponse {
    tracing::info!("Received launch request for model: {}", req.model);
    
    // Parse device specification
    let device_indices = if let Some(ref device) = req.device {
        parse_device_spec(device)
    } else {
        vec![0] // Default to device 0
    };
    
    // Determine backend type
    let backend = match req.backend.as_deref() {
        Some("docker") => EngineBackend::Docker,
        _ => EngineBackend::Native,
    };
    
    // Build engine config
    let config = EngineConfig {
        engine_type: req.engine.unwrap_or_else(|| "vllm".to_string()),
        model_name: req.model.clone(),
        model_path: req.model_path.clone(),
        tensor_parallel_size: req.tensor_parallel_size.unwrap_or(1),
        pipeline_parallel_size: 1,
        max_num_seqs: 256,
        max_model_len: None,
        dtype: Some("auto".to_string()),
        quantization: None,
        kv_cache_dtype: Some("auto".to_string()),
        gpu_memory_utilization: req.gpu_memory_utilization.unwrap_or(0.9),
        enforce_eager: false,
        additional_args: std::collections::HashMap::new(),
        backend,
        docker_image: req.docker_image.clone(),
        container_name: None,
    };
    
    // Create a running engine entry in MetaStore
    let engine_id = uuid::Uuid::new_v4();
    let running_engine = RunningEngine {
        engine_id,
        config: config.clone(),
        status: EngineStatus::Pending,
        endpoint: None,
        node_id: "".to_string(), // Will be assigned by scheduler
        device_indices: device_indices.clone(),
        pid: None,
        container_id: None,
        started_at: None,
        error_message: None,
    };
    
    // Store in MetaStore under /engines/{engine_id}
    let key = format!("/engines/{}", engine_id);
    let value = serde_json::to_vec(&running_engine).unwrap();
    
    if let Err(e) = store.put(&key, value).await {
        tracing::error!("Failed to store engine in MetaStore: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(LaunchResponse {
                status: "error".to_string(),
                message: format!("Failed to store engine: {}", e),
                model: req.model,
            })
        );
    }
    
    tracing::info!("Engine {} created for model {}", engine_id, req.model);
    
    let response = LaunchResponse {
        status: "accepted".to_string(),
        message: format!("Model launch request accepted for {} (engine_id: {})", req.model, engine_id),
        model: req.model,
    };
    
    (StatusCode::ACCEPTED, Json(response))
}

fn parse_device_spec(device: &str) -> Vec<u32> {
    // Parse device specifications like "cuda:0", "cuda:0,1", "0", "0,1"
    let device_str = device.strip_prefix("cuda:").unwrap_or(device);
    
    device_str
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect()
}
