use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Node status in the cluster
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Active,
    Draining,
    Offline,
    Unknown,
}

/// Replica role in a deployment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplicaRole {
    Primary,
    Secondary,
    Standalone,
}

/// GPU requirement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirement {
    pub count: u32,
    pub min_memory_gb: u32,
    pub compute_capability: Option<(u32, u32)>,
}

/// Placement request from scheduler to find suitable nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementRequest {
    pub model_name: String,
    pub engine_type: String,
    pub replicas: u32,
    pub gpu_requirement: Option<GpuRequirement>,
}

/// Assignment of a replica to a specific node/device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaAssignment {
    pub replica_id: Uuid,
    pub node_id: String,
    pub device_indices: Vec<u32>,
    pub role: ReplicaRole,
}

/// Complete placement plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub plan_id: Uuid,
    pub model_name: String,
    pub assignments: Vec<ReplicaAssignment>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Endpoint information for accessing a running engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointInfo {
    pub address: String,
    pub port: u16,
    pub protocol: String, // "grpc" or "http"
}

/// Engine backend type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EngineBackend {
    #[default]
    Native,
    Docker,
}

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub engine_type: String,
    pub model_name: String,
    pub model_path: Option<String>,
    pub tensor_parallel_size: u32,
    pub pipeline_parallel_size: u32,
    pub max_num_seqs: u32,
    pub max_model_len: Option<u32>,
    pub dtype: Option<String>,
    pub quantization: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub gpu_memory_utilization: f32,
    pub enforce_eager: bool,
    pub additional_args: HashMap<String, String>,
    /// Engine backend: native (Python subprocess) or docker
    #[serde(default)]
    pub backend: EngineBackend,
    /// Docker image for docker backend (e.g. "vllm/vllm-openai:v0.11.0")
    pub docker_image: Option<String>,
    /// Container name prefix for docker backend
    pub container_name: Option<String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            engine_type: "vllm".to_string(),
            model_name: String::new(),
            model_path: None,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            max_num_seqs: 256,
            max_model_len: None,
            dtype: Some("auto".to_string()),
            quantization: None,
            kv_cache_dtype: Some("auto".to_string()),
            gpu_memory_utilization: 0.9,
            enforce_eager: false,
            additional_args: HashMap::new(),
            backend: EngineBackend::Native,
            docker_image: None,
            container_name: None,
        }
    }
}

/// Engine status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EngineStatus {
    Pending,
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
}

/// Running engine instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningEngine {
    pub engine_id: Uuid,
    pub config: EngineConfig,
    pub status: EngineStatus,
    pub endpoint: Option<EndpointInfo>,
    pub node_id: String,
    pub device_indices: Vec<u32>,
    pub pid: Option<u32>,
    /// Docker container ID (for docker backend)
    pub container_id: Option<String>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub error_message: Option<String>,
}
