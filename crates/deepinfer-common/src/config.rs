use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Global configuration for the xinf platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub scheduler: SchedulerConfig,
    pub agent: AgentConfig,
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_connections: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub reconcile_interval_secs: u64,
    pub placement_strategy: String,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            reconcile_interval_secs: 5,
            placement_strategy: "idle_first".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub node_id: Option<String>,
    pub heartbeat_interval_secs: u64,
    pub reconcile_interval_secs: u64,
    pub health_check_interval_secs: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            node_id: None,
            heartbeat_interval_secs: 10,
            reconcile_interval_secs: 5,
            health_check_interval_secs: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub backend: String, // "embedded" or "etcd"
    pub path: Option<PathBuf>,
    pub etcd_endpoints: Option<Vec<String>>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: "embedded".to_string(),
            path: Some(PathBuf::from("./data/meta")),
            etcd_endpoints: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            scheduler: SchedulerConfig::default(),
            agent: AgentConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

impl Config {
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }
}
