use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub worker: WorkerConfig,
    pub model: ModelConfig,
    #[serde(default)]
    pub max_new_tokens_default: Option<u32>,
}

#[derive(Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Deserialize)]
pub struct WorkerConfig {
    pub address: String, // e.g., http://127.0.0.1:50051
}

#[derive(Clone, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub dtype: String,
    pub device: String,
    pub tokenizer_json: String,
    #[serde(default)]
    pub adapters: Vec<String>,
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let s = std::fs::read_to_string(path)?;
        let cfg: Self = serde_yaml::from_str(&s)?;
        Ok(cfg)
    }
}

