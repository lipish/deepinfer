use serde::{Deserialize, Serialize};

/// Engine capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapabilities {
    pub supports_streaming: bool,
    pub supports_chat: bool,
    pub supports_embedding: bool,
    pub supports_structured_output: bool,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub max_batch_size: u32,
    pub kv_cache_sharing: bool,
}

impl Default for EngineCapabilities {
    fn default() -> Self {
        Self {
            supports_streaming: true,
            supports_chat: true,
            supports_embedding: false,
            supports_structured_output: false,
            supports_vision: false,
            supports_audio: false,
            max_batch_size: 256,
            kv_cache_sharing: true,
        }
    }
}
