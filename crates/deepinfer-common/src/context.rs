use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Priority level for execution
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

/// Execution context for requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub request_id: Uuid,
    pub user_id: Option<String>,
    pub priority: Priority,
    pub timeout_ms: Option<u64>,
    pub session_id: Option<String>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            request_id: Uuid::new_v4(),
            user_id: None,
            priority: Priority::Normal,
            timeout_ms: None,
            session_id: None,
        }
    }
}
