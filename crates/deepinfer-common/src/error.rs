use thiserror::Error;

#[derive(Error, Debug)]
pub enum XinfError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Scheduling error: {0}")]
    Scheduling(String),

    #[error("Engine error: {0}")]
    Engine(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, XinfError>;
