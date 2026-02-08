use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use deepinfer_router::KvAwareRouter;
use deepinfer_common::types::{RunningEngine, EngineStatus};
use tracing::{info, error};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub async fn chat_completions(
    State((store, _router)): State<(Arc<dyn deepinfer_meta::MetaStore>, Arc<KvAwareRouter>)>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // 1. Find running engine for the requested model
    let engine = match find_running_engine(&store, &req.model).await {
        Ok(e) => e,
        Err(e) => {
            error!("Failed to find engine for model {}: {}", req.model, e);
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("No running engine found for model: {}", req.model),
                        "type": "model_not_found"
                    }
                }))
            ).into_response();
        }
    };
    
    let endpoint = match &engine.endpoint {
        Some(ep) => ep,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "Engine endpoint not available",
                        "type": "service_unavailable"
                    }
                }))
            ).into_response();
        }
    };
    
    info!("Routing request to {}:{} (protocol: {})", endpoint.address, endpoint.port, endpoint.protocol);
    
    // 2. Forward request based on protocol
    match endpoint.protocol.as_str() {
        "http" => forward_http_request(endpoint, req).await,
        "grpc" => forward_grpc_request(endpoint, req).await,
        _ => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Unknown protocol: {}", endpoint.protocol),
                        "type": "internal_error"
                    }
                }))
            ).into_response()
        }
    }
}

async fn find_running_engine(
    store: &Arc<dyn deepinfer_meta::MetaStore>,
    model_name: &str,
) -> anyhow::Result<RunningEngine> {
    let engines = store.list("/engines").await?;
    
    for (_key, value) in engines {
        let engine: RunningEngine = serde_json::from_slice(&value)?;
        if engine.status == EngineStatus::Running && engine.config.model_name == model_name {
            return Ok(engine);
        }
    }
    
    // Also check by model path (for docker mode where model_path is set)
    let engines = store.list("/engines").await?;
    for (_key, value) in engines {
        let engine: RunningEngine = serde_json::from_slice(&value)?;
        if engine.status == EngineStatus::Running {
            if let Some(path) = &engine.config.model_path {
                if path.contains(model_name) || model_name.contains(path) {
                    return Ok(engine);
                }
            }
        }
    }
    
    anyhow::bail!("No running engine found for model: {}", model_name)
}

/// Forward request to HTTP backend (Docker vLLM with OpenAI-compatible API)
async fn forward_http_request(
    endpoint: &deepinfer_common::types::EndpointInfo,
    req: ChatCompletionRequest,
) -> Response {
    let client = reqwest::Client::new();
    let url = format!("http://{}:{}/v1/chat/completions", endpoint.address, endpoint.port);
    
    info!("Forwarding HTTP request to: {}", url);
    
    // For Docker backend, the model name in vLLM is typically "/model" (the mount point)
    let mut forward_req = req.clone();
    forward_req.model = "/model".to_string();
    
    match client.post(&url)
        .json(&forward_req)
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.json::<ChatCompletionResponse>().await {
                    Ok(mut chat_resp) => {
                        // Restore the original model name in the response
                        chat_resp.model = req.model;
                        (StatusCode::OK, Json(chat_resp)).into_response()
                    }
                    Err(e) => {
                        error!("Failed to parse response: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": {
                                    "message": format!("Failed to parse backend response: {}", e),
                                    "type": "internal_error"
                                }
                            }))
                        ).into_response()
                    }
                }
            } else {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                error!("Backend returned error: {} - {}", status, body);
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    body
                ).into_response()
            }
        }
        Err(e) => {
            error!("Failed to forward request: {}", e);
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Failed to connect to backend: {}", e),
                        "type": "bad_gateway"
                    }
                }))
            ).into_response()
        }
    }
}

/// Forward request to gRPC backend (native vLLM with gRPC shim)
async fn forward_grpc_request(
    endpoint: &deepinfer_common::types::EndpointInfo,
    req: ChatCompletionRequest,
) -> Response {
    // TODO: Implement gRPC forwarding
    // For now, return not implemented
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "gRPC forwarding not yet implemented",
                "type": "not_implemented"
            }
        }))
    ).into_response()
}
