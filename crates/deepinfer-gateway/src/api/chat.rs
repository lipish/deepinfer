use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    Json,
};
use axum::response::sse::{Event, KeepAlive};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::convert::Infallible;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<ChatDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
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
    
    info!("Routing request to {}:{} (protocol: {}, stream: {})", 
          endpoint.address, endpoint.port, endpoint.protocol, req.stream);
    
    // 2. Forward request based on protocol and stream mode
    match endpoint.protocol.as_str() {
        "http" => {
            if req.stream {
                forward_http_stream(endpoint, req).await
            } else {
                forward_http_request(endpoint, req).await
            }
        }
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

/// Forward streaming request to HTTP backend
async fn forward_http_stream(
    endpoint: &deepinfer_common::types::EndpointInfo,
    req: ChatCompletionRequest,
) -> Response {
    let client = reqwest::Client::new();
    let url = format!("http://{}:{}/v1/chat/completions", endpoint.address, endpoint.port);
    
    info!("Forwarding streaming HTTP request to: {}", url);
    
    // For Docker backend, the model name in vLLM is typically "/model"
    let mut forward_req = req.clone();
    forward_req.model = "/model".to_string();
    forward_req.stream = true;
    
    let original_model = req.model.clone();
    
    match client.post(&url)
        .json(&forward_req)
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                let byte_stream = resp.bytes_stream();
                
                let stream = byte_stream.map(move |chunk_result| {
                    match chunk_result {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            // Process SSE data lines
                            let mut events = Vec::new();
                            for line in text.lines() {
                                if line.starts_with("data: ") {
                                    let data = &line[6..];
                                    if data == "[DONE]" {
                                        events.push(Event::default().data("[DONE]"));
                                    } else {
                                        // Replace /model with original model name in response
                                        let modified = data.replace("\"/model\"", &format!("\"{}\"", original_model));
                                        events.push(Event::default().data(modified));
                                    }
                                }
                            }
                            if events.is_empty() {
                                Ok::<_, Infallible>(Event::default().comment("heartbeat"))
                            } else {
                                // Return first event, the rest will be in subsequent chunks
                                Ok(events.into_iter().next().unwrap_or_else(|| Event::default().comment("empty")))
                            }
                        }
                        Err(e) => {
                            error!("Stream error: {}", e);
                            Ok(Event::default().data(format!("{{\"error\": \"{}\"}}", e)))
                        }
                    }
                });
                
                Sse::new(stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
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
            error!("Failed to forward streaming request: {}", e);
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
    _endpoint: &deepinfer_common::types::EndpointInfo,
    _req: ChatCompletionRequest,
) -> Response {
    // Not implemented - we use Docker mode only
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "gRPC forwarding not supported. Use Docker backend.",
                "type": "not_implemented"
            }
        }))
    ).into_response()
}
