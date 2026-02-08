use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use deepinfer_router::KvAwareRouter;
use deepinfer_common::types::{RunningEngine, EngineStatus};
use tracing::error;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

pub async fn list_models(
    State((store, _router)): State<(Arc<dyn deepinfer_meta::MetaStore>, Arc<KvAwareRouter>)>,
) -> impl IntoResponse {
    // List all running engines from MetaStore
    let engines = match store.list("/engines").await {
        Ok(e) => e,
        Err(e) => {
            error!("Failed to list engines: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ModelList {
                    object: "list".to_string(),
                    data: vec![],
                })
            );
        }
    };
    
    let mut models = Vec::new();
    
    for (_key, value) in engines {
        if let Ok(engine) = serde_json::from_slice::<RunningEngine>(&value) {
            if engine.status == EngineStatus::Running {
                let created = engine.started_at
                    .map(|t| t.timestamp())
                    .unwrap_or(0);
                
                models.push(ModelInfo {
                    id: engine.config.model_name.clone(),
                    object: "model".to_string(),
                    created,
                    owned_by: "deepinfer".to_string(),
                    root: engine.config.model_path.clone(),
                    parent: None,
                });
            }
        }
    }
    
    let response = ModelList {
        object: "list".to_string(),
        data: models,
    };
    
    (StatusCode::OK, Json(response))
}
