use axum::{http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

pub async fn list_models() -> impl IntoResponse {
    // TODO: List models from registry
    
    let response = ModelList {
        object: "list".to_string(),
        data: vec![],
    };
    
    (StatusCode::OK, Json(response))
}
