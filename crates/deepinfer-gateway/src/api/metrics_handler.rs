use axum::response::IntoResponse;
use axum::http::{header, StatusCode};
use crate::metrics::gather_metrics;

/// Prometheus metrics endpoint handler
pub async fn metrics_endpoint() -> impl IntoResponse {
    let metrics = gather_metrics();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        metrics
    )
}
