use lazy_static::lazy_static;
use prometheus::{
    HistogramOpts, HistogramVec,
    IntCounterVec, IntGaugeVec, Opts, Registry, TextEncoder, Encoder,
};
use std::sync::Arc;
use deepinfer_common::types::{EngineStatus, RunningEngine};
use deepinfer_meta::MetaStore;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();
    
    // Request metrics
    pub static ref HTTP_REQUESTS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("deepinfer_http_requests_total", "Total number of HTTP requests"),
        &["method", "path", "status"]
    ).unwrap();
    
    pub static ref HTTP_REQUEST_DURATION_SECONDS: HistogramVec = HistogramVec::new(
        HistogramOpts::new("deepinfer_http_request_duration_seconds", "HTTP request duration in seconds")
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        &["method", "path"]
    ).unwrap();
    
    // Engine metrics
    pub static ref ENGINES_TOTAL: IntGaugeVec = IntGaugeVec::new(
        Opts::new("deepinfer_engines_total", "Total number of engines by status"),
        &["status"]
    ).unwrap();
    
    pub static ref ENGINE_RESTARTS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("deepinfer_engine_restarts_total", "Total number of engine restarts"),
        &["model"]
    ).unwrap();
    
    // Inference metrics
    pub static ref INFERENCE_REQUESTS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("deepinfer_inference_requests_total", "Total number of inference requests"),
        &["model", "stream"]
    ).unwrap();
    
    pub static ref INFERENCE_LATENCY_SECONDS: HistogramVec = HistogramVec::new(
        HistogramOpts::new("deepinfer_inference_latency_seconds", "Inference latency in seconds")
            .buckets(vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]),
        &["model"]
    ).unwrap();
    
    pub static ref INFERENCE_TOKENS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("deepinfer_inference_tokens_total", "Total number of tokens processed"),
        &["model", "type"]  // type: prompt, completion
    ).unwrap();
}

pub fn register_metrics() {
    REGISTRY.register(Box::new(HTTP_REQUESTS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(HTTP_REQUEST_DURATION_SECONDS.clone())).ok();
    REGISTRY.register(Box::new(ENGINES_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(ENGINE_RESTARTS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(INFERENCE_REQUESTS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(INFERENCE_LATENCY_SECONDS.clone())).ok();
    REGISTRY.register(Box::new(INFERENCE_TOKENS_TOTAL.clone())).ok();
}

pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Update engine status metrics by querying the metadata store
pub async fn update_engine_metrics(store: Arc<dyn MetaStore>) {
    // Reset all engine gauges to zero first
    ENGINES_TOTAL.with_label_values(&["running"]).set(0);
    ENGINES_TOTAL.with_label_values(&["failed"]).set(0);
    ENGINES_TOTAL.with_label_values(&["stopping"]).set(0);
    ENGINES_TOTAL.with_label_values(&["unknown"]).set(0);
    
    // Query all engines from the store
    if let Ok(engines) = store.list("/engines").await {
        let mut running_count = 0;
        let mut failed_count = 0;
        let mut stopping_count = 0;
        let mut unknown_count = 0;
        
        for (_key, value) in engines {
            if let Ok(engine) = serde_json::from_slice::<RunningEngine>(&value) {
                match engine.status {
                    EngineStatus::Running => running_count += 1,
                    EngineStatus::Failed => failed_count += 1,
                    EngineStatus::Stopping => stopping_count += 1,
                    _ => unknown_count += 1,
                }
            }
        }
        
        ENGINES_TOTAL.with_label_values(&["running"]).set(running_count);
        ENGINES_TOTAL.with_label_values(&["failed"]).set(failed_count);
        ENGINES_TOTAL.with_label_values(&["stopping"]).set(stopping_count);
        ENGINES_TOTAL.with_label_values(&["unknown"]).set(unknown_count);
    }
}
