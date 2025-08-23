mod config;

use axum::{extract::State, response::{IntoResponse, Response, sse::{Sse, Event}}, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, convert::Infallible, sync::Arc};
use tonic::transport::Channel;
use tracing::{error, info};
use worker_proto::engine::v1::{
    engine_worker_client::EngineWorkerClient,
    CreateSessionRequest, DecodeRequest, HealthRequest, PrefillRequest,
    SamplingParams, StartSequenceRequest,
};
use tokenizers::Tokenizer;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
struct AppState {
    channel: Channel,
    config: config::Config,
    tokenizer: Arc<Tokenizer>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // load config
    let cfg_path = std::env::var("DEEPINFER_CONFIG").unwrap_or_else(|_| "config/default.yaml".to_string());
    let cfg = config::Config::from_file(&cfg_path)?;

    // connect to worker (tonic channel is cheap to clone)
    let channel = Channel::from_shared(cfg.worker.address.clone())?
        .connect()
        .await?;

    // load tokenizer
    let tokenizer = Tokenizer::from_file(&cfg.model.tokenizer_json)
        .map_err(|e| format!("failed to load tokenizer: {}", e))?;

    let state = AppState { channel, config: cfg.clone(), tokenizer: Arc::new(tokenizer) };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/generate", post(generate_handler))
        .with_state(state.clone());

    let addr: SocketAddr = format!("{}:{}", cfg.server.host, cfg.server.port).parse()?;
    info!("listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut client = EngineWorkerClient::new(state.channel.clone());
    let res = client.health(HealthRequest{}).await;
    match res {
        Ok(resp) => {
            let inner = resp.into_inner();
            Json(serde_json::json!({
                "status": if inner.ok {"ok"} else {"error"},
                "worker_version": inner.worker_version,
            }))
        }
        Err(e) => {
            error!("worker health error: {}", e);
            Json(serde_json::json!({ "status": "unreachable", "error": e.to_string() }))
        }
    }
}

#[derive(Serialize)]
struct GenerateOut {
    text: String,
    tokens: Vec<u32>,
}

#[derive(Deserialize, Clone, Default)]
struct SamplingIn {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
}

impl SamplingIn {
    fn to_proto(&self) -> SamplingParams {
        SamplingParams{
            temperature: self.temperature.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            top_p: self.top_p.unwrap_or(1.0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            seed: self.seed.unwrap_or(0),
        }
    }
}

#[derive(Deserialize)]
struct GenerateIn {
    prompt: String,
    #[serde(default)]
    max_new_tokens: Option<u32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    sample_in_worker: Option<bool>,
    #[serde(default)]
    sampling: Option<SamplingIn>,
}

async fn generate_handler(State(state): State<AppState>, Json(input): Json<GenerateIn>) -> Response {
    let mut client = EngineWorkerClient::new(state.channel.clone());

    // Create a session per request (for demo)
    let m = &state.config.model;
    let cs = client.create_session(CreateSessionRequest{
        model_id: m.model_id.clone(),
        dtype: m.dtype.clone(),
        device: m.device.clone(),
        adapters: m.adapters.clone(),
    }).await.expect("create_session failed").into_inner();
    assert!(cs.ok, "worker returned error: {}", cs.error);

    let ss = client.start_sequence(StartSequenceRequest{ session_id: cs.session_id.clone() })
        .await.expect("start_sequence failed").into_inner();
    assert!(ss.ok, "worker returned error: {}", ss.error);

    // real tokenization via tokenizers
    let encoding = state.tokenizer.encode(input.prompt.clone(), true)
        .map_err(|e| format!("tokenize error: {}", e)).unwrap();
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

    let pre = client.prefill(PrefillRequest{
        session_id: cs.session_id.clone(),
        seq_id: ss.seq_id.clone(),
        tokens: tokens.clone(),
        start_pos: 0,
        return_last_logits: false,
    }).await.expect("prefill failed").into_inner();
    assert!(pre.ok, "worker prefill error: {}", pre.error);

    // start with last prompt token (or EOS if empty)
    let eos_id = state.tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);
    let mut cur_token: u32 = tokens.last().copied().unwrap_or(eos_id);
    let mut pos: u32 = pre.consumed;
    let steps = input.max_new_tokens.or(state.config.max_new_tokens_default).unwrap_or(8);

    let do_stream = input.stream.unwrap_or(false);
    let do_sample_in_worker = input.sample_in_worker.unwrap_or(true);
    let sampling_params = input.sampling.clone().unwrap_or_default().to_proto();

    if do_stream {
        // SSE streaming
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
        let channel = state.channel.clone();
        let tokenizer = state.tokenizer.clone();
        let session_id = cs.session_id.clone();
        let seq_id = ss.seq_id.clone();
        let mut init_tokens = tokens.clone();

        tokio::spawn(async move {
            let mut client = EngineWorkerClient::new(channel);
            let mut prev_text = String::new();

            // send a start event
            let _ = tx.send(Ok(Event::default().json_data(serde_json::json!({
                "event": "start"
            })).unwrap())).await;

            for _ in 0..steps {
                let dec = match client.decode(DecodeRequest{
                    session_id: session_id.clone(),
                    seq_id: seq_id.clone(),
                    token: cur_token,
                    pos,
                    sample_in_worker: do_sample_in_worker,
                    sampling: Some(sampling_params.clone()),
                }).await {
                    Ok(r) => r.into_inner(),
                    Err(e) => {
                        let _ = tx.send(Ok(Event::default().json_data(serde_json::json!({
                            "event": "error",
                            "message": e.to_string()
                        })).unwrap())).await;
                        break;
                    }
                };
                if !dec.ok {
                    let _ = tx.send(Ok(Event::default().json_data(serde_json::json!({
                        "event": "error",
                        "message": dec.error
                    })).unwrap())).await;
                    break;
                }
                let next = dec.next_token;
                init_tokens.push(next);
                cur_token = next;
                pos += 1;

                // decode current full text and send delta
                let text_all = tokenizer.decode(&init_tokens, true).unwrap_or_default();
                let delta = text_all.get(prev_text.len()..).unwrap_or("");
                prev_text = text_all.clone();

                let _ = tx.send(Ok(Event::default().json_data(serde_json::json!({
                    "event": "delta",
                    "token": next,
                    "text_delta": delta,
                })).unwrap())).await;
            }

            // release
            let _ = client.release_sequence(worker_proto::engine::v1::ReleaseSequenceRequest{
                session_id: session_id.clone(),
                seq_id: seq_id.clone(),
            }).await;

            // done
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        let stream = ReceiverStream::new(rx);
        return Sse::new(stream).into_response();
    }

    // non-stream path: generate then return final JSON
    let mut new_tokens = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let dec = client.decode(DecodeRequest{
            session_id: cs.session_id.clone(),
            seq_id: ss.seq_id.clone(),
            token: cur_token,
            pos,
            sample_in_worker: do_sample_in_worker,
            sampling: Some(sampling_params.clone()),
        }).await.expect("decode failed").into_inner();
        assert!(dec.ok, "worker decode error: {}", dec.error);
        let next = dec.next_token;
        new_tokens.push(next);
        cur_token = next;
        pos += 1;
    }

    // best-effort release (ignore error)
    let _ = client.release_sequence(worker_proto::engine::v1::ReleaseSequenceRequest{
        session_id: cs.session_id,
        seq_id: ss.seq_id,
    }).await;

    // decode full sequence (prompt + new tokens)
    let mut all_tokens = tokens.clone();
    all_tokens.extend_from_slice(&new_tokens);
    let text = state.tokenizer.decode(&all_tokens, true).unwrap_or_else(|_| "".to_string());
    Json(GenerateOut { text, tokens: all_tokens }).into_response()
}

