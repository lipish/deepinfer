mod config;

use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, net::SocketAddr, sync::Arc, time::Duration};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;
use tracing::{error, info};
use worker_proto::engine::v1::{
    engine_worker_client::EngineWorkerClient, BatchDecodeRequest, BatchPrefillRequest,
    CreateSessionRequest, DecodeRequest, HealthRequest, SamplingParams,
    StartSequenceRequest, Tokens,
};

#[derive(Clone)]
struct AppState {
    config: config::Config,
    tokenizer: Arc<Tokenizer>,
    workers: Arc<Vec<WorkerCtx>>,
    rr: Arc<Mutex<usize>>, // round-robin cursor
}

impl AppState {
    async fn pick_worker(&self) -> WorkerCtx {
        let mut idx = self.rr.lock().await;
        let i = *idx % self.workers.len();
        *idx = *idx + 1;
        self.workers[i].clone()
    }
}

#[derive(Clone)]
struct WorkerCtx {
    channel: Channel,
    session: Arc<Mutex<Option<String>>>,
    decode_batcher: DecodeBatcher,
    prefill_batcher: PrefillBatcher,
}

#[derive(Clone)]
struct DecodeBatcher {
    tx: mpsc::Sender<DecodeJob>,
}

struct DecodeJob {
    seq_id: String,
    token: u32,
    pos: u32,
    sample_in_worker: bool,
    sampling: SamplingParams,
    resp: oneshot::Sender<Result<u32, String>>,
}

impl DecodeBatcher {
    fn start(
        channel: Channel,
        session: Arc<Mutex<Option<String>>>,
        cfg: config::Config,
        max_batch: usize,
        delay: Duration,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<DecodeJob>(1024);
        tokio::spawn(async move {
            loop {
                let first = match rx.recv().await {
                    Some(j) => j,
                    None => break,
                };
                // only support sample_in_worker=true in batcher path
                if !first.sample_in_worker {
                    let _ = first
                        .resp
                        .send(Err("batcher only supports sample_in_worker=true".into()));
                    continue;
                }
                let mut batch = vec![first];
                let start = tokio::time::Instant::now();
                while batch.len() < max_batch {
                    let remain = delay.saturating_sub(start.elapsed());
                    if remain.is_zero() {
                        break;
                    }
                    match tokio::time::timeout(remain, rx.recv()).await {
                        Ok(Some(j)) => {
                            if j.sample_in_worker {
                                batch.push(j);
                            } else {
                                let _ = j.resp.send(Err(
                                    "batcher only supports sample_in_worker=true".into(),
                                ));
                            }
                        }
                        Ok(None) => break,
                        Err(_) => break,
                    }
                }
                // ensure session id
                let sid = if let Some(s) = session.lock().await.as_ref().cloned() {
                    s
                } else {
                    let mut client = EngineWorkerClient::new(channel.clone());
                    let m = &cfg.model;
                    let cs = match client
                        .create_session(CreateSessionRequest {
                            model_id: m.model_id.clone(),
                            dtype: m.dtype.clone(),
                            device: m.device.clone(),
                            adapters: m.adapters.clone(),
                        })
                        .await
                    {
                        Ok(r) => r.into_inner(),
                        Err(e) => {
                            for j in batch {
                                let _ = j.resp.send(Err(format!("create_session error: {}", e)));
                            }
                            continue;
                        }
                    };
                    if !cs.ok {
                        for j in batch {
                            let _ = j
                                .resp
                                .send(Err(format!("create_session failed: {}", cs.error)));
                        }
                        continue;
                    }
                    let mut lock = session.lock().await;
                    *lock = Some(cs.session_id.clone());
                    cs.session_id
                };
                let mut client = EngineWorkerClient::new(channel.clone());
                let seq_ids: Vec<String> = batch.iter().map(|j| j.seq_id.clone()).collect();
                let last_tokens: Vec<u32> = batch.iter().map(|j| j.token).collect();
                let pos_list: Vec<u32> = batch.iter().map(|j| j.pos).collect();
                let sampling = batch.get(0).map(|j| j.sampling.clone()).unwrap_or_default();
                let req = BatchDecodeRequest {
                    session_id: sid,
                    seq_ids,
                    last_tokens,
                    pos_list,
                    sample_in_worker: true,
                    sampling: Some(sampling),
                };
                match client.batch_decode(req).await {
                    Ok(resp) => {
                        let inner = resp.into_inner();
                        if !inner.ok {
                            let err = inner.error;
                            for j in batch {
                                let _ = j.resp.send(Err(err.clone()));
                            }
                            continue;
                        }
                        let outs = inner.next_tokens;
                        if outs.len() != batch.len() {
                            let err = format!(
                                "mismatched next_tokens: {} vs {}",
                                outs.len(),
                                batch.len()
                            );
                            for j in batch {
                                let _ = j.resp.send(Err(err.clone()));
                            }
                            continue;
                        }
                        for (j, t) in batch.into_iter().zip(outs.into_iter()) {
                            let _ = j.resp.send(Ok(t));
                        }
                    }
                    Err(e) => {
                        let err = e.to_string();
                        for j in batch {
                            let _ = j.resp.send(Err(err.clone()));
                        }
                    }
                }
            }
        });
        Self { tx }
    }

    async fn decode(
        &self,
        seq_id: String,
        token: u32,
        pos: u32,
        sample_in_worker: bool,
        sampling: SamplingParams,
    ) -> Result<u32, String> {
        let (tx, rx) = oneshot::channel();
        let job = DecodeJob {
            seq_id,
            token,
            pos,
            sample_in_worker,
            sampling,
            resp: tx,
        };
        self.tx.send(job).await.map_err(|e| e.to_string())?;
        rx.await.map_err(|e| e.to_string())?
    }
}

#[derive(Clone)]
struct PrefillBatcher {
    tx: mpsc::Sender<PrefillJob>,
}

struct PrefillJob {
    seq_id: String,
    tokens: Vec<u32>,
    start_pos: u32,
    return_last_logits: bool,
    resp: oneshot::Sender<Result<u32, String>>, // consumed only
}

impl PrefillBatcher {
    fn start(
        channel: Channel,
        session: Arc<Mutex<Option<String>>>,
        cfg: config::Config,
        max_batch: usize,
        delay: Duration,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<PrefillJob>(1024);
        tokio::spawn(async move {
            loop {
                let first = match rx.recv().await {
                    Some(j) => j,
                    None => break,
                };
                let mut batch = vec![first];
                let start = tokio::time::Instant::now();
                while batch.len() < max_batch {
                    let remain = delay.saturating_sub(start.elapsed());
                    if remain.is_zero() {
                        break;
                    }
                    match tokio::time::timeout(remain, rx.recv()).await {
                        Ok(Some(j)) => batch.push(j),
                        Ok(None) => break,
                        Err(_) => break,
                    }
                }
                // ensure session
                let sid = if let Some(s) = session.lock().await.as_ref().cloned() {
                    s
                } else {
                    let mut client = EngineWorkerClient::new(channel.clone());
                    let m = &cfg.model;
                    let cs = match client
                        .create_session(CreateSessionRequest {
                            model_id: m.model_id.clone(),
                            dtype: m.dtype.clone(),
                            device: m.device.clone(),
                            adapters: m.adapters.clone(),
                        })
                        .await
                    {
                        Ok(r) => r.into_inner(),
                        Err(e) => {
                            for j in batch {
                                let _ = j.resp.send(Err(format!("create_session error: {}", e)));
                            }
                            continue;
                        }
                    };
                    if !cs.ok {
                        for j in batch {
                            let _ = j
                                .resp
                                .send(Err(format!("create_session failed: {}", cs.error)));
                        }
                        continue;
                    }
                    let mut lock = session.lock().await;
                    *lock = Some(cs.session_id.clone());
                    cs.session_id
                };
                let mut client = EngineWorkerClient::new(channel.clone());
                let seq_ids: Vec<String> = batch.iter().map(|j| j.seq_id.clone()).collect();
                let tokens_list: Vec<Tokens> = batch
                    .iter()
                    .map(|j| Tokens {
                        tokens: j.tokens.clone(),
                    })
                    .collect();
                let start_pos_list: Vec<u32> = batch.iter().map(|j| j.start_pos).collect();
                let return_last_logits = batch.iter().any(|j| j.return_last_logits);
                let req = BatchPrefillRequest {
                    session_id: sid,
                    seq_ids,
                    tokens_list,
                    start_pos_list,
                    return_last_logits,
                };
                match client.batch_prefill(req).await {
                    Ok(resp) => {
                        let inner = resp.into_inner();
                        if !inner.ok {
                            let err = inner.error;
                            for j in batch {
                                let _ = j.resp.send(Err(err.clone()));
                            }
                            continue;
                        }
                        let outs = inner.consumed_list;
                        if outs.len() != batch.len() {
                            let err = format!(
                                "mismatched consumed_list: {} vs {}",
                                outs.len(),
                                batch.len()
                            );
                            for j in batch {
                                let _ = j.resp.send(Err(err.clone()));
                            }
                            continue;
                        }
                        for (j, c) in batch.into_iter().zip(outs.into_iter()) {
                            let _ = j.resp.send(Ok(c));
                        }
                    }
                    Err(e) => {
                        let err = e.to_string();
                        for j in batch {
                            let _ = j.resp.send(Err(err.clone()));
                        }
                    }
                }
            }
        });
        Self { tx }
    }

    async fn prefill(
        &self,
        seq_id: String,
        tokens: Vec<u32>,
        start_pos: u32,
        return_last_logits: bool,
    ) -> Result<u32, String> {
        let (tx, rx) = oneshot::channel();
        let job = PrefillJob {
            seq_id,
            tokens,
            start_pos,
            return_last_logits,
            resp: tx,
        };
        self.tx.send(job).await.map_err(|e| e.to_string())?;
        rx.await.map_err(|e| e.to_string())?
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // load config
    let cfg_path =
        std::env::var("DEEPINFER_CONFIG").unwrap_or_else(|_| "config/default.yaml".to_string());
    let cfg = config::Config::from_file(&cfg_path)?;

    // connect to workers (support single or multiple)
    let mut worker_addrs = cfg.worker.addresses.clone();
    if worker_addrs.is_empty() {
        worker_addrs.push(cfg.worker.address.clone());
    }
    let mut workers: Vec<WorkerCtx> = Vec::new();
    for addr in worker_addrs.iter() {
        let channel = Channel::from_shared(addr.clone())?.connect().await?;
        // per-worker session and batchers
        let session: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let max_batch: usize = std::env::var("BATCH_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        let delay_ms: u64 = std::env::var("BATCH_DELAY_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
        let decode_batcher = DecodeBatcher::start(channel.clone(), session.clone(), cfg.clone(), max_batch, Duration::from_millis(delay_ms));
        let prefill_batcher = PrefillBatcher::start(channel.clone(), session.clone(), cfg.clone(), max_batch, Duration::from_millis(delay_ms));
        workers.push(WorkerCtx { channel, session, decode_batcher, prefill_batcher });
    }

    // load tokenizer
    let tokenizer = Tokenizer::from_file(&cfg.model.tokenizer_json)
        .map_err(|e| format!("failed to load tokenizer: {}", e))?;

    let state = AppState {
        config: cfg.clone(),
        tokenizer: Arc::new(tokenizer),
        workers: Arc::new(workers),
        rr: Arc::new(Mutex::new(0)),
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/generate", post(generate_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state.clone());

    let addr: SocketAddr = format!("{}:{}", cfg.server.host, cfg.server.port).parse()?;
    info!("listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    // ping the first worker for health
    let mut client = EngineWorkerClient::new(state.workers[0].channel.clone());
    let res = client.health(HealthRequest {}).await;
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
        SamplingParams {
            temperature: self.temperature.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            top_p: self.top_p.unwrap_or(1.0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            seed: self.seed.unwrap_or(0),
        }
    }
}

#[derive(Deserialize, Clone)]
struct ChatMessageIn {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatMessageOut {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionsIn {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessageIn>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: Option<bool>,
    // OpenAI-style sampling fields (subset)
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct ChatChoiceOut {
    index: u32,
    message: ChatMessageOut,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatCompletionsOut {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoiceOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct ChatChunkChoice {
    index: u32,
    delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatCompletionsChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice>,
}

fn build_prompt_from_messages(msgs: &[ChatMessageIn]) -> String {
    let mut s = String::new();
    for m in msgs.iter() {
        match m.role.as_str() {
            "system" => {
                s.push_str("System: ");
                s.push_str(&m.content);
                s.push_str("\n\n");
            }
            "user" => {
                s.push_str("User: ");
                s.push_str(&m.content);
                s.push_str("\n\n");
            }
            "assistant" => {
                s.push_str("Assistant: ");
                s.push_str(&m.content);
                s.push_str("\n\n");
            }
            _ => {
                s.push_str(&m.content);
                s.push_str("\n\n");
            }
        }
    }
    s.push_str("Assistant: ");
    s
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

async fn generate_handler(
    State(state): State<AppState>,
    Json(input): Json<GenerateIn>,
) -> Response {
    // pick a worker (round-robin)
    let wk = state.pick_worker().await;
    let mut client = EngineWorkerClient::new(wk.channel.clone());

    // Ensure a shared session (for batching)
    let session_id = if let Some(s) = wk.session.lock().await.as_ref().cloned() { s } else {
        let m = &state.config.model;
        let cs = client.create_session(CreateSessionRequest{ model_id: m.model_id.clone(), dtype: m.dtype.clone(), device: m.device.clone(), adapters: m.adapters.clone() }).await.expect("create_session failed").into_inner();
        assert!(cs.ok, "worker returned error: {}", cs.error);
        let mut lock = wk.session.lock().await; *lock = Some(cs.session_id.clone()); cs.session_id
    };

    // Start a new sequence per request
    let ss = client
        .start_sequence(StartSequenceRequest {
            session_id: session_id.clone(),
        })
        .await
        .expect("start_sequence failed")
        .into_inner();
    assert!(ss.ok, "worker returned error: {}", ss.error);

    // real tokenization via tokenizers
    let encoding = state
        .tokenizer
        .encode(input.prompt.clone(), true)
        .map_err(|e| format!("tokenize error: {}", e))
        .unwrap();
    let tokens: Vec<u32> = encoding.get_ids().to_vec();

    // Batch prefill via batcher
    let consumed = wk
        .prefill_batcher
        .prefill(ss.seq_id.clone(), tokens.clone(), 0, false)
        .await
        .expect("batch prefill failed");

    // start with last prompt token (or EOS if empty)
    let eos_id = state.tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);
    let mut cur_token: u32 = tokens.last().copied().unwrap_or(eos_id);
    let mut pos: u32 = consumed;
    let steps = input
        .max_new_tokens
        .or(state.config.max_new_tokens_default)
        .unwrap_or(8);

    let do_stream = input.stream.unwrap_or(false);
    let do_sample_in_worker = input.sample_in_worker.unwrap_or(true);
    let sampling_params = input.sampling.clone().unwrap_or_default().to_proto();

    if do_stream {
        // SSE streaming
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
        let st = state.clone();
        let wk2 = wk.clone();
        let seq_id = ss.seq_id.clone();
        let mut init_tokens = tokens.clone();

        tokio::spawn(async move {
            let mut client = EngineWorkerClient::new(wk2.channel.clone());
            let mut prev_text = String::new();

            // send a start event
            let _ = tx
                .send(Ok(Event::default()
                    .json_data(serde_json::json!({
                        "event": "start"
                    }))
                    .unwrap()))
                .await;

            for _ in 0..steps {
                let next = if do_sample_in_worker {
                    match wk2
                        .decode_batcher
                        .decode(
                            seq_id.clone(),
                            cur_token,
                            pos,
                            true,
                            sampling_params.clone(),
                        )
                        .await
                    {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = tx
                                .send(Ok(Event::default()
                                    .json_data(serde_json::json!({
                                        "event": "error",
                                        "message": e
                                    }))
                                    .unwrap()))
                                .await;
                            break;
                        }
                    }
                } else {
                    // fallback to direct decode when not sampling in worker
                    // ensure a session on this worker
                    let session_id = if let Some(s) = wk2.session.lock().await.as_ref().cloned() { s } else {
                        let m = &st.config.model;
                        let cs = client.create_session(CreateSessionRequest{ model_id: m.model_id.clone(), dtype: m.dtype.clone(), device: m.device.clone(), adapters: m.adapters.clone() }).await.expect("create_session failed").into_inner();
                        assert!(cs.ok, "worker returned error: {}", cs.error);
                        let mut lock = wk2.session.lock().await; *lock = Some(cs.session_id.clone()); cs.session_id
                    };
                    match client
                        .decode(DecodeRequest {
                            session_id,
                            seq_id: seq_id.clone(),
                            token: cur_token,
                            pos,
                            sample_in_worker: false,
                            sampling: Some(sampling_params.clone()),
                        })
                        .await
                    {
                        Ok(r) => {
                            let inner = r.into_inner();
                            if !inner.ok {
                                break;
                            }
                            // pick argmax on client if needed; but in this path logits are returned; keeping simple
                            inner.next_token
                        }
                        Err(e) => {
                            let _ = tx
                                .send(Ok(Event::default()
                                    .json_data(serde_json::json!({
                                        "event": "error",
                                        "message": e.to_string()
                                    }))
                                    .unwrap()))
                                .await;
                            break;
                        }
                    }
                };
                init_tokens.push(next);
                cur_token = next;
                pos += 1;

                // decode current full text and send delta
                let text_all = st.tokenizer.decode(&init_tokens, true).unwrap_or_default();
                let delta = text_all.get(prev_text.len()..).unwrap_or("");
                prev_text = text_all.clone();

                let _ = tx
                    .send(Ok(Event::default()
                        .json_data(serde_json::json!({
                            "event": "delta",
                            "token": next,
                            "text_delta": delta,
                        }))
                        .unwrap()))
                    .await;
            }

            // release
            // best-effort release on this worker
            // fetch session id (if any)
            let sid = wk2.session.lock().await.clone().unwrap_or_default();
            if !sid.is_empty() {
                let _ = client
                    .release_sequence(worker_proto::engine::v1::ReleaseSequenceRequest {
                        session_id: sid,
                        seq_id: seq_id.clone(),
                    })
                    .await;
            }

            // done
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        let stream = ReceiverStream::new(rx);
        return Sse::new(stream).into_response();
    }

    // non-stream path: generate then return final JSON
    let mut new_tokens = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let next = if do_sample_in_worker {
            wk
                .decode_batcher
                .decode(
                    ss.seq_id.clone(),
                    cur_token,
                    pos,
                    true,
                    sampling_params.clone(),
                )
                .await
                .expect("batch decode failed")
        } else {
            let dec = client
                .decode(DecodeRequest {
                    session_id: session_id.clone(),
                    seq_id: ss.seq_id.clone(),
                    token: cur_token,
                    pos,
                    sample_in_worker: false,
                    sampling: Some(sampling_params.clone()),
                })
                .await
                .expect("decode failed")
                .into_inner();
            assert!(dec.ok, "worker decode error: {}", dec.error);
            dec.next_token
        };
        new_tokens.push(next);
        cur_token = next;
        pos += 1;
    }

    // best-effort release (ignore error)
    let _ = client
        .release_sequence(worker_proto::engine::v1::ReleaseSequenceRequest {
            session_id: session_id,
            seq_id: ss.seq_id,
        })
        .await;

    // decode full sequence (prompt + new tokens)
    let mut all_tokens = tokens.clone();
    all_tokens.extend_from_slice(&new_tokens);
    let text = state
        .tokenizer
        .decode(&all_tokens, true)
        .unwrap_or_else(|_| "".to_string());
    Json(GenerateOut {
        text,
        tokens: all_tokens,
    })
    .into_response()
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(input): Json<ChatCompletionsIn>,
) -> Response {
    // pick worker RR
    let wk = state.pick_worker().await;
    let mut client = EngineWorkerClient::new(wk.channel.clone());

    // Build prompt from messages
    let prompt = build_prompt_from_messages(&input.messages);
    let encoding = state
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("tokenize error: {}", e))
        .unwrap();
    let tokens: Vec<u32> = encoding.get_ids().to_vec();

    // ensure session
    let session_id = if let Some(s) = wk.session.lock().await.as_ref().cloned() { s } else {
        let m = &state.config.model;
        let cs = client.create_session(CreateSessionRequest{ model_id: m.model_id.clone(), dtype: m.dtype.clone(), device: m.device.clone(), adapters: m.adapters.clone() }).await.expect("create_session failed").into_inner();
        assert!(cs.ok, "worker returned error: {}", cs.error);
        let mut lock = wk.session.lock().await; *lock = Some(cs.session_id.clone()); cs.session_id
    };

    // start sequence
    let ss = client
        .start_sequence(StartSequenceRequest{ session_id: session_id.clone() })
        .await.expect("start_sequence failed").into_inner();
    assert!(ss.ok, "worker returned error: {}", ss.error);

    // prefill
    let consumed = wk.prefill_batcher
        .prefill(ss.seq_id.clone(), tokens.clone(), 0, false)
        .await.expect("batch prefill failed");

    // sampling params from chat input
    let sampling = SamplingIn{
        temperature: input.temperature,
        top_k: None,
        top_p: input.top_p,
        repetition_penalty: None,
        frequency_penalty: input.frequency_penalty.or(input.presence_penalty),
        seed: None,
    }.to_proto();

    // controls
    let steps = input.max_tokens.or(state.config.max_new_tokens_default).unwrap_or(8);
    let do_stream = input.stream.unwrap_or(false);
    let model_name = input.model.unwrap_or_else(|| state.config.model.model_id.clone());

    // starting token is last of prompt or EOS
    let eos_id = state.tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);
    let mut cur_token: u32 = tokens.last().copied().unwrap_or(eos_id);
    let mut pos: u32 = consumed;

    // OpenAI fields
    let created: u64 = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let id = format!("chatcmpl-{}", created);

    if do_stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
        let st = state.clone();
        let wk2 = wk.clone();
        let seq_id = ss.seq_id.clone();
        let mut init_tokens = tokens.clone();
        tokio::spawn(async move {
            let mut client = EngineWorkerClient::new(wk2.channel.clone());
            // First chunk with role
            let first = ChatCompletionsChunk{
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChatChunkChoice{ index: 0, delta: ChatDelta{ role: Some("assistant".to_string()), content: None }, finish_reason: None }],
            };
            let _ = tx.send(Ok(Event::default().json_data(first).unwrap())).await;

            let mut prev_text = String::new();
            for _ in 0..steps {
                let next = match wk2.decode_batcher.decode(seq_id.clone(), cur_token, pos, true, sampling.clone()).await {
                    Ok(t) => t,
                    Err(e) => {
                        // send an error-like finish
                        let done = ChatCompletionsChunk{
                            id: id.clone(), object: "chat.completion.chunk".to_string(), created, model: model_name.clone(),
                            choices: vec![ChatChunkChoice{ index: 0, delta: ChatDelta{ role: None, content: None }, finish_reason: Some(format!("error:{}", e)) }]
                        };
                        let _ = tx.send(Ok(Event::default().json_data(done).unwrap())).await;
                        break;
                    }
                };
                init_tokens.push(next);
                cur_token = next;
                pos += 1;

                let text_all = st.tokenizer.decode(&init_tokens, true).unwrap_or_default();
                let delta_txt = text_all.get(prev_text.len()..).unwrap_or("").to_string();
                prev_text = text_all;

                if !delta_txt.is_empty() {
                    let chunk = ChatCompletionsChunk{
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChatChunkChoice{ index: 0, delta: ChatDelta{ role: None, content: Some(delta_txt) }, finish_reason: None }],
                    };
                    let _ = tx.send(Ok(Event::default().json_data(chunk).unwrap())).await;
                }
            }
            // best-effort release
            let sid = wk2.session.lock().await.clone().unwrap_or_default();
            if !sid.is_empty() {
                let _ = client.release_sequence(worker_proto::engine::v1::ReleaseSequenceRequest{ session_id: sid, seq_id: seq_id.clone() }).await;
            }
            // final done chunk
            let done = ChatCompletionsChunk{
                id: id.clone(), object: "chat.completion.chunk".to_string(), created, model: model_name.clone(),
                choices: vec![ChatChunkChoice{ index: 0, delta: ChatDelta{ role: None, content: None }, finish_reason: Some("stop".to_string()) }]
            };
            let _ = tx.send(Ok(Event::default().json_data(done).unwrap())).await;
            // [DONE]
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });
        let stream = ReceiverStream::new(rx);
        return Sse::new(stream).into_response();
    }

    // non-stream: generate tokens
    let mut new_tokens = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let next = wk.decode_batcher.decode(ss.seq_id.clone(), cur_token, pos, true, sampling.clone()).await.expect("batch decode failed");
        new_tokens.push(next);
        cur_token = next;
        pos += 1;
    }
    // best-effort release
    let _ = client.release_sequence(worker_proto::engine::v1::ReleaseSequenceRequest{ session_id: session_id.clone(), seq_id: ss.seq_id.clone() }).await;

    // decode text
    let mut all_tokens = tokens.clone();
    all_tokens.extend_from_slice(&new_tokens);
    let text = state.tokenizer.decode(&all_tokens, true).unwrap_or_else(|_| "".to_string());

    let out = ChatCompletionsOut{
        id,
        object: "chat.completion".to_string(),
        created,
        model: model_name,
        choices: vec![ChatChoiceOut{ index: 0, message: ChatMessageOut{ role: "assistant".to_string(), content: text.clone() }, finish_reason: Some("stop".to_string()) }],
        usage: Some(Usage{ prompt_tokens: tokens.len() as u32, completion_tokens: new_tokens.len() as u32, total_tokens: (tokens.len() + new_tokens.len()) as u32 }),
    };
    Json(out).into_response()
}
