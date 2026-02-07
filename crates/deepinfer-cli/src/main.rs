use clap::{Parser, Subcommand};
use anyhow::Result;
use tracing_subscriber;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "deepinfer")]
#[command(about = "deepinfer - Rust-Native Inference Platform", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the gateway and scheduler
    Serve {
        #[arg(short, long, default_value = "configs/default.toml")]
        config: String,
    },
    /// Start a worker agent
    Worker {
        #[arg(short, long)]
        node_id: Option<String>,
    },
    /// Launch a model
    Launch {
        #[arg(short, long)]
        model: String,
        #[arg(short, long, default_value = "vllm")]
        engine: String,
        #[arg(short, long)]
        device: Option<String>,
    },
    /// List running models
    List,
    /// Terminate a model instance
    Terminate {
        #[arg(short, long)]
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { config } => {
            serve(config).await?;
        }
        Commands::Worker { node_id } => {
            worker(node_id).await?;
        }
        Commands::Launch { model, engine, device } => {
            launch(model, engine, device).await?;
        }
        Commands::List => {
            list().await?;
        }
        Commands::Terminate { model } => {
            terminate(model).await?;
        }
    }

    Ok(())
}

async fn serve(config_path: String) -> Result<()> {
    tracing::info!("Starting xinf server with config: {}", config_path);
    
    let config = deepinfer_common::Config::load_from_file(&config_path)
        .unwrap_or_default();
    
    // Initialize MetaStore (etcd only)
    let endpoints = config.storage.etcd_endpoints
        .unwrap_or_else(|| vec!["localhost:2379".to_string()]);
    
    tracing::info!("Connecting to etcd: {:?}", endpoints);
    let store = Arc::new(deepinfer_meta::EtcdStore::connect(endpoints).await?);
    
    // Initialize scheduler
    let snapshot = Arc::new(deepinfer_scheduler::ClusterSnapshot::new());
    let strategy = Arc::new(deepinfer_scheduler::IdleFirstStrategy);
    let _scheduler = deepinfer_scheduler::Scheduler::new(snapshot.clone(), strategy);
    
    // Initialize router
    let endpoint_mgr = Arc::new(deepinfer_router::EndpointManager::new());
    let router = Arc::new(deepinfer_router::KvAwareRouter::new(endpoint_mgr));
    
    // Build Axum app
    use axum::{
        routing::{get, post},
        Router,
    };
    use tower_http::trace::TraceLayer;
    
    // Create a shared state tuple
    type AppState = (Arc<dyn deepinfer_meta::MetaStore>, Arc<deepinfer_router::KvAwareRouter>);
    let state: AppState = (store.clone() as Arc<dyn deepinfer_meta::MetaStore>, router);
    
    let app = Router::new()
        .route("/health", get(deepinfer_gateway::health_check))
        .route("/v1/models", get(deepinfer_gateway::list_models))
        .route("/v1/deployments", post(deepinfer_gateway::launch_model))
        .route("/v1/chat/completions", post(deepinfer_gateway::chat_completions))
        .with_state(state)
        .layer(TraceLayer::new_for_http());
    
    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!("Listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn worker(node_id: Option<String>) -> Result<()> {
    let node_id = node_id.unwrap_or_else(|| {
        hostname::get()
            .ok()
            .and_then(|h| h.into_string().ok())
            .unwrap_or_else(|| "worker-1".to_string())
    });
    
    tracing::info!("Starting worker agent: {}", node_id);
    
    // Load config
    let config = deepinfer_common::Config::load_from_file("configs/default.toml")
        .unwrap_or_default();
    
    // Initialize MetaStore (etcd only)
    let endpoints = config.storage.etcd_endpoints
        .unwrap_or_else(|| vec!["localhost:2379".to_string()]);
    
    tracing::info!("Connecting to etcd: {:?}", endpoints);
    let store = Arc::new(deepinfer_meta::EtcdStore::connect(endpoints).await?);
    
    // Create and run agent
    let agent = deepinfer_agent::WorkerAgent::new(node_id, store);
    agent.run().await;
    
    Ok(())
}

async fn launch(model: String, engine: String, device: Option<String>) -> Result<()> {
    tracing::info!("Launching model via API: {}", model);
    
    let client = reqwest::Client::new();
    
    let payload = serde_json::json!({
        "model": model,
        "engine": engine,
        "device": device,
    });
    
    let response = client
        .post("http://localhost:8082/v1/deployments")
        .json(&payload)
        .send()
        .await?;
    
    if response.status().is_success() {
        let result: serde_json::Value = response.json().await?;
        println!("✓ {}", result);
    } else {
        let error_text = response.text().await?;
        println!("✗ Failed to launch model: {}", error_text);
    }
    
    Ok(())
}

async fn list() -> Result<()> {
    tracing::info!("List command - TODO: Implement");
    println!("TODO: Implement list running models via API call");
    Ok(())
}

async fn terminate(_model: String) -> Result<()> {
    tracing::info!("Terminate command - TODO: Implement");
    println!("TODO: Implement terminate model via API call");
    Ok(())
}
