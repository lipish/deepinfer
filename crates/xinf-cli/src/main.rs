use clap::{Parser, Subcommand};
use anyhow::Result;
use tracing_subscriber;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "xinf")]
#[command(about = "xinf - Rust-Native Inference Platform", long_about = None)]
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
    
    let config = xinf_common::Config::load_from_file(&config_path)
        .unwrap_or_default();
    
    // Initialize MetaStore
    let store = Arc::new(xinf_meta::EmbeddedStore::open(
        config.storage.path.unwrap_or_else(|| std::path::PathBuf::from("./data/meta"))
    )?);
    
    // Initialize scheduler
    let snapshot = Arc::new(xinf_scheduler::ClusterSnapshot::new());
    let strategy = Arc::new(xinf_scheduler::IdleFirstStrategy);
    let _scheduler = xinf_scheduler::Scheduler::new(snapshot.clone(), strategy);
    
    // Initialize router
    let endpoint_mgr = Arc::new(xinf_router::EndpointManager::new());
    let router = Arc::new(xinf_router::KvAwareRouter::new(endpoint_mgr));
    
    // Build Axum app
    use axum::{
        routing::{get, post},
        Router,
    };
    use tower_http::trace::TraceLayer;
    
    let app = Router::new()
        .route("/health", get(xinf_gateway::health_check))
        .route("/v1/models", get(xinf_gateway::list_models))
        .route("/v1/chat/completions", post(xinf_gateway::chat_completions))
        .with_state(router)
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
    
    // Initialize MetaStore
    let store = Arc::new(xinf_meta::EmbeddedStore::open("./data/meta")?);
    
    // Create and run agent
    let agent = xinf_agent::WorkerAgent::new(node_id, store);
    agent.run().await;
    
    Ok(())
}

async fn launch(_model: String, _engine: String, _device: Option<String>) -> Result<()> {
    tracing::info!("Launch command - TODO: Implement");
    println!("TODO: Implement model launch via API call to scheduler");
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
