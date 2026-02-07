use async_trait::async_trait;
use anyhow::{Result, anyhow};
use etcd_client::{Client, GetOptions, WatchOptions, EventType};
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::store::MetaStore;

pub struct EtcdStore {
    client: Arc<tokio::sync::Mutex<Client>>,
    watchers: Arc<dashmap::DashMap<String, broadcast::Sender<Vec<u8>>>>,
}

impl EtcdStore {
    pub async fn connect(endpoints: Vec<String>) -> Result<Self> {
        let client = Client::connect(endpoints, None).await
            .map_err(|e| anyhow!("Failed to connect to etcd: {}", e))?;
        
        Ok(Self {
            client: Arc::new(tokio::sync::Mutex::new(client)),
            watchers: Arc::new(dashmap::DashMap::new()),
        })
    }
}

#[async_trait]
impl MetaStore for EtcdStore {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut client = self.client.lock().await;
        let resp = client.get(key, None).await
            .map_err(|e| anyhow!("etcd get failed: {}", e))?;
        
        Ok(resp.kvs().first().map(|kv| kv.value().to_vec()))
    }
    
    async fn put(&self, key: &str, value: Vec<u8>) -> Result<()> {
        let mut client = self.client.lock().await;
        client.put(key, value, None).await
            .map_err(|e| anyhow!("etcd put failed: {}", e))?;
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<()> {
        let mut client = self.client.lock().await;
        client.delete(key, None).await
            .map_err(|e| anyhow!("etcd delete failed: {}", e))?;
        Ok(())
    }
    
    async fn list(&self, prefix: &str) -> Result<Vec<(String, Vec<u8>)>> {
        let mut client = self.client.lock().await;
        let resp = client.get(prefix, Some(GetOptions::new().with_prefix())).await
            .map_err(|e| anyhow!("etcd list failed: {}", e))?;
        
        let results = resp.kvs().iter()
            .map(|kv| (
                String::from_utf8_lossy(kv.key()).to_string(),
                kv.value().to_vec()
            ))
            .collect();
        
        Ok(results)
    }
    
    async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let mut client = self.client.lock().await;
        let resp = client.get(prefix, Some(GetOptions::new().with_prefix())).await
            .map_err(|e| anyhow!("etcd list_prefix failed: {}", e))?;
        
        let results = resp.kvs().iter()
            .map(|kv| String::from_utf8_lossy(kv.key()).to_string())
            .collect();
        
        Ok(results)
    }
    
    async fn compare_and_swap(
        &self,
        key: &str,
        old_value: Option<Vec<u8>>,
        new_value: Vec<u8>,
    ) -> Result<bool> {
        use etcd_client::{Compare, CompareOp, TxnOp, Txn};
        
        let mut client = self.client.lock().await;
        
        // Build transaction based on old_value
        let compare = if let Some(old) = old_value {
            Compare::value(key, CompareOp::Equal, old)
        } else {
            // Check key doesn't exist (version = 0)
            Compare::version(key, CompareOp::Equal, 0)
        };
        
        let put_op = TxnOp::put(key, new_value, None);
        let txn = Txn::new()
            .when(vec![compare])
            .and_then(vec![put_op])
            .or_else(vec![]);
        
        let resp = client.txn(txn).await
            .map_err(|e| anyhow!("etcd compare_and_swap failed: {}", e))?;
        
        Ok(resp.succeeded())
    }
    
    async fn watch(&self, prefix: &str) -> Result<broadcast::Receiver<Vec<u8>>> {
        let prefix_owned = prefix.to_string();
        
        // Check if watcher already exists
        if let Some(tx) = self.watchers.get(&prefix_owned) {
            return Ok(tx.subscribe());
        }
        
        // Create new watcher
        let (tx, rx) = broadcast::channel(100);
        self.watchers.insert(prefix_owned.clone(), tx.clone());
        
        let client = self.client.clone();
        let watchers = self.watchers.clone();
        let prefix_for_task = prefix_owned.clone();
        
        tokio::spawn(async move {
            let mut client_guard = client.lock().await;
            let (_watcher, mut stream) = match client_guard
                .watch(prefix_for_task.clone(), Some(WatchOptions::new().with_prefix()))
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Failed to create etcd watch: {}", e);
                    return;
                }
            };
            drop(client_guard);
            
            loop {
                match stream.message().await {
                    Ok(Some(resp)) => {
                        for event in resp.events() {
                            if event.event_type() == EventType::Put {
                                if let Some(kv) = event.kv() {
                                    let value = kv.value().to_vec();
                                    if tx.send(value).is_err() {
                                        // No active receivers, stop watching
                                        watchers.remove(&prefix_for_task);
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    Ok(None) => {
                        tracing::info!("etcd watch stream closed");
                        break;
                    }
                    Err(e) => {
                        tracing::error!("etcd watch error: {}", e);
                        break;
                    }
                }
            }
            
            watchers.remove(&prefix_for_task);
        });
        
        Ok(rx)
    }
}
