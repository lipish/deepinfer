use crate::store::MetaStore;
use crate::watch::{WatchStream, WatchEvent};
use anyhow::Result;
use async_trait::async_trait;
use sled::Db;
use std::path::Path;
use tokio::sync::broadcast;
use tracing::info;

/// Embedded sled-based metadata store for single-machine mode
pub struct EmbeddedStore {
    db: Db,
    watch_tx: broadcast::Sender<WatchEvent>,
}

impl EmbeddedStore {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = sled::open(path)?;
        let (watch_tx, _) = broadcast::channel(1000);
        
        info!("Opened embedded metadata store");
        
        Ok(Self { db, watch_tx })
    }
}

#[async_trait]
impl MetaStore for EmbeddedStore {
    async fn put(&self, key: &str, value: Vec<u8>) -> Result<()> {
        self.db.insert(key.as_bytes(), value.clone())?;
        
        // Notify watchers
        let _ = self.watch_tx.send(WatchEvent::Put {
            key: key.to_string(),
            value,
        });
        
        Ok(())
    }
    
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.db.get(key.as_bytes())?.map(|v| v.to_vec()))
    }
    
    async fn delete(&self, key: &str) -> Result<()> {
        self.db.remove(key.as_bytes())?;
        
        // Notify watchers
        let _ = self.watch_tx.send(WatchEvent::Delete {
            key: key.to_string(),
        });
        
        Ok(())
    }
    
    async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let prefix_bytes = prefix.as_bytes();
        
        for item in self.db.scan_prefix(prefix_bytes) {
            let (k, _) = item?;
            if let Ok(key_str) = String::from_utf8(k.to_vec()) {
                keys.push(key_str);
            }
        }
        
        Ok(keys)
    }
    
    async fn compare_and_swap(
        &self,
        key: &str,
        old_value: Option<Vec<u8>>,
        new_value: Vec<u8>,
    ) -> Result<bool> {
        let result = self.db.compare_and_swap(
            key.as_bytes(),
            old_value.as_deref(),
            Some(new_value.clone()),
        )?;
        
        let success = result.is_ok();
        
        if success {
            // Notify watchers
            let _ = self.watch_tx.send(WatchEvent::Put {
                key: key.to_string(),
                value: new_value,
            });
        }
        
        Ok(success)
    }
    
    async fn watch(&self, prefix: &str) -> Result<WatchStream> {
        let rx = self.watch_tx.subscribe();
        Ok(WatchStream::new(prefix.to_string(), rx))
    }
}
