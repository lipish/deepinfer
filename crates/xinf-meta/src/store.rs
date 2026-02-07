use async_trait::async_trait;
use anyhow::Result;
use crate::watch::WatchStream;

/// Key-value metadata store trait
#[async_trait]
pub trait MetaStore: Send + Sync {
    /// Put a key-value pair
    async fn put(&self, key: &str, value: Vec<u8>) -> Result<()>;
    
    /// Get a value by key
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// Delete a key
    async fn delete(&self, key: &str) -> Result<()>;
    
    /// List all keys with a given prefix
    async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>>;
    
    /// Compare-and-swap operation
    async fn compare_and_swap(
        &self,
        key: &str,
        old_value: Option<Vec<u8>>,
        new_value: Vec<u8>,
    ) -> Result<bool>;
    
    /// Watch for changes on keys with a given prefix
    async fn watch(&self, prefix: &str) -> Result<WatchStream>;
}
