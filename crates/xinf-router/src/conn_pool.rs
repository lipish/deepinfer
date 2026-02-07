/// Connection pool stub
/// TODO: Implement gRPC/HTTP connection pooling

pub struct ConnectionPool;

impl ConnectionPool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}
