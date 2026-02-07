/// Middleware to extract ExecutionContext from requests

use xinf_common::ExecutionContext;
use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};

pub async fn extract_context(
    mut request: Request,
    next: Next,
) -> Response {
    // Create ExecutionContext from request headers
    let ctx = ExecutionContext::default();
    
    // TODO: Extract from headers:
    // - X-Request-ID
    // - X-User-ID
    // - X-Priority
    // - X-Session-ID
    
    request.extensions_mut().insert(ctx);
    
    next.run(request).await
}
