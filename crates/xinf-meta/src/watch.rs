use tokio::sync::broadcast;
use tokio_stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Watch event types
#[derive(Debug, Clone)]
pub enum WatchEvent {
    Put { key: String, value: Vec<u8> },
    Delete { key: String },
}

/// Stream of watch events
pub struct WatchStream {
    prefix: String,
    rx: broadcast::Receiver<WatchEvent>,
}

impl WatchStream {
    pub fn new(prefix: String, rx: broadcast::Receiver<WatchEvent>) -> Self {
        Self { prefix, rx }
    }
}

impl Stream for WatchStream {
    type Item = WatchEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.rx.try_recv() {
                Ok(event) => {
                    // Filter by prefix
                    let key = match &event {
                        WatchEvent::Put { key, .. } => key,
                        WatchEvent::Delete { key } => key,
                    };
                    
                    if key.starts_with(&self.prefix) {
                        return Poll::Ready(Some(event));
                    }
                    // Continue to next event if prefix doesn't match
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    // No event ready, register waker
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    return Poll::Ready(None);
                }
                Err(broadcast::error::TryRecvError::Lagged(_)) => {
                    // Skip lagged messages
                    continue;
                }
            }
        }
    }
}
