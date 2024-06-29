
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct AudioMessage {
    pub timestamp: DateTime<Utc>,
    pub data: Arc<[f32]>,
}

impl AudioMessage {
    pub fn new(data: Arc<[f32]>) -> Self {
        Self {
            timestamp: Utc::now(),
            data,
        }
    }
}


