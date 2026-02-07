use async_trait::async_trait;
use xinf_common::{PlacementRequest, PlacementPlan, ReplicaAssignment, ReplicaRole};
use xinf_device::NodeDeviceInfo;
use crate::state::ClusterSnapshot;
use anyhow::{anyhow, Result};
use uuid::Uuid;

/// Placement strategy trait
#[async_trait]
pub trait PlacementStrategy: Send + Sync {
    async fn place(
        &self,
        request: &PlacementRequest,
        snapshot: &ClusterSnapshot,
    ) -> Result<PlacementPlan>;
}

/// Idle-first placement strategy
pub struct IdleFirstStrategy;

#[async_trait]
impl PlacementStrategy for IdleFirstStrategy {
    async fn place(
        &self,
        request: &PlacementRequest,
        snapshot: &ClusterSnapshot,
    ) -> Result<PlacementPlan> {
        let nodes = snapshot.get_all_nodes();
        
        if nodes.is_empty() {
            return Err(anyhow!("No nodes available"));
        }
        
        let mut assignments = Vec::new();
        
        // Simple strategy: place all replicas on the first available node
        // TODO: Implement more sophisticated placement logic
        for i in 0..request.replicas {
            if let Some((node_id, node_info)) = nodes.first() {
                let role = if request.replicas == 1 {
                    ReplicaRole::Standalone
                } else if i == 0 {
                    ReplicaRole::Primary
                } else {
                    ReplicaRole::Secondary
                };
                
                // Select devices based on GPU requirement
                let device_indices = select_devices(node_info, &request.gpu_requirement)?;
                
                assignments.push(ReplicaAssignment {
                    replica_id: Uuid::new_v4(),
                    node_id: node_id.clone(),
                    device_indices,
                    role,
                });
            }
        }
        
        Ok(PlacementPlan {
            plan_id: Uuid::new_v4(),
            model_name: request.model_name.clone(),
            assignments,
            created_at: chrono::Utc::now(),
        })
    }
}

fn select_devices(
    node_info: &NodeDeviceInfo,
    gpu_req: &Option<xinf_common::GpuRequirement>,
) -> Result<Vec<u32>> {
    if let Some(req) = gpu_req {
        // Filter devices by requirement
        let suitable: Vec<u32> = node_info.devices.iter()
            .filter(|d| {
                d.total_memory_mb >= (req.min_memory_gb as u64 * 1024)
                    && req.compute_capability.map_or(true, |cc| {
                        d.compute_capability.map_or(false, |dcc| dcc >= cc)
                    })
            })
            .take(req.count as usize)
            .map(|d| d.index)
            .collect();
        
        if suitable.len() < req.count as usize {
            return Err(anyhow!("Insufficient suitable devices"));
        }
        
        Ok(suitable)
    } else {
        // No GPU requirement, use first device
        Ok(vec![0])
    }
}
