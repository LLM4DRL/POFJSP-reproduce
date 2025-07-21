"""
Graph Convolutional Neural Network for POFJSP

Implements GraphCNN for processing POFJSP instances as graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Tuple, Dict, List


class GraphCNN(nn.Module):
    """
    Graph Convolutional Neural Network for POFJSP.
    
    Processes scheduling problems as graphs where:
    - Nodes represent operations (job, operation)
    - Edges represent precedence constraints and machine compatibility
    - Features encode processing times and scheduling state
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        pooling: str = "mean",
        dropout: float = 0.1
    ):
        super(GraphCNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for graph-level pooling
            
        Returns:
            Node embeddings [num_nodes, hidden_dim] or graph embedding [batch_size, hidden_dim]
        """
        # Apply graph convolutions
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
            
        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        
        # Apply final projection
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        
        if batch is not None:
            # Graph-level pooling
            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = torch_geometric.nn.global_max_pool(x, batch)
            elif self.pooling == "add":
                x = torch_geometric.nn.global_add_pool(x, batch)
                
        return x
    
    def create_graph_features(
        self,
        problem,
        current_time: float = 0.0,
        machine_ready_times: np.ndarray = None,
        job_ready_times: np.ndarray = None,
        scheduled_ops: set = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create graph representation from POFJSP problem.
        
        Args:
            problem: POFJSP problem instance
            current_time: Current simulation time
            machine_ready_times: When each machine becomes available
            job_ready_times: When each job becomes available
            scheduled_ops: Set of already scheduled operations
            
        Returns:
            x: Node features tensor
            edge_index: Edge indices tensor
            node_masks: Node validity masks
        """
        if machine_ready_times is None:
            machine_ready_times = np.zeros(problem.num_machines)
        if job_ready_times is None:
            job_ready_times = np.zeros(problem.num_jobs)
        if scheduled_ops is None:
            scheduled_ops = set()
            
        # Create node features
        num_ops = problem.total_operations
        node_features = []
        
        op_idx = 0
        for job_idx in range(problem.num_jobs):
            for op_idx_in_job in range(problem.num_operations_per_job[job_idx]):
                op = (job_idx, op_idx_in_job)
                
                # Processing time statistics across compatible machines
                proc_times = problem.processing_times[job_idx][op_idx_in_job, :]
                valid_machines = ~np.isinf(proc_times)
                
                if np.any(valid_machines):
                    min_proc_time = np.min(proc_times[valid_machines])
                    max_proc_time = np.max(proc_times[valid_machines])
                    avg_proc_time = np.mean(proc_times[valid_machines])
                else:
                    min_proc_time = max_proc_time = avg_proc_time = 0.0
                
                # Operation status
                if op in scheduled_ops:
                    status = 2.0  # Scheduled
                elif self._is_operation_ready(op, problem, scheduled_ops):
                    status = 1.0  # Ready
                else:
                    status = 0.0  # Not ready
                
                # Ready times
                job_ready = job_ready_times[job_idx]
                min_machine_ready = np.min(machine_ready_times[valid_machines]) if np.any(valid_machines) else 0.0
                
                node_features.append([
                    min_proc_time,
                    max_proc_time,
                    avg_proc_time,
                    status,
                    job_ready,
                    min_machine_ready,
                    job_idx / problem.num_jobs,  # Normalized job id
                    op_idx_in_job / problem.num_operations_per_job[job_idx],  # Normalized op position
                ])
                
                op_idx += 1
        
        x = torch.FloatTensor(node_features)
        
        # Create edge indices for precedence constraints
        edge_indices = []
        
        for job_idx in range(problem.num_jobs):
            for op_idx_in_job in range(problem.num_operations_per_job[job_idx]):
                op = (job_idx, op_idx_in_job)
                
                # Add precedence edges
                predecessors = problem.predecessors_map.get(op, set())
                for pred in predecessors:
                    pred_idx = self._get_operation_index(pred, problem)
                    curr_idx = self._get_operation_index(op, problem)
                    edge_indices.append([pred_idx, curr_idx])
        
        # Convert to tensor
        if edge_indices:
            edge_index = torch.LongTensor(edge_indices).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return x, edge_index
    
    def _is_operation_ready(self, op: tuple, problem, scheduled_ops: set) -> bool:
        """Check if an operation is ready to be scheduled."""
        predecessors = problem.predecessors_map.get(op, set())
        return all(pred in scheduled_ops for pred in predecessors)
    
    def _get_operation_index(self, op: tuple, problem) -> int:
        """Get the flattened index of an operation."""
        job_idx, op_idx_in_job = op
        index = sum(problem.num_operations_per_job[:job_idx]) + op_idx_in_job
        return index