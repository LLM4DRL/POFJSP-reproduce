"""
Multi-Agent Architecture for POFJSP with GNN+PPO

Implements hierarchical multi-agent RL with:
- JobActor: Selects which job to process next
- MachineActor: Selects which machine to use for the selected job
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np
from torch_geometric.data import Data, Batch

from src.rl.models.graph_cnn import GraphCNN


class JobActor(nn.Module):
    """
    Job selection actor using GNN for state representation.
    
    Selects which ready job to process next based on:
    - Current graph state of operations
    - Precedence constraints
    - Available operations
    """
    
    def __init__(
        self,
        gnn: GraphCNN,
        hidden_dim: int = 128,
        num_jobs: int = 10,
        dropout: float = 0.1
    ):
        super(JobActor, self).__init__()
        self.gnn = gnn
        self.hidden_dim = hidden_dim
        self.num_jobs = num_jobs
        
        # Policy network for job selection
        self.job_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer for job selection logits
        self.job_head = nn.Linear(hidden_dim // 2, num_jobs)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        job_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for job selection.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for graph pooling
            job_masks: Mask for valid jobs [batch_size, num_jobs]
            
        Returns:
            Job selection logits [batch_size, num_jobs]
        """
        # Get graph embeddings
        graph_emb = self.gnn(x, edge_index, batch)  # [batch_size, hidden_dim]
        
        # Encode for job selection
        job_features = self.job_encoder(graph_emb)  # [batch_size, hidden_dim//2]
        
        # Get job logits
        job_logits = self.job_head(job_features)  # [batch_size, num_jobs]
        
        # Apply mask to ensure only valid jobs are selected
        job_logits = job_logits.masked_fill(~job_masks.bool(), -float('inf'))
        
        return job_logits


class MachineActor(nn.Module):
    """
    Machine selection actor using GNN for state representation.
    
    Selects which machine to use for a given job based on:
    - Current machine availability
    - Processing times on different machines
    - State of the scheduling system
    """
    
    def __init__(
        self,
        gnn: GraphCNN,
        hidden_dim: int = 128,
        num_machines: int = 10,
        dropout: float = 0.1
    ):
        super(MachineActor, self).__init__()
        self.gnn = gnn
        self.hidden_dim = hidden_dim
        self.num_machines = num_machines
        
        # Policy network for machine selection
        self.machine_encoder = nn.Sequential(
            nn.Linear(hidden_dim + num_machines, hidden_dim),  # +num_machines for job features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer for machine selection logits
        self.machine_head = nn.Linear(hidden_dim // 2, num_machines)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        selected_job: int,
        machine_masks: torch.Tensor,
        processing_times: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for machine selection.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for graph pooling
            selected_job: Selected job index
            machine_masks: Mask for valid machines [batch_size, num_machines]
            processing_times: Processing times for job on machines [batch_size, num_machines]
            
        Returns:
            Machine selection logits [batch_size, num_machines]
        """
        # Get graph embeddings
        graph_emb = self.gnn(x, edge_index, batch)  # [batch_size, hidden_dim]
        
        # Create job-specific features
        job_features = torch.zeros(graph_emb.size(0), self.num_machines, device=graph_emb.device)
        if processing_times is not None:
            # Handle different shapes of processing_times
            if processing_times.dim() == 3 and processing_times.size(1) == 1:
                # If shape is [batch_size, 1, num_machines], squeeze the middle dimension
                job_features = processing_times.squeeze(1)
            elif processing_times.shape == (graph_emb.size(0), self.num_machines):
                job_features = processing_times
            
        # Concatenate graph embedding with job features
        combined_features = torch.cat([graph_emb, job_features], dim=-1)  # [batch_size, hidden_dim + num_machines]
        
        # Encode for machine selection
        machine_features = self.machine_encoder(combined_features)  # [batch_size, hidden_dim//2]
        
        # Get machine logits
        machine_logits = self.machine_head(machine_features)  # [batch_size, num_machines]
        
        # Apply mask to ensure only valid machines are selected
        machine_logits = machine_logits.masked_fill(~machine_masks.bool(), -float('inf'))
        
        return machine_logits


class HierarchicalActor(nn.Module):
    """
    Hierarchical actor combining JobActor and MachineActor.
    
    Implements two-level decision making:
    1. Job selection: Which job to process next
    2. Machine selection: Which machine to use for selected job
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_jobs: int = 10,
        num_machines: int = 10,
        dropout: float = 0.1
    ):
        super(HierarchicalActor, self).__init__()
        
        # Shared GNN backbone
        self.gnn = GraphCNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Individual actors
        self.job_actor = JobActor(
            gnn=self.gnn,
            hidden_dim=hidden_dim,
            num_jobs=num_jobs,
            dropout=dropout
        )
        
        self.machine_actor = MachineActor(
            gnn=self.gnn,
            hidden_dim=hidden_dim,
            num_machines=num_machines,
            dropout=dropout
        )
        
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        job_masks: torch.Tensor,
        machine_masks: torch.Tensor,
        processing_times: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for hierarchical action selection.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for graph pooling
            job_masks: Mask for valid jobs [batch_size, num_jobs]
            machine_masks: Mask for valid machines [batch_size, num_machines]
            processing_times: Processing times [batch_size, num_machines]
            
        Returns:
            job_logits: Job selection logits [batch_size, num_jobs]
            machine_logits: Machine selection logits [batch_size, num_machines]
        """
        # Get job selection logits
        job_logits = self.job_actor(x, edge_index, batch, job_masks)
        
        # Get machine selection logits (conditioned on job selection)
        selected_job_idx = torch.argmax(job_logits, dim=-1)
        
        # Extract processing times for selected job
        if processing_times is not None and processing_times.dim() == 3:
            # processing_times shape: [batch_size, num_jobs, num_machines]
            batch_indices = torch.arange(processing_times.size(0), device=processing_times.device)
            # Clamp selected_job_idx to valid range to avoid index errors
            clamped_job_idx = torch.clamp(selected_job_idx, 0, processing_times.size(1) - 1)
            selected_processing_times = processing_times[batch_indices, clamped_job_idx, :]  # [batch_size, num_machines]
        elif processing_times is not None and processing_times.dim() == 2:
            # If processing_times is [num_jobs, num_machines], we need to select by job and add batch dim
            if processing_times.size(0) > selected_job_idx.max():
                clamped_job_idx = torch.clamp(selected_job_idx, 0, processing_times.size(0) - 1)
                selected_processing_times = processing_times[clamped_job_idx, :].unsqueeze(0)  # [1, num_machines]
            else:
                selected_processing_times = processing_times[:1, :]  # Take first job as fallback
        else:
            # If processing_times is already [batch_size, num_machines] or None, use as is
            selected_processing_times = processing_times
        
        machine_logits = self.machine_actor(
            x, edge_index, batch, 
            selected_job=selected_job_idx,
            machine_masks=machine_masks,
            processing_times=selected_processing_times
        )
        
        return job_logits, machine_logits


class Critic(nn.Module):
    """
    Value function critic for POFJSP.
    
    Estimates the expected return from a given state.
    """
    
    def __init__(
        self,
        gnn: GraphCNN,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super(Critic, self).__init__()
        self.gnn = gnn
        
        # Value network
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for value estimation.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for graph pooling
            
        Returns:
            State values [batch_size, 1]
        """
        # Get graph embedding
        graph_emb = self.gnn(x, edge_index, batch)  # [batch_size, hidden_dim]
        
        # Get state value
        value = self.value_head(graph_emb)  # [batch_size, 1]
        
        return value