"""
PPO Agent for POFJSP with GNN

Implements Proximal Policy Optimization with Graph Neural Networks
for Partially Ordered Flexible Job Shop Scheduling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random

from src.rl.models.multi_agent import HierarchicalActor, Critic
from src.rl.models.graph_cnn import GraphCNN


class PPOBuffer:
    """
    Buffer for storing PPO rollout data.
    
    Stores observations, actions, rewards, values, log probabilities,
    and masks for valid actions.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Dict[str, Any]) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch from buffer."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        
    def __len__(self) -> int:
        return len(self.buffer)


class PPOAgent:
    """
    PPO Agent with GNN for POFJSP.
    
    Implements Proximal Policy Optimization with:
    - Graph Neural Networks for state representation
    - Hierarchical action selection (Job + Machine)
    - Precedence constraint handling
    - Advantage estimation with GAE
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_jobs: int = 10,
        num_machines: int = 10,
        learning_rate: float = 1e-6,  # Extremely low learning rate
        clip_ratio: float = 0.05,     # Very tight clipping
        value_loss_coef: float = 0.001, # Minimal value loss coefficient
        entropy_coef: float = 0.0001,  # Minimal entropy coefficient
        max_grad_norm: float = 0.01,   # Extremely tight gradient clipping
        gamma: float = 0.99,
        lam: float = 0.95,
        batch_size: int = 64,
        epochs: int = 10,
        device: str = "cpu"
    ):
        """
        Initialize PPO agent.
        
        Args:
            input_dim: Input feature dimension for nodes
            hidden_dim: Hidden dimension for GNN and networks
            num_layers: Number of GNN layers
            num_jobs: Number of jobs in problem
            num_machines: Number of machines in problem
            learning_rate: Learning rate for optimizer
            clip_ratio: PPO clipping ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy regularization coefficient
            max_grad_norm: Maximum gradient norm for clipping
            gamma: Discount factor
            lam: GAE lambda parameter
            batch_size: Training batch size
            epochs: Number of training epochs per update
            device: Device to run on
        """
        self.device = torch.device(device)
        
        # Networks
        self.actor = HierarchicalActor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_jobs=num_jobs,
            num_machines=num_machines
        ).to(self.device)
        
        self.critic = Critic(
            gnn=GraphCNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ),
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        
        # Hyperparameters
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Buffers
        self.buffer = PPOBuffer()
        
    def get_action(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        job_masks: torch.Tensor,
        machine_masks: torch.Tensor,
        processing_times: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices
            job_masks: Valid job masks
            machine_masks: Valid machine masks
            processing_times: Processing times for jobs on machines
            deterministic: Whether to use deterministic policy
            
        Returns:
            job_action: Selected job indices
            machine_action: Selected machine indices
            job_log_prob: Log probabilities for job selection
            machine_log_prob: Log probabilities for machine selection
            value: State value estimates
        """
        with torch.no_grad():
            # Get action distributions
            job_logits, machine_logits = self.actor(
                x, edge_index, batch, job_masks, machine_masks, processing_times
            )
            
            # Job selection
            job_probs = torch.softmax(job_logits, dim=-1)
            if deterministic:
                job_action = torch.argmax(job_probs, dim=-1)
            else:
                job_action = torch.multinomial(job_probs, 1).squeeze(-1)
            job_log_prob = torch.log(job_probs.gather(-1, job_action.unsqueeze(-1))).squeeze(-1)
            
            # Machine selection
            machine_probs = torch.softmax(machine_logits, dim=-1)
            if deterministic:
                machine_action = torch.argmax(machine_probs, dim=-1)
            else:
                machine_action = torch.multinomial(machine_probs, 1).squeeze(-1)
            machine_log_prob = torch.log(machine_probs.gather(-1, machine_action.unsqueeze(-1))).squeeze(-1)
            
            # Get state value
            value = self.critic(x, edge_index, batch).squeeze(-1)
            
            return job_action, machine_action, job_log_prob, machine_log_prob, value
    
    def compute_returns_and_advantages(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute returns and advantages using GAE.
        
        Args:
            rewards: List of rewards
            values: List of state values
            next_value: Next state value
            dones: List of done flags
            
        Returns:
            returns: Computed returns
            advantages: Computed advantages
        """
        returns = []
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[step])
                next_return = next_value
            else:
                next_non_terminal = 1.0 - float(dones[step])
                next_return = returns[0] if returns else next_value
                
            delta = rewards[step] + self.gamma * next_return * next_non_terminal - values[step]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            
        # Don't normalize advantages here - do it per batch during training
        return returns, advantages
    
    def update(self) -> Dict[str, float]:
        """
        Update the agent using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < 8:  # Lower threshold for debugging
            return {}
            
        # Sample batch from buffer
        batch = self.buffer.sample(min(self.batch_size, len(self.buffer)))
        
        # Training metrics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        num_updates = 0
        
        # Process experiences in mini-batches to improve efficiency
        mini_batch_size = min(64, len(batch))  # Process 64 experiences at a time for better GPU utilization
        
        for _ in range(self.epochs):
            # Shuffle batch for each epoch
            import random
            random.shuffle(batch)
            
            for i in range(0, len(batch), mini_batch_size):
                mini_batch = batch[i:i + mini_batch_size]
                
                # Accumulate gradients over mini-batch
                batch_loss = 0
                batch_policy_loss = 0
                batch_value_loss = 0 
                batch_entropy_loss = 0
                
                self.optimizer.zero_grad()
                
                # Collect advantages for normalization
                advantages = []
                for exp in mini_batch:
                    advantages.append(exp['advantage'])
                
                # Normalize advantages
                advantages = np.array(advantages)
                if len(advantages) > 1:
                    adv_mean = np.mean(advantages)
                    adv_std = np.std(advantages) + 1e-8
                    advantages = (advantages - adv_mean) / adv_std
                    # Clip normalized advantages
                    advantages = np.clip(advantages, -10.0, 10.0)
                
                for i, exp in enumerate(mini_batch):
                    # Extract experience data
                    state_x, edge_index, batch_idx = exp['state']
                    job_action = torch.tensor(exp['job_action'], dtype=torch.long, device=self.device).unsqueeze(0)
                    machine_action = torch.tensor(exp['machine_action'], dtype=torch.long, device=self.device).unsqueeze(0)
                    old_job_log_prob = torch.tensor(exp['job_log_prob'], dtype=torch.float32, device=self.device).unsqueeze(0)
                    old_machine_log_prob = torch.tensor(exp['machine_log_prob'], dtype=torch.float32, device=self.device).unsqueeze(0)
                    return_val = torch.tensor(exp['return'], dtype=torch.float32, device=self.device).unsqueeze(0)
                    advantage = torch.tensor(advantages[i] if len(advantages) > 1 else exp['advantage'], dtype=torch.float32, device=self.device).unsqueeze(0)
                    job_mask = exp['job_mask'].unsqueeze(0)
                    machine_mask = exp['machine_mask'].unsqueeze(0)
                    
                    # Ensure tensors are float32
                    state_x = state_x.float().to(self.device)
                    edge_index = edge_index.long().to(self.device)
                    batch_idx = batch_idx.long().to(self.device)
                    job_mask = job_mask.bool().to(self.device)
                    machine_mask = machine_mask.bool().to(self.device)
                    
                    # Get current predictions for this single experience
                    new_job_logits, new_machine_logits = self.actor(
                        state_x, edge_index, batch_idx, job_mask, machine_mask
                    )
                    new_value = self.critic(state_x, edge_index, batch_idx).squeeze(-1)
                    
                    # Job policy loss
                    new_job_probs = torch.softmax(new_job_logits, dim=-1)
                    new_job_log_prob = torch.log(new_job_probs.gather(-1, job_action.unsqueeze(-1))).squeeze(-1)
                    job_ratio = torch.exp(new_job_log_prob - old_job_log_prob)
                    job_surr1 = job_ratio * advantage
                    job_surr2 = torch.clamp(job_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
                    job_policy_loss = -torch.min(job_surr1, job_surr2).mean()
                    
                    # Machine policy loss
                    new_machine_probs = torch.softmax(new_machine_logits, dim=-1)
                    new_machine_log_prob = torch.log(new_machine_probs.gather(-1, machine_action.unsqueeze(-1))).squeeze(-1)
                    machine_ratio = torch.exp(new_machine_log_prob - old_machine_log_prob)
                    machine_surr1 = machine_ratio * advantage
                    machine_surr2 = torch.clamp(machine_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
                    machine_policy_loss = -torch.min(machine_surr1, machine_surr2).mean()
                    
                    # Value loss with clipped targets
                    return_val_clipped = torch.clamp(return_val, -100.0, 100.0)
                    new_value_clipped = torch.clamp(new_value, -100.0, 100.0)
                    value_loss = F.mse_loss(new_value_clipped, return_val_clipped)
                    
                    # Entropy loss
                    job_entropy = -(new_job_probs * torch.log(new_job_probs + 1e-8)).sum(-1).mean()
                    machine_entropy = -(new_machine_probs * torch.log(new_machine_probs + 1e-8)).sum(-1).mean()
                    entropy_loss = -(job_entropy + machine_entropy) / 2
                    
                    # Total loss for this experience
                    loss = (job_policy_loss + machine_policy_loss + \
                           self.value_loss_coef * value_loss + \
                           self.entropy_coef * entropy_loss) / len(mini_batch)  # Scale by mini-batch size
                    
                    # Check for NaN or infinite loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss detected: {loss.item()}")
                        continue
                    
                    # Accumulate gradients
                    loss.backward()
                    
                    # Track metrics
                    batch_loss += loss.item() * len(mini_batch)  # Unscale for metrics
                    batch_policy_loss += (job_policy_loss + machine_policy_loss).item()
                    batch_value_loss += value_loss.item()
                    batch_entropy_loss += entropy_loss.item()
                
                # Update parameters after accumulating gradients from mini-batch
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Add to total metrics
                total_loss += batch_loss
                total_policy_loss += batch_policy_loss
                total_value_loss += batch_value_loss
                total_entropy_loss += batch_entropy_loss
                num_updates += len(mini_batch)
            
        # Clear buffer after update
        self.buffer.clear()
        
        return {
            'total_loss': total_loss / num_updates if num_updates > 0 else 0,
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0,
            'entropy_loss': total_entropy_loss / num_updates if num_updates > 0 else 0
        }
    
    def save(self, path: str) -> None:
        """Save the agent."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str) -> None:
        """Load the agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])