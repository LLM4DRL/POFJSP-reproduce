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
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
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
        Update the agent using PPO with improved memory management.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.batch_size // 2:
            return {}
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_data = self._prepare_batch_data()
        if not batch_data:
            return {}
        
        metrics = {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
        num_updates = 0
        
        for epoch in range(self.epochs):
            epoch_metrics = self._update_epoch(batch_data)
            for key in metrics:
                metrics[key] += epoch_metrics.get(key, 0)
            num_updates += epoch_metrics.get('num_updates', 0)
        
        # Normalize metrics
        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        
        self.buffer.clear()
        return metrics
    
    def _prepare_batch_data(self) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare batch data efficiently."""
        try:
            batch = self.buffer.sample(min(self.batch_size, len(self.buffer)))
            
            # Extract all data at once
            states_x = []
            edge_indices = []
            batch_indices = []
            job_actions = []
            machine_actions = []
            old_job_log_probs = []
            old_machine_log_probs = []
            returns = []
            advantages = []
            job_masks = []
            machine_masks = []
            
            for exp in batch:
                state_x, edge_index, batch_idx = exp['state']
                states_x.append(state_x.float())
                edge_indices.append(edge_index.long())
                batch_indices.append(batch_idx.long())
                job_actions.append(exp['job_action'])
                machine_actions.append(exp['machine_action'])
                old_job_log_probs.append(exp['job_log_prob'])
                old_machine_log_probs.append(exp['machine_log_prob'])
                returns.append(exp['return'])
                advantages.append(exp['advantage'])
                job_masks.append(exp['job_mask'])
                machine_masks.append(exp['machine_mask'])
            
            # Normalize advantages
            advantages = np.array(advantages)
            if len(advantages) > 1 and np.std(advantages) > 1e-8:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
                advantages = np.clip(advantages, -10.0, 10.0)
            
            # Convert to tensors efficiently
            return {
                'states_x': states_x,
                'edge_indices': edge_indices,
                'batch_indices': batch_indices,
                'job_actions': torch.tensor(job_actions, dtype=torch.long, device=self.device),
                'machine_actions': torch.tensor(machine_actions, dtype=torch.long, device=self.device),
                'old_job_log_probs': torch.tensor(old_job_log_probs, dtype=torch.float32, device=self.device),
                'old_machine_log_probs': torch.tensor(old_machine_log_probs, dtype=torch.float32, device=self.device),
                'returns': torch.tensor(returns, dtype=torch.float32, device=self.device),
                'advantages': torch.tensor(advantages, dtype=torch.float32, device=self.device),
                'job_masks': job_masks,
                'machine_masks': machine_masks
            }
        except Exception as e:
            print(f"Error preparing batch data: {e}")
            return None
    
    def _update_epoch(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update for one epoch with efficient batching."""
        metrics = {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0, 'num_updates': 0}
        
        batch_size = len(batch_data['job_actions'])
        indices = torch.randperm(batch_size, device=self.device)
        
        mini_batch_size = min(32, batch_size)  # Smaller batches for memory efficiency
        
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, batch_size)
            mini_indices = indices[start_idx:end_idx]
            
            self.optimizer.zero_grad()
            
            try:
                loss_info = self._compute_loss_batch(batch_data, mini_indices)
                if loss_info is None:
                    continue
                
                loss, policy_loss, value_loss, entropy_loss = loss_info
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Update metrics
                batch_len = len(mini_indices)
                metrics['total_loss'] += loss.item() * batch_len
                metrics['policy_loss'] += policy_loss.item() * batch_len
                metrics['value_loss'] += value_loss.item() * batch_len
                metrics['entropy_loss'] += entropy_loss.item() * batch_len
                metrics['num_updates'] += batch_len
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"GPU memory error, skipping batch: {e}")
                    continue
                else:
                    raise e
        
        return metrics
    
    def _compute_loss_batch(self, batch_data: Dict, indices: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute loss for a mini-batch efficiently."""
        try:
            batch_len = len(indices)
            
            # Prepare batch tensors
            job_actions = batch_data['job_actions'][indices]
            machine_actions = batch_data['machine_actions'][indices]
            old_job_log_probs = batch_data['old_job_log_probs'][indices]
            old_machine_log_probs = batch_data['old_machine_log_probs'][indices]
            returns = batch_data['returns'][indices]
            advantages = batch_data['advantages'][indices]
            
            # Process states in batch
            total_job_policy_loss = 0
            total_machine_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            
            for i, idx in enumerate(indices):
                state_x = batch_data['states_x'][idx].to(self.device)
                edge_index = batch_data['edge_indices'][idx].to(self.device)
                batch_idx = batch_data['batch_indices'][idx].to(self.device)
                job_mask = batch_data['job_masks'][idx].bool().to(self.device)
                machine_mask = batch_data['machine_masks'][idx].bool().to(self.device)
                
                # Forward pass
                job_logits, machine_logits = self.actor(
                    state_x, edge_index, batch_idx, job_mask.unsqueeze(0), machine_mask.unsqueeze(0)
                )
                value = self.critic(state_x, edge_index, batch_idx).squeeze(-1)
                
                # Policy losses
                job_probs = torch.softmax(job_logits, dim=-1)
                machine_probs = torch.softmax(machine_logits, dim=-1)
                
                new_job_log_prob = torch.log(job_probs.gather(-1, job_actions[i].unsqueeze(-1))).squeeze(-1)
                new_machine_log_prob = torch.log(machine_probs.gather(-1, machine_actions[i].unsqueeze(-1))).squeeze(-1)
                
                job_ratio = torch.exp(new_job_log_prob - old_job_log_probs[i])
                machine_ratio = torch.exp(new_machine_log_prob - old_machine_log_probs[i])
                
                adv = advantages[i]
                job_surr1 = job_ratio * adv
                job_surr2 = torch.clamp(job_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                machine_surr1 = machine_ratio * adv
                machine_surr2 = torch.clamp(machine_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                
                total_job_policy_loss += -torch.min(job_surr1, job_surr2)
                total_machine_policy_loss += -torch.min(machine_surr1, machine_surr2)
                
                # Value loss
                total_value_loss += F.mse_loss(value.squeeze(), returns[i])
                
                # Entropy loss
                job_entropy = -(job_probs * torch.log(job_probs + 1e-8)).sum(-1)
                machine_entropy = -(machine_probs * torch.log(machine_probs + 1e-8)).sum(-1)
                total_entropy_loss += -(job_entropy + machine_entropy) / 2
            
            # Average losses
            policy_loss = (total_job_policy_loss + total_machine_policy_loss) / batch_len
            value_loss = total_value_loss / batch_len
            entropy_loss = total_entropy_loss / batch_len
            
            total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            return total_loss, policy_loss, value_loss, entropy_loss
            
        except Exception as e:
            print(f"Error computing batch loss: {e}")
            return None
    
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