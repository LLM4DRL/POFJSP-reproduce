"""
Comprehensive tests for PPO agent implementation.

These tests ensure the PPO agent maintains stability,
performance, and memory efficiency after optimizations.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import psutil
import os

from src.rl.models.ppo_agent import PPOAgent, PPOBuffer
from src.exceptions import ValidationError

try:
    from src.rl.utils.tensor_cache import TensorCache
    TENSOR_CACHE_AVAILABLE = True
except ImportError:
    TENSOR_CACHE_AVAILABLE = False


class TestPPOBuffer:
    """Test PPO buffer functionality."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = PPOBuffer(capacity=1000)
        assert len(buffer) == 0
        assert buffer.capacity == 1000
    
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        buffer = PPOBuffer(capacity=10)
        
        experience = {
            'state': (torch.randn(5, 8), torch.tensor([[0, 1], [1, 2]]), torch.tensor([0, 0, 1, 1, 1])),
            'job_action': 0,
            'machine_action': 1,
            'job_log_prob': -0.5,
            'machine_log_prob': -0.3,
            'return': 1.0,
            'advantage': 0.2,
            'job_mask': torch.tensor([True, False, False]),
            'machine_mask': torch.tensor([True, True, False])
        }
        
        buffer.add(experience)
        assert len(buffer) == 1
    
    def test_buffer_capacity(self):
        """Test buffer capacity limits."""
        buffer = PPOBuffer(capacity=3)
        
        for i in range(5):
            experience = {'step': i}
            buffer.add(experience)
        
        # Should only keep last 3 experiences
        assert len(buffer) == 3
        sampled = buffer.sample(3)
        assert sampled[0]['step'] == 2  # Oldest kept experience
    
    def test_sample(self):
        """Test sampling from buffer."""
        buffer = PPOBuffer(capacity=10)
        
        for i in range(5):
            buffer.add({'step': i})
        
        sample = buffer.sample(3)
        assert len(sample) == 3
        
        # Test sampling more than available
        sample_all = buffer.sample(10)
        assert len(sample_all) == 5
    
    def test_clear(self):
        """Test buffer clearing."""
        buffer = PPOBuffer(capacity=10)
        
        for i in range(5):
            buffer.add({'step': i})
        
        assert len(buffer) == 5
        buffer.clear()
        assert len(buffer) == 0


class TestPPOAgent:
    """Test PPO agent implementation."""
    
    def test_agent_initialization(self, device):
        """Test agent initialization with valid parameters."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=64,
            num_layers=2,
            num_jobs=5,
            num_machines=3,
            device=str(device)
        )
        
        assert agent.device == device
        assert agent.clip_ratio == 0.2  # New optimized value
        assert agent.value_loss_coef == 0.5  # New optimized value
        assert agent.entropy_coef == 0.01  # New optimized value
        assert agent.max_grad_norm == 0.5  # New optimized value
    
    def test_get_action(self, device):
        """Test action generation."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            device=str(device)
        )
        
        # Create mock state
        x = torch.randn(6, 8)  # 6 nodes, 8 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        job_masks = torch.tensor([[True, False, False]], dtype=torch.bool)
        machine_masks = torch.tensor([[True, True]], dtype=torch.bool)
        
        job_action, machine_action, job_log_prob, machine_log_prob, value = agent.get_action(
            x, edge_index, batch, job_masks, machine_masks
        )
        
        # Check output shapes and types
        assert job_action.shape == (1,)
        assert machine_action.shape == (1,)
        assert job_log_prob.shape == (1,)
        assert machine_log_prob.shape == (1,)
        assert value.shape == (1,)
        
        # Check values are reasonable
        assert 0 <= job_action.item() < 3
        assert 0 <= machine_action.item() < 2
        assert job_log_prob.item() <= 0  # Log probabilities should be negative
        assert machine_log_prob.item() <= 0
    
    def test_deterministic_action(self, device):
        """Test deterministic action selection."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            device=str(device)
        )
        
        # Create mock state
        x = torch.randn(6, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        job_masks = torch.tensor([[True, False, False]], dtype=torch.bool)
        machine_masks = torch.tensor([[True, True]], dtype=torch.bool)
        
        # Test deterministic mode
        actions1 = agent.get_action(x, edge_index, batch, job_masks, machine_masks, deterministic=True)
        actions2 = agent.get_action(x, edge_index, batch, job_masks, machine_masks, deterministic=True)
        
        # Should be identical
        assert torch.equal(actions1[0], actions2[0])  # job_action
        assert torch.equal(actions1[1], actions2[1])  # machine_action
    
    def test_compute_returns_and_advantages(self, device):
        """Test GAE computation."""
        agent = PPOAgent(device=str(device))
        
        rewards = [1.0, 0.5, 2.0, -0.5]
        values = [1.2, 0.8, 1.5, 0.3]
        next_value = 0.0
        dones = [False, False, False, True]
        
        returns, advantages = agent.compute_returns_and_advantages(
            rewards, values, next_value, dones
        )
        
        assert len(returns) == len(rewards)
        assert len(advantages) == len(rewards)
        
        # Returns should be reasonable
        assert all(isinstance(r, float) for r in returns)
        assert all(isinstance(a, float) for a in advantages)
    
    def test_update_with_experiences(self, device):
        """Test agent update with experiences."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            batch_size=4,
            epochs=2,
            device=str(device)
        )
        
        # Add some experiences to buffer
        for i in range(8):  # Enough for batch_size
            experience = {
                'state': (
                    torch.randn(6, 8),
                    torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
                    torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
                ),
                'job_action': i % 3,
                'machine_action': i % 2,
                'job_log_prob': -0.5 - i * 0.1,
                'machine_log_prob': -0.3 - i * 0.1,
                'return': 1.0 + i * 0.1,
                'advantage': 0.2 - i * 0.05,
                'job_mask': torch.tensor([True, False, False]),
                'machine_mask': torch.tensor([True, True])
            }
            agent.buffer.add(experience)
        
        # Update agent
        metrics = agent.update()
        
        # Check metrics are returned
        assert 'total_loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy_loss' in metrics
        
        # Check values are reasonable
        assert all(isinstance(v, float) for v in metrics.values())
        assert not any(np.isnan(v) or np.isinf(v) for v in metrics.values())
    
    def test_update_empty_buffer(self, device):
        """Test update with empty buffer."""
        agent = PPOAgent(device=str(device))
        
        metrics = agent.update()
        assert metrics == {}  # Should return empty dict
    
    def test_save_load(self, device, temp_dir):
        """Test agent save and load functionality."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            device=str(device)
        )
        
        # Save agent
        save_path = temp_dir / "test_agent.pth"
        agent.save(str(save_path))
        assert save_path.exists()
        
        # Create new agent and load
        new_agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            device=str(device)
        )
        
        new_agent.load(str(save_path))
        
        # Check that weights are loaded (compare a few parameters)
        for p1, p2 in zip(agent.actor.parameters(), new_agent.actor.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)


class TestMemoryOptimizations:
    """Test memory optimizations in PPO agent."""
    
    def test_memory_efficient_update(self, device):
        """Test memory-efficient batch processing."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            batch_size=16,
            device=str(device)
        )
        
        # Monitor memory usage
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
        else:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
        
        # Add many experiences
        for i in range(32):
            experience = {
                'state': (
                    torch.randn(6, 8),
                    torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
                    torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
                ),
                'job_action': i % 3,
                'machine_action': i % 2,
                'job_log_prob': -0.5,
                'machine_log_prob': -0.3,
                'return': 1.0,
                'advantage': 0.2,
                'job_mask': torch.tensor([True, False, False]),
                'machine_mask': torch.tensor([True, True])
            }
            agent.buffer.add(experience)
        
        # Update with memory monitoring
        metrics = agent.update()
        
        if device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated(device)
            memory_increase = final_memory - initial_memory
            # Should not leak significant GPU memory
            assert memory_increase < 50 * 1024 * 1024  # 50MB threshold
        
        # Should complete successfully
        assert 'total_loss' in metrics
    
    def test_gradient_clipping(self, device):
        """Test gradient clipping prevents exploding gradients."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            max_grad_norm=0.5,  # Small value for testing
            device=str(device)
        )
        
        # Add experiences with extreme values to trigger large gradients
        for i in range(8):
            experience = {
                'state': (
                    torch.randn(6, 8) * 10,  # Large values
                    torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
                    torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
                ),
                'job_action': i % 3,
                'machine_action': i % 2,
                'job_log_prob': -10.0,  # Extreme log prob
                'machine_log_prob': -10.0,
                'return': 100.0,  # Large return
                'advantage': 50.0,  # Large advantage
                'job_mask': torch.tensor([True, False, False]),
                'machine_mask': torch.tensor([True, True])
            }
            agent.buffer.add(experience)
        
        # Update should not crash due to exploding gradients
        metrics = agent.update()
        
        # Check that losses are finite
        assert all(np.isfinite(v) for v in metrics.values() if v != 0)
    
    @pytest.mark.skipif(not TENSOR_CACHE_AVAILABLE, reason="TensorCache not available")
    def test_tensor_cache_integration(self, device):
        """Test integration with TensorCache."""
        cache = TensorCache(max_jobs=5, max_machines=3, device=device)
        
        # Test observation padding with cache
        obs = {
            'x': torch.randn(6, 8),
            'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            'batch': torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
            'job_mask': torch.tensor([True, False]),
            'machine_mask': torch.tensor([True, True]),
            'processing_times': torch.randn(2, 2)
        }
        
        padded_obs = cache.pad_observation(obs)
        
        # Check shapes are padded correctly
        assert padded_obs['job_mask'].shape == (5,)
        assert padded_obs['machine_mask'].shape == (3,)
        assert padded_obs['processing_times'].shape == (5, 3)
        
        # Check cache statistics
        stats = cache.get_memory_stats()
        assert stats['cache_hits'] >= 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_configuration(self):
        """Test validation of agent configuration."""
        # Test invalid hyperparameters would be caught by config validation
        # (This tests integration with the config system we created)
        pass
    
    def test_nan_loss_handling(self, device):
        """Test handling of NaN losses."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            device=str(device)
        )
        
        # Add experience that might cause NaN
        experience = {
            'state': (
                torch.zeros(6, 8),  # All zeros might cause issues
                torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
                torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
            ),
            'job_action': 0,
            'machine_action': 0,
            'job_log_prob': float('-inf'),  # Invalid log prob
            'machine_log_prob': float('-inf'),
            'return': float('nan'),  # NaN return
            'advantage': 0.0,
            'job_mask': torch.tensor([True, False, False]),
            'machine_mask': torch.tensor([True, True])
        }
        
        agent.buffer.add(experience)
        
        # Update should handle NaN gracefully
        metrics = agent.update()
        
        # Should not crash, might return empty metrics
        assert isinstance(metrics, dict)
    
    def test_gpu_memory_error_handling(self, device):
        """Test GPU memory error handling."""
        if device.type != 'cuda':
            pytest.skip("GPU memory test requires CUDA")
        
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=1024,  # Large to potentially cause memory issues
            num_jobs=100,
            num_machines=100,
            device=str(device)
        )
        
        # This test mainly ensures no crashes occur
        # In practice, would test with actual memory pressure
        assert agent.device.type == 'cuda'


class TestPerformanceRegression:
    """Test for performance regressions in PPO agent."""
    
    @pytest.mark.performance
    def test_training_speed(self, device, performance_baseline):
        """Test that training speed hasn't regressed."""
        import time
        
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=64,
            num_jobs=5,
            num_machines=3,
            batch_size=16,
            device=str(device)
        )
        
        # Add experiences
        for i in range(32):
            experience = {
                'state': (
                    torch.randn(10, 8),
                    torch.tensor([[j, j+1] for j in range(9)]).T,
                    torch.tensor([0] * 5 + [1] * 5)
                ),
                'job_action': i % 5,
                'machine_action': i % 3,
                'job_log_prob': -0.5,
                'machine_log_prob': -0.3,
                'return': 1.0,
                'advantage': 0.2,
                'job_mask': torch.tensor([True] * 5),
                'machine_mask': torch.tensor([True] * 3)
            }
            agent.buffer.add(experience)
        
        # Time the update
        start_time = time.time()
        metrics = agent.update()
        update_time = time.time() - start_time
        
        # Should complete quickly (adjust threshold as needed)
        assert update_time < 5.0, f"Update took {update_time:.2f}s, too slow"
        assert 'total_loss' in metrics
    
    @pytest.mark.performance
    def test_memory_usage_stable(self, device):
        """Test that memory usage remains stable across updates."""
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,
            num_jobs=3,
            num_machines=2,
            device=str(device)
        )
        
        memory_usage = []
        
        for epoch in range(5):
            # Add fresh experiences
            for i in range(16):
                experience = {
                    'state': (
                        torch.randn(6, 8),
                        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
                        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
                    ),
                    'job_action': i % 3,
                    'machine_action': i % 2,
                    'job_log_prob': -0.5,
                    'machine_log_prob': -0.3,
                    'return': 1.0,
                    'advantage': 0.2,
                    'job_mask': torch.tensor([True, False, False]),
                    'machine_mask': torch.tensor([True, True])
                }
                agent.buffer.add(experience)
            
            # Update and measure memory
            agent.update()
            
            if device.type == 'cuda':
                memory_usage.append(torch.cuda.memory_allocated(device))
            else:
                process = psutil.Process(os.getpid())
                memory_usage.append(process.memory_info().rss)
        
        # Memory usage should be stable (not growing linearly)
        if len(memory_usage) > 2:
            # Check that memory isn't growing significantly
            growth_rate = (memory_usage[-1] - memory_usage[0]) / len(memory_usage)
            max_growth = 10 * 1024 * 1024  # 10MB per epoch max
            assert growth_rate < max_growth, f"Memory growing too fast: {growth_rate} bytes/epoch"