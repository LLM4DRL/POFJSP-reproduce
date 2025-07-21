#!/usr/bin/env python3
"""
Quick test script for the RL pipeline to ensure everything works.
"""

import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.problems.problem_instance import ProblemInstance
        from src.rl.environments.pofjsp_env import POFJSPEnv
        from src.rl.models.ppo_agent import PPOAgent
        from src.algorithms.iaoa_gns import IAOAGNSAlgorithm
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_device_detection():
    """Test device detection."""
    print("\\nTesting device detection...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Detected device: {device}")
    
    if device == "cuda":
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Running on CPU (CUDA not available)")
    
    return device

def create_test_problem():
    """Create a small test problem."""
    print("\\nCreating test problem...")
    
    # Small 3x3 problem
    import numpy as np
    num_operations_per_job = [2, 2, 2]
    processing_times = [
        np.array([[3, 5, 7], [6, 4, 8]]),      # Job 0
        np.array([[4, 2, 6], [5, 7, 3]]),      # Job 1  
        np.array([[2, 8, 4], [7, 3, 5]])       # Job 2
    ]
    
    from src.problems.problem_instance import Operation, ProblemInstance
    
    # Simple precedence: each job's second operation depends on first
    predecessors_map = {
        Operation(0, 1): {Operation(0, 0)},
        Operation(1, 1): {Operation(1, 0)},
        Operation(2, 1): {Operation(2, 0)},
        Operation(0, 0): set(),
        Operation(1, 0): set(),
        Operation(2, 0): set()
    }
    
    successors_map = {
        Operation(0, 0): {Operation(0, 1)},
        Operation(1, 0): {Operation(1, 1)},
        Operation(2, 0): {Operation(2, 1)},
        Operation(0, 1): set(),
        Operation(1, 1): set(),
        Operation(2, 1): set()
    }
    
    problem = ProblemInstance(
        num_jobs=3,
        num_machines=3,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )
    
    print(f"✓ Created test problem: {problem.num_jobs}J x {problem.num_machines}M, "
          f"{problem.total_operations} operations")
    
    return problem

def test_rl_components(problem, device):
    """Test RL components."""
    print("\\nTesting RL components...")
    
    try:
        from src.rl.environments.pofjsp_env import POFJSPEnv
        from src.rl.models.ppo_agent import PPOAgent
        
        # Test environment
        env = POFJSPEnv(problem, time_limit=20)
        obs = env.reset()
        print(f"✓ Environment created, initial observation shape: {obs['node_features'].shape}")
        
        # Test agent
        agent = PPOAgent(
            input_dim=8,
            hidden_dim=32,  # Small for testing
            num_layers=2,
            num_jobs=problem.num_jobs,
            num_machines=problem.num_machines,
            learning_rate=1e-3,
            batch_size=4,
            device=device
        )
        print(f"✓ PPO agent created on {device}")
        
        # Test action selection
        action = agent.select_action(obs)
        print(f"✓ Action selected: {action}")
        
        # Test environment step
        next_obs, reward, done, info = env.step(action)
        print(f"✓ Environment step: reward={reward:.2f}, done={done}")
        
        return True
        
    except Exception as e:
        print(f"✗ RL component test failed: {e}")
        return False

def test_traditional_algorithm(problem):
    """Test traditional algorithm."""
    print("\\nTesting traditional algorithm...")
    
    try:
        from src.algorithms.iaoa_gns import IAOAGNSAlgorithm
        
        algorithm = IAOAGNSAlgorithm(pop_size=10, max_iterations=5)
        solution = algorithm.solve(problem)
        
        print(f"✓ IAOA+GNS completed, makespan: {solution.makespan:.1f}")
        return True
        
    except Exception as e:
        print(f"✗ Traditional algorithm test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== RL Pipeline Test ===")
    
    # Test imports
    if not test_imports():
        return False
    
    # Test device
    device = test_device_detection()
    
    # Create test problem
    problem = create_test_problem()
    
    # Test RL components
    if not test_rl_components(problem, device):
        return False
    
    # Test traditional algorithm
    if not test_traditional_algorithm(problem):
        return False
    
    print("\\n✓ All tests passed! Pipeline is ready.")
    print("\\nNext steps:")
    print("1. Generate flexible dataset: python generate_flexible_dataset.py --custom-config")
    print("2. Run comparison: python run_comparison.py --generate-data --rl-timesteps 10000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)