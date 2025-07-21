#!/usr/bin/env python3
"""
Demo script showing RL vs Traditional algorithms on POFJSP.

This script demonstrates the complete pipeline with auto device detection.
"""

import sys
import os
import torch
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.problems.problem_instance import ProblemInstance, Operation
from src.algorithms.iaoa_gns import IAOAGNSAlgorithm

def create_demo_problem():
    """Create a demonstration POFJSP problem."""
    print("Creating demo problem (5 jobs, 4 machines)...")
    
    # Problem configuration
    num_jobs = 5
    num_machines = 4
    num_operations_per_job = [3, 2, 4, 3, 2]  # Variable operations per job
    
    # Processing times (job_id, operation_id, machine_id)
    processing_times = [
        # Job 0: 3 operations
        np.array([[5, 3, 7, 4],    # Operation 0
                  [6, 8, 2, 5],    # Operation 1  
                  [4, 6, 9, 3]]),  # Operation 2
        # Job 1: 2 operations
        np.array([[7, 5, 6, 8],    # Operation 0
                  [3, 9, 4, 6]]),  # Operation 1
        # Job 2: 4 operations  
        np.array([[8, 4, 5, 7],    # Operation 0
                  [6, 7, 3, 9],    # Operation 1
                  [5, 8, 6, 4],    # Operation 2
                  [9, 3, 7, 5]]),  # Operation 3
        # Job 3: 3 operations
        np.array([[4, 6, 8, 5],    # Operation 0
                  [7, 3, 5, 9],    # Operation 1
                  [6, 8, 4, 7]]),  # Operation 2
        # Job 4: 2 operations
        np.array([[5, 7, 6, 4],    # Operation 0
                  [8, 5, 9, 6]])   # Operation 1
    ]
    
    # Build precedence constraints
    predecessors_map = {}
    successors_map = {}
    
    # Initialize all operations
    all_operations = []
    for job_id in range(num_jobs):
        for op_id in range(num_operations_per_job[job_id]):
            all_operations.append(Operation(job_id, op_id))
    
    # Within-job precedence (sequential operations in each job)
    for job_id in range(num_jobs):
        for op_id in range(num_operations_per_job[job_id]):
            op = Operation(job_id, op_id)
            
            # Predecessors: previous operation in same job
            predecessors = set()
            if op_id > 0:
                predecessors.add(Operation(job_id, op_id - 1))
            predecessors_map[op] = predecessors
            
            # Successors: next operation in same job
            successors = set()
            if op_id < num_operations_per_job[job_id] - 1:
                successors.add(Operation(job_id, op_id + 1))
            successors_map[op] = successors
    
    # Add some cross-job precedence constraints
    # Job 1's first operation depends on Job 0's second operation
    predecessors_map[Operation(1, 0)].add(Operation(0, 1))
    successors_map[Operation(0, 1)].add(Operation(1, 0))
    
    # Job 3's first operation depends on Job 2's first operation  
    predecessors_map[Operation(3, 0)].add(Operation(2, 0))
    successors_map[Operation(2, 0)].add(Operation(3, 0))
    
    problem = ProblemInstance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )
    
    print(f"Created problem: {problem.num_jobs} jobs, {problem.num_machines} machines, "
          f"{problem.total_operations} operations")
    
    return problem

def test_traditional_algorithm(problem):
    """Test traditional IAOA+GNS algorithm."""
    print("\\n" + "="*50)
    print("Testing IAOA+GNS Algorithm")
    print("="*50)
    
    start_time = time.time()
    
    # Run with small parameters for demo
    algorithm = IAOAGNSAlgorithm(pop_size=20, max_iterations=30)
    solution = algorithm.solve(problem)
    
    solve_time = time.time() - start_time
    
    print(f"IAOA+GNS Results:")
    print(f"  Makespan: {solution.makespan:.2f}")
    print(f"  Solve time: {solve_time:.2f} seconds")
    
    return solution.makespan, solve_time

def test_rl_setup():
    """Test RL setup and device detection."""
    print("\\n" + "="*50)
    print("Testing RL Setup")
    print("="*50)
    
    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected device: {device}")
    
    if device == "cuda":
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Running on CPU (CUDA not available)")
    
    try:
        from src.rl.environments.pofjsp_env import POFJSPEnv
        from src.rl.models.ppo_agent import PPOAgent
        print("✓ RL components import successfully")
        
        return device
    except Exception as e:
        print(f"✗ RL import error: {e}")
        return None

def main():
    """Main demonstration."""
    print("="*60)
    print("POFJSP: RL vs Traditional Algorithms Demo")
    print("="*60)
    
    # Create demo problem
    problem = create_demo_problem()
    
    # Test traditional algorithm
    traditional_makespan, traditional_time = test_traditional_algorithm(problem)
    
    # Test RL setup
    rl_device = test_rl_setup()
    
    print("\\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Problem size: {problem.num_jobs}J x {problem.num_machines}M")
    print(f"Total operations: {problem.total_operations}")
    print()
    print("Traditional Algorithm (IAOA+GNS):")
    print(f"  Makespan: {traditional_makespan:.2f}")
    print(f"  Time: {traditional_time:.2f}s")
    print()
    if rl_device:
        print("RL Setup: ✓ Ready")
        print(f"  Device: {rl_device}")
        print("  To train RL: Use the generated flexible dataset")
        print("  Command: python generate_flexible_dataset.py --custom-config")
    else:
        print("RL Setup: ✗ Issues found")
    
    print("\\nDemo completed successfully!")
    print("\\nNext Steps:")
    print("1. Generate larger datasets: python generate_flexible_dataset.py --custom-config")
    print("2. Compare algorithms: python run_iaoa_gns.py --sample")
    print("3. For RL training: Fix environment issues and train on single instances")

if __name__ == "__main__":
    main()