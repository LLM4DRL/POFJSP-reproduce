#!/usr/bin/env python3
"""
IAOA+GNS Runner for POFJSP

This script provides a simple interface to run the IAOA+GNS algorithm
on POFJSP problem instances.
"""

import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.algorithms.iaoa_gns import IAOAGNSAlgorithm
from src.problems.problem_instance import ProblemInstance, Operation


def create_sample_problem():
    """Create a sample POFJSP problem for testing."""
    import numpy as np
    
    # Sample problem: 2 jobs, 2 machines
    num_operations_per_job = [2, 2]  # J0 has 2 ops, J1 has 2 ops
    processing_times = [
        # Job 0: [[Op0_0_M0, Op0_0_M1], [Op0_1_M0, Op0_1_M1]]
        np.array([[3, 5], [6, np.inf]]),  # J0,O0: M0=3, M1=5; J0,O1: M0=6, M1=cannot
        # Job 1: [[Op1_0_M0, Op1_0_M1], [Op1_1_M0, Op1_1_M1]]
        np.array([[4, 2], [np.inf, 7]])   # J1,O0: M0=4, M1=2; J1,O1: M0=cannot, M1=7
    ]

    from src.problems.problem_instance import Operation
    predecessors_map = {
        Operation(0, 1): {Operation(0, 0)},  # J0,O1 needs J0,O0
        Operation(1, 1): {Operation(1, 0)},  # J1,O1 needs J1,O0
        Operation(0, 0): set(),
        Operation(1, 0): set()
    }
    successors_map = {
        Operation(0, 0): {Operation(0, 1)},
        Operation(1, 0): {Operation(1, 1)},
        Operation(0, 1): set(),
        Operation(1, 1): set()
    }
    
    return ProblemInstance(
        num_jobs=2,
        num_machines=2,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )


def main():
    parser = argparse.ArgumentParser(description='Run IAOA+GNS algorithm on POFJSP instances')
    parser.add_argument('--dataset', type=str, help='Dataset directory path')
    parser.add_argument('--instance', type=str, help='Specific instance ID to run')
    parser.add_argument('--pop-size', type=int, default=80, help='Population size (default: 80)')
    parser.add_argument('--max-iter', type=int, default=60, help='Maximum iterations (default: 60)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--sample', action='store_true', help='Run on sample problem')
    
    args = parser.parse_args()
    
    # Initialize algorithm
    algorithm = IAOAGNSAlgorithm(pop_size=args.pop_size, max_iterations=args.max_iter)
    
    if args.sample:
        # Run on sample problem
        print("Running IAOA+GNS on sample problem...")
        problem = create_sample_problem()
        
        solution = algorithm.solve(problem, verbose=args.verbose)
        
        print(f"\nBest solution found:")
        print(f"Makespan: {solution.makespan}")
        
        if args.verbose:
            print("\nOperation Sequence (Job, Op) -> Machine:")
            for op in solution.operation_sequence:
                idx = solution.operation_sequence.index(op)
                machine = solution.machine_assignment[idx]
                print(f"  ({op.job_idx}, {op.op_idx_in_job}) -> M{machine}")
    
    elif args.dataset and args.instance:
        # Load specific instance from dataset
        print(f"Loading instance {args.instance} from dataset {args.dataset}...")
        
        try:
            from pathlib import Path
            dataset_path = Path(args.dataset)
            instance_file = dataset_path / f"{args.instance}.json"
            
            if not instance_file.exists():
                # Try looking in the dataset directory
                instance_files = list(dataset_path.glob("*.json"))
                if not instance_files:
                    print("No JSON files found in dataset directory")
                    return 1
                
                print(f"Available instances:")
                for f in instance_files[:10]:  # Show first 10
                    print(f"  {f.stem}")
                return 1
            
            problem = ProblemInstance.from_json(str(instance_file))
            
            print(f"Problem: {problem.num_jobs} jobs, {problem.num_machines} machines, {problem.total_operations} operations")
            
            solution = algorithm.solve(problem, verbose=args.verbose)
            
            print(f"\nBest solution found:")
            print(f"Makespan: {solution.makespan}")
            
        except Exception as e:
            print(f"Error loading instance: {e}")
            print("Make sure the dataset exists and contains the specified instance.")
            return 1
    
    elif args.dataset:
        # List available instances in dataset
        try:
            from pathlib import Path
            dataset_path = Path(args.dataset)
            
            if not dataset_path.exists():
                print(f"Dataset directory not found: {args.dataset}")
                return 1
            
            instance_files = list(dataset_path.glob("*.json"))
            if not instance_files:
                print("No JSON files found in dataset directory")
                return 1
            
            print(f"Found {len(instance_files)} instances:")
            for f in instance_files[:10]:  # Show first 10
                print(f"  {f.stem}")
                
        except Exception as e:
            print(f"Error accessing dataset: {e}")
            return 1
    
    else:
        # Interactive mode
        print("IAOA+GNS Runner for POFJSP")
        print("Usage examples:")
        print("  python run_iaoa_gns.py --sample                    # Run on sample problem")
        print("  python run_iaoa_gns.py --dataset ./data/dev          # List available instances")
        print("  python run_iaoa_gns.py --dataset ./data/dev --instance config_000_small_mixed_uniform_80_000")
        print("  python run_iaoa_gns.py --dataset ./data/dev --instance config_000_small_mixed_uniform_80_000 --verbose")
        print("  python run_iaoa_gns.py --dataset ./data/dev --instance config_000_small_mixed_uniform_80_000 --pop-size 50 --max-iter 30")


if __name__ == "__main__":
    main()