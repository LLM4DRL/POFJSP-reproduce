#!/usr/bin/env python3
"""
Standalone Simulated Annealing Runner for POFJSP

This script provides a command-line interface to run the Simulated Annealing
algorithm for Partially Ordered Flexible Job Shop Scheduling problems.
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict

# Add src to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.problems.problem_instance import ProblemInstance
from src.algorithms.simulated_annealing import solve_with_sa, SAParams
from src.algorithms.decoder import decode_solution


def load_problem(problem_path: str) -> ProblemInstance:
    """Load problem instance from file."""
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file not found: {problem_path}")
    
    # Load based on file extension
    if problem_path.endswith('.json'):
        return ProblemInstance.from_json(problem_path)
    elif problem_path.endswith('.txt'):
        return ProblemInstance.from_fjsp_file(problem_path)
    else:
        raise ValueError(f"Unsupported file format: {problem_path}")


def solve_single_instance(problem_path: str, output_dir: str, sa_params: SAParams, verbose: bool = False) -> Dict:
    """Solve a single POFJSP instance with SA."""
    
    # Load problem
    problem = load_problem(problem_path)
    problem_name = os.path.splitext(os.path.basename(problem_path))[0]
    
    if verbose:
        print(f"Solving {problem_name} with SA...")
        print(f"  Jobs: {problem.num_jobs}")
        print(f"  Machines: {problem.num_machines}")
        print(f"  Total operations: {problem.total_operations}")
    
    # Solve with SA
    start_time = time.time()
    result = solve_with_sa(problem, sa_params, verbose=verbose)
    solve_time = time.time() - start_time
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save solution
    solution_data = {
        'problem_name': problem_name,
        'makespan': result['makespan'],
        'solve_time': solve_time,
        'operation_sequence': result['solution'].operation_sequence,
        'machine_assignment': result['solution'].machine_assignment,
        'statistics': result['statistics'],
        'sa_parameters': {
            'initial_temp': sa_params.initial_temp,
            'final_temp': sa_params.final_temp,
            'cooling_rate': sa_params.cooling_rate,
            'max_iterations': sa_params.max_iterations,
            'max_stagnation': sa_params.max_stagnation,
            'acceptance_criterion': sa_params.acceptance_criterion
        }
    }
    
    output_file = os.path.join(output_dir, f"{problem_name}_sa_solution.json")
    with open(output_file, 'w') as f:
        json.dump(solution_data, f, indent=2)
    
    if verbose:
        print(f"Solution found with makespan: {result['makespan']:.2f}")
        print(f"Solution saved to: {output_file}")
    
    return solution_data


def solve_dataset(dataset_dir: str, output_dir: str, sa_params: SAParams, verbose: bool = False) -> Dict:
    """Solve all instances in a dataset directory."""
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Find all JSON files
    instance_files = list(dataset_path.glob("*.json"))
    if not instance_files:
        # Check instances subdirectory
        instances_dir = dataset_path / "instances"
        if instances_dir.exists():
            instance_files = list(instances_dir.glob("*.json"))
    
    if not instance_files:
        raise ValueError(f"No JSON instance files found in {dataset_dir}")
    
    instance_files.sort()
    
    if verbose:
        print(f"Found {len(instance_files)} instances to solve")
    
    results = []
    total_time = 0
    
    for instance_file in instance_files:
        try:
            result = solve_single_instance(
                str(instance_file), 
                output_dir, 
                sa_params, 
                verbose=False
            )
            results.append(result)
            total_time += result['solve_time']
            
            if verbose:
                print(f"✓ {instance_file.name}: {result['makespan']:.2f} ({result['solve_time']:.2f}s)")
        
        except Exception as e:
            print(f"✗ {instance_file.name}: Error - {str(e)}")
            results.append({
                'problem_name': instance_file.stem,
                'makespan': None,
                'solve_time': None,
                'error': str(e)
            })
    
    # Save summary
    valid_results = [r for r in results if r['makespan'] is not None]
    if valid_results:
        summary = {
            'total_instances': len(results),
            'successful': len(valid_results),
            'failed': len(results) - len(valid_results),
            'avg_makespan': np.mean([r['makespan'] for r in valid_results]),
            'min_makespan': min([r['makespan'] for r in valid_results]),
            'max_makespan': max([r['makespan'] for r in valid_results]),
            'total_time': total_time,
            'avg_time': total_time / len(valid_results),
            'results': results
        }
    else:
        summary = {
            'total_instances': len(results),
            'successful': 0,
            'failed': len(results),
            'error': 'No successful solutions'
        }
    
    summary_file = os.path.join(output_dir, "sa_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\nDataset processing completed:")
        print(f"  Successful: {summary['successful']}/{summary['total_instances']}")
        if valid_results:
            print(f"  Average makespan: {summary['avg_makespan']:.2f}")
            print(f"  Total time: {summary['total_time']:.2f}s")
        print(f"  Summary saved to: {summary_file}")
    
    return summary


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Run Simulated Annealing for POFJSP")
    
    # Required arguments
    parser.add_argument("problem", type=str, help="Path to problem file or dataset directory")
    
    # Optional arguments
    parser.add_argument("--output-dir", type=str, default="./outputs/sa",
                        help="Output directory for results")
    parser.add_argument("--initial-temp", type=float, default=100.0,
                        help="Initial temperature")
    parser.add_argument("--final-temp", type=float, default=0.01,
                        help="Final temperature")
    parser.add_argument("--cooling-rate", type=float, default=0.95,
                        help="Cooling rate")
    parser.add_argument("--max-iterations", type=int, default=10000,
                        help="Maximum iterations")
    parser.add_argument("--max-stagnation", type=int, default=1000,
                        help="Maximum iterations without improvement")
    parser.add_argument("--reheat-factor", type=float, default=1.5,
                        help="Reheat factor when stagnated")
    parser.add_argument("--acceptance-criterion", type=str, choices=["metropolis", "threshold"],
                        default="metropolis", help="Acceptance criterion")
    parser.add_argument("--dataset-mode", action="store_true",
                        help="Process all instances in directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create SA parameters
    sa_params = SAParams(
        initial_temp=args.initial_temp,
        final_temp=args.final_temp,
        cooling_rate=args.cooling_rate,
        max_iterations=args.max_iterations,
        max_stagnation=args.max_stagnation,
        reheat_factor=args.reheat_factor,
        acceptance_criterion=args.acceptance_criterion
    )
    
    # Run appropriate mode
    if args.dataset_mode:
        solve_dataset(args.problem, args.output_dir, sa_params, args.verbose)
    else:
        solve_single_instance(args.problem, args.output_dir, sa_params, args.verbose)


if __name__ == "__main__":
    main()