#!/usr/bin/env python3
"""
Standalone Genetic Algorithm Runner for POFJSP

This script provides a command-line interface to run the Genetic Algorithm
for Partially Ordered Flexible Job Shop Scheduling problems.
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
# Go up two directories from scripts/algorithms/ to reach project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.problems.problem_instance import ProblemInstance
from src.algorithms.genetic_algorithm import solve_with_ga, GAParams
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


def solve_single_instance(problem_path: str, output_dir: str, ga_params: GAParams, verbose: bool = False) -> Dict:
    """Solve a single POFJSP instance with GA."""
    
    # Load problem
    problem = load_problem(problem_path)
    problem_name = os.path.splitext(os.path.basename(problem_path))[0]
    
    if verbose:
        print(f"Solving {problem_name} with GA...")
        print(f"  Jobs: {problem.num_jobs}")
        print(f"  Machines: {problem.num_machines}")
        print(f"  Total operations: {problem.total_operations}")
    
    # Solve with GA
    start_time = time.time()
    result = solve_with_ga(problem, ga_params, verbose=verbose)
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
        'ga_parameters': {
            'population_size': ga_params.population_size,
            'max_generations': ga_params.max_generations,
            'crossover_rate': ga_params.crossover_rate,
            'mutation_rate': ga_params.mutation_rate,
            'elitism_rate': ga_params.elitism_rate
        }
    }
    
    output_file = os.path.join(output_dir, f"{problem_name}_ga_solution.json")
    with open(output_file, 'w') as f:
        json.dump(solution_data, f, indent=2)
    
    if verbose:
        print(f"Solution found with makespan: {result['makespan']:.2f}")
        print(f"Solution saved to: {output_file}")
    
    return solution_data


def solve_dataset(dataset_dir: str, output_dir: str, ga_params: GAParams, verbose: bool = False) -> Dict:
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
                ga_params, 
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
    
    summary_file = os.path.join(output_dir, "ga_summary.json")
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
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for POFJSP")
    
    # Get project root directory for default paths
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    default_problem_path = os.path.join(project_root, "data", "flexible", "flex_30x50_000.json")
    
    # Required arguments
    parser.add_argument("--problem", type=str, default=default_problem_path, help="Path to problem file or dataset directory")
    
    # Optional arguments
    default_output_dir = os.path.join(project_root, "outputs", "ga")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Output directory for results")
    parser.add_argument("--population-size", type=int, default=100,
                        help="Population size")
    parser.add_argument("--max-generations", type=int, default=1000,
                        help="Maximum generations")
    parser.add_argument("--crossover-rate", type=float, default=0.8,
                        help="Crossover rate")
    parser.add_argument("--mutation-rate", type=float, default=0.1,
                        help="Mutation rate")
    parser.add_argument("--elitism-rate", type=float, default=0.1,
                        help="Elitism rate")
    parser.add_argument("--tournament-size", type=int, default=3,
                        help="Tournament size for selection")
    parser.add_argument("--no-improvement-limit", type=int, default=50,
                        help="Stop after this many generations without improvement")
    parser.add_argument("--dataset-mode", action="store_true",
                        help="Process all instances in directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create GA parameters
    ga_params = GAParams(
        population_size=args.population_size,
        max_generations=args.max_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism_rate=args.elitism_rate,
        tournament_size=args.tournament_size,
        no_improvement_limit=args.no_improvement_limit
    )
    
    # Run appropriate mode
    if args.dataset_mode:
        solve_dataset(args.problem, args.output_dir, ga_params, args.verbose)
    else:
        solve_single_instance(args.problem, args.output_dir, ga_params, args.verbose)


if __name__ == "__main__":
    main()