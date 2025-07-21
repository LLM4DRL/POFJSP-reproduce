#!/usr/bin/env python3
"""
Test traditional algorithms on flexible dataset.
"""

import sys
import os
import time
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.problems.problem_instance import ProblemInstance
from src.algorithms.iaoa_gns import IAOAGNSAlgorithm
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.simulated_annealing import SimulatedAnnealing

def load_instances(dataset_dir):
    """Load problem instances from dataset."""
    instances = []
    dataset_path = Path(dataset_dir)
    
    for json_file in sorted(dataset_path.glob("*.json")):
        if json_file.name.startswith("dataset_summary"):
            continue
        try:
            instance = ProblemInstance.from_json(str(json_file))
            instances.append((json_file.name, instance))
            print(f"Loaded {json_file.name}: {instance.num_jobs}J x {instance.num_machines}M")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return instances

def test_algorithm(algorithm_name, algorithm_class, instance_name, instance, **params):
    """Test a single algorithm on an instance."""
    print(f"\\nTesting {algorithm_name} on {instance_name}")
    
    start_time = time.time()
    algorithm = algorithm_class(**params)
    solution = algorithm.solve(instance)
    solve_time = time.time() - start_time
    
    makespan = solution.makespan if hasattr(solution, 'makespan') else float('inf')
    
    print(f"  Makespan: {makespan:.1f}, Time: {solve_time:.2f}s")
    
    return {
        'algorithm': algorithm_name,
        'instance': instance_name,
        'num_jobs': instance.num_jobs,
        'num_machines': instance.num_machines,
        'total_operations': instance.total_operations,
        'makespan': makespan,
        'solve_time': solve_time
    }

def main():
    print("=== Testing Traditional Algorithms ===")
    
    # Load instances
    instances = load_instances("./data/flexible")
    if not instances:
        print("No instances found!")
        return
    
    # Test small subset for verification
    test_instances = instances[:3]  # Take first 3 instances
    
    # Algorithm configurations
    algorithms = [
        ('IAOA_GNS', IAOAGNSAlgorithm, {'pop_size': 20, 'max_iterations': 50}),
        ('GA', GeneticAlgorithm, {'population_size': 50, 'max_generations': 100}),
        ('SA', SimulatedAnnealing, {'initial_temp': 100.0, 'max_iterations': 500}),
    ]
    
    results = []
    
    for instance_name, instance in test_instances:
        print(f"\\n{'='*50}")
        print(f"Processing {instance_name}: {instance.num_jobs}J x {instance.num_machines}M")
        print(f"{'='*50}")
        
        for algo_name, algo_class, params in algorithms:
            try:
                result = test_algorithm(algo_name, algo_class, instance_name, instance, **params)
                results.append(result)
            except Exception as e:
                print(f"Error with {algo_name}: {e}")
                results.append({
                    'algorithm': algo_name,
                    'instance': instance_name,
                    'num_jobs': instance.num_jobs,
                    'num_machines': instance.num_machines,
                    'total_operations': instance.total_operations,
                    'makespan': float('inf'),
                    'solve_time': 0,
                    'error': str(e)
                })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("traditional_algorithms_test_results.csv", index=False)
    print(f"\\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(df.groupby('algorithm')[['makespan', 'solve_time']].mean().round(2))
    print(f"\\nResults saved to traditional_algorithms_test_results.csv")

if __name__ == "__main__":
    main()