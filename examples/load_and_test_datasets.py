#!/usr/bin/env python3
"""
Sample script demonstrating POFJSP data loading and algorithm evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import POFJSPDataLoader
from algorithms import iaoa_gns_algorithm

def main():
    # Initialize data loader
    loader = POFJSPDataLoader('./data')
    
    # List available datasets
    datasets = loader.list_available_datasets()
    print(f"Available datasets: {datasets}")
    
    if not datasets:
        print("No datasets found. Run dataset generation first.")
        return
    
    # Print available datasets with indices
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets):
        print(f"{i}: {dataset}")
    
    # Get user selection
    while True:
        try:
            selection = int(input("\nSelect dataset number: "))
            if 0 <= selection < len(datasets):
                dataset_name = datasets[selection]
                print(f"\nUsing dataset: {dataset_name}")
                break
            else:
                print(f"Please enter a number between 0 and {len(datasets)-1}")
        except ValueError:
            print("Please enter a valid number")
    
    # Get dataset statistics
    stats = loader.get_dataset_statistics(dataset_name)
    print(f"Dataset statistics:")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Complexity range: {stats['complexity_stats']['min']:.1f} - {stats['complexity_stats']['max']:.1f}")
    print(f"  Operations range: {stats['operations_stats']['min']} - {stats['operations_stats']['max']}")
    
    # Load some small instances for testing
    small_instances = loader.load_instances_by_criteria(
        dataset_name=dataset_name,
        size_config='xlarge',
        max_instances=30
    )
    
    print(f"\nLoaded {len(small_instances)} small instances for testing:")
    
    # Test algorithm on loaded instances
    for instance_id, problem_instance in small_instances:
        print(f"\nTesting instance: {instance_id}")
        print(f"  Jobs: {problem_instance.num_jobs}, Machines: {problem_instance.num_machines}")
        print(f"  Total operations: {problem_instance.total_operations}")
        
        # Run algorithm with small parameters for quick test
        try:
            solution = iaoa_gns_algorithm(problem_instance, pop_size=20, max_iterations=10)
            print(f"  Best makespan: {solution.makespan}")
        except Exception as e:
            print(f"  Error running algorithm: {e}")

if __name__ == "__main__":
    main()
