#!/usr/bin/env python3
"""
Algorithm Parameter Study for POFJSP

This script demonstrates how to use Hydra with the main.py interface
to perform systematic parameter studies for the IAOA+GNS algorithm.

The script tests different combinations of:
- Population size
- Number of iterations
- Crossover probability
- Mutation probability

Results are collected and analyzed to identify optimal parameter settings.
"""

import subprocess
import os
import sys
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import time

def run_evaluation(params, dataset_name="dev_experiment"):
    """
    Run an evaluation with specific algorithm parameters.
    
    Args:
        params: Dictionary of parameter name-value pairs
        dataset_name: The name of the dataset to use
    
    Returns:
        Output of the command
    """
    # Construct parameter string
    param_str = " ".join([f"algorithm.{k}={v}" for k, v in params.items()])
    
    # Unique result file for this parameter set
    timestamp = int(time.time())
    result_file = f"results_{timestamp}.csv"
    
    # Build full command
    cmd = f"python main.py mode=evaluate dataset.name={dataset_name} {param_str}"
    
    print(f"Running: {cmd}")
    
    # Execute command
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        print("Error:")
        print(result.stderr)
        return None
    
    return result.stdout


def conduct_parameter_study():
    """Perform a systematic parameter study."""
    print("Starting IAOA+GNS Algorithm Parameter Study")
    print("="*50)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Parameter combinations to test
    param_space = {
        'pop_size': [20, 50, 100],
        'max_iterations': [50, 100, 200],
        'crossover_prob': [0.7, 0.8, 0.9],
        'mutation_prob': [0.1, 0.2, 0.3]
    }
    
    # Generate more focused parameter combinations instead of full factorial
    param_combinations = [
        # Base configuration
        {'pop_size': 50, 'max_iterations': 100, 'crossover_prob': 0.8, 'mutation_prob': 0.2},
        
        # Population size variations
        {'pop_size': 20, 'max_iterations': 100, 'crossover_prob': 0.8, 'mutation_prob': 0.2},
        {'pop_size': 100, 'max_iterations': 100, 'crossover_prob': 0.8, 'mutation_prob': 0.2},
        
        # Iteration variations
        {'pop_size': 50, 'max_iterations': 50, 'crossover_prob': 0.8, 'mutation_prob': 0.2},
        {'pop_size': 50, 'max_iterations': 200, 'crossover_prob': 0.8, 'mutation_prob': 0.2},
        
        # Crossover/mutation variations
        {'pop_size': 50, 'max_iterations': 100, 'crossover_prob': 0.7, 'mutation_prob': 0.3},
        {'pop_size': 50, 'max_iterations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
    ]
    
    # First, make sure we have a development dataset
    print("Ensuring development dataset exists...")
    subprocess.run(
        "python main.py mode=generate dataset.name=dev_experiment dataset.instances_per_config=3",
        shell=True,
        check=True
    )
    
    # Run evaluations for each parameter combination
    results = []
    for idx, params in enumerate(param_combinations):
        print(f"\nTesting parameter set {idx+1}/{len(param_combinations)}:")
        print(f"Parameters: {params}")
        
        # Run evaluation
        output = run_evaluation(params)
        
        if output:
            # Load results
            try:
                df = pd.read_csv("results_dev_experiment.csv")
                avg_makespan = df['makespan'].mean()
                avg_time = df['execution_time'].mean()
                
                result = {
                    **params,  # Include all parameters
                    'avg_makespan': avg_makespan,
                    'avg_time': avg_time
                }
                results.append(result)
                
                print(f"Average makespan: {avg_makespan:.2f}")
                print(f"Average time: {avg_time:.2f}s")
            except Exception as e:
                print(f"Error processing results: {e}")
    
    # Analyze results
    if results:
        results_df = pd.DataFrame(results)
        print("\nParameter Study Results:")
        print("="*50)
        print(results_df)
        
        # Save results
        results_df.to_csv("parameter_study_results.csv", index=False)
        print("\nResults saved to parameter_study_results.csv")
        
        # Find best parameters
        best_idx = results_df['avg_makespan'].idxmin()
        best_params = results_df.iloc[best_idx]
        
        print("\nBest Parameters:")
        print(f"Population Size: {best_params['pop_size']}")
        print(f"Max Iterations: {best_params['max_iterations']}")
        print(f"Crossover Probability: {best_params['crossover_prob']}")
        print(f"Mutation Probability: {best_params['mutation_prob']}")
        print(f"Average Makespan: {best_params['avg_makespan']:.2f}")
        print(f"Average Runtime: {best_params['avg_time']:.2f}s")
    else:
        print("No valid results obtained from parameter study.")


if __name__ == "__main__":
    conduct_parameter_study() 