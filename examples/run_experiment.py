#!/usr/bin/env python3
"""
Sample experiment script for POFJSP algorithm

This script demonstrates how to run experiments using the POFJSP reproduction system.
It performs the following steps:
1. Generate a small dataset
2. Evaluate algorithm performance
3. Compare results

This provides a complete workflow example from data generation to results analysis.
"""

import subprocess
import os
import sys
import pandas as pd
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and print output."""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print("Output:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("Error:")
        print(result.stderr)
        sys.exit(1)
    
    return result.stdout


def main():
    """Run a complete experiment workflow."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Generate a small development dataset
    run_command(
        "python main.py mode=generate dataset.name=dev_experiment dataset.instances_per_config=3",
        "Generating Development Dataset"
    )
    
    # Step 2: Run algorithm evaluation with small parameters for quick testing
    run_command(
        "python main.py mode=evaluate dataset.name=dev_experiment algorithm.pop_size=30 algorithm.max_iterations=50",
        "Evaluating Algorithm Performance"
    )
    
    # Step 3: Run a custom experiment
    run_command(
        "python main.py --config-name=custom_experiment dataset.name=dev_experiment algorithm.pop_size=50",
        "Running Custom Experiment"
    )
    
    # Step 4: Load and analyze results
    try:
        results_file = Path("results_dev_experiment.csv")
        if results_file.exists():
            results = pd.read_csv(results_file)
            
            print(f"\n{'='*50}")
            print("Results Analysis")
            print(f"{'='*50}")
            print(f"Total instances evaluated: {len(results)}")
            print(f"Average makespan: {results['makespan'].mean():.2f}")
            print(f"Min makespan: {results['makespan'].min():.2f}")
            print(f"Max makespan: {results['makespan'].max():.2f}")
            print(f"Average execution time: {results['execution_time'].mean():.2f} seconds")
        else:
            print("\nResults file not found. Check if the evaluation step completed successfully.")
    except Exception as e:
        print(f"Error analyzing results: {e}")
    
    print(f"\n{'='*50}")
    print("Experiment Workflow Completed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 