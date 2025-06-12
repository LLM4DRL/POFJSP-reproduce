#!/usr/bin/env python3
"""
Example script demonstrating POFJSP visualization capabilities.

This script shows how to:
1. Generate dataset visualizations
2. Generate Gantt charts for schedules
3. Create comparative visualizations
4. Analyze algorithm performance

Usage:
    python examples/visualization_example.py
"""

import os
import sys
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import random

# Add the root directory to the path
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.insert(0, str(root_dir))

from src.visualization import gantt, analysis
from src.data_loader import POFJSPDataLoader
from src.algorithms import iaoa_gns_algorithm, Operation, ProblemInstance

# Set up paths
data_dir = root_dir / 'data'
figures_dir = root_dir / 'figures' / 'examples'


def generate_dataset_visualizations():
    """Generate visualizations for an existing dataset."""
    print(f"\n{'='*50}")
    print("Generating Dataset Visualizations")
    print(f"{'='*50}")
    
    # First, check if dataset exists
    loader = POFJSPDataLoader(str(data_dir))
    available_datasets = loader.list_available_datasets()
    
    if not available_datasets:
        print("No datasets available. Generating a development dataset...")
        subprocess.run(
            "python main.py mode=generate dataset.name=development dataset.instances_per_config=3",
            shell=True,
            check=True
        )
        dataset_name = "development"
    else:
        dataset_name = available_datasets[0]
        print(f"Using existing dataset: {dataset_name}")
    
    # Load dataset metadata
    metadata_df = loader.load_metadata(dataset_name)
    
    # Create visualization output directory
    output_dir = figures_dir / "dataset" / dataset_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    analysis.visualize_dataset_distribution(
        metadata_df=metadata_df,
        output_dir=str(output_dir),
        file_formats=["png", "pdf"]
    )
    
    analysis.plot_complexity_characteristics(
        metadata_df=metadata_df,
        output_dir=str(output_dir),
        file_formats=["png", "pdf"]
    )
    
    print(f"Dataset visualizations saved to {output_dir}")
    return dataset_name, loader


def generate_gantt_charts(dataset_name, loader):
    """Generate Gantt charts for a few instances."""
    print(f"\n{'='*50}")
    print("Generating Gantt Charts")
    print(f"{'='*50}")
    
    # Load a few instances
    instances = loader.load_instances_by_criteria(
        dataset_name=dataset_name,
        max_instances=3
    )
    
    if not instances:
        print("No instances available for Gantt charts.")
        return
    
    # Create output directory
    output_dir = figures_dir / "gantt"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Gantt charts for each instance
    for instance_id, problem_instance in instances:
        print(f"Generating Gantt chart for {instance_id}...")
        
        # Run algorithm
        solution = iaoa_gns_algorithm(
            problem_instance=problem_instance,
            pop_size=30,
            max_iterations=50
        )
        
        # Visualize solution
        gantt.visualize_solution(
            solution=solution,
            problem_instance=problem_instance,
            instance_id=instance_id,
            output_dir=str(output_dir),
            file_formats=["png", "pdf"]
        )
    
    print(f"Gantt charts saved to {output_dir}")
    return instances


def generate_comparative_visualization(instances):
    """Generate comparative visualizations between different algorithm configurations."""
    print(f"\n{'='*50}")
    print("Generating Comparative Visualizations")
    print(f"{'='*50}")
    
    if not instances:
        print("No instances available for comparison.")
        return
    
    # Take just the first instance
    instance_id, problem_instance = instances[0]
    
    # Configure different algorithm variants
    variants = {
        "Basic IAO (50 pop)": {
            "pop_size": 30, 
            "max_iterations": 50, 
            "crossover_prob": 0.8, 
            "mutation_prob": 0.2
        },
        "IAO+GNS (100 pop)": {
            "pop_size": 50, 
            "max_iterations": 50, 
            "crossover_prob": 0.85, 
            "mutation_prob": 0.15
        },
        "IAO+GNS (high xover)": {
            "pop_size": 40, 
            "max_iterations": 50, 
            "crossover_prob": 0.95, 
            "mutation_prob": 0.05
        }
    }
    
    # Create output directory
    output_dir = figures_dir / "comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run each variant and collect solutions
    solutions = {}
    for name, params in variants.items():
        print(f"Running {name}...")
        solution = iaoa_gns_algorithm(
            problem_instance=problem_instance,
            **params
        )
        solutions[name] = solution
    
    # Generate comparative visualization
    gantt.visualize_comparative_schedules(
        solutions=solutions,
        problem_instance=problem_instance,
        instance_id=instance_id,
        output_dir=str(output_dir),
        file_formats=["png", "pdf"]
    )
    
    print(f"Comparative visualizations saved to {output_dir}")


def analyze_algorithm_performance():
    """Visualize algorithm performance metrics."""
    print(f"\n{'='*50}")
    print("Analyzing Algorithm Performance")
    print(f"{'='*50}")
    
    # Create simulated results data
    results = []
    for i in range(30):
        # Simulate results with different parameters
        pop_size = random.choice([20, 50, 100])
        max_iterations = random.choice([50, 100, 200])
        crossover_prob = random.choice([0.7, 0.8, 0.9])
        
        # Makespan tends to be better with higher population/iterations
        makespan_base = 100 - pop_size/5 - max_iterations/10
        makespan = max(20, random.normalvariate(makespan_base, 5))
        
        # Execution time correlates with population size and iterations
        execution_time = (pop_size * max_iterations) / 1000 + random.uniform(0.5, 2.0)
        
        results.append({
            'instance_id': f"instance_{i:02d}",
            'makespan': makespan,
            'execution_time': execution_time,
            'pop_size': pop_size,
            'max_iterations': max_iterations,
            'crossover_prob': crossover_prob
        })
    
    results_df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = figures_dir / "analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate performance visualizations
    analysis.plot_algorithm_performance(
        results_df=results_df,
        output_dir=str(output_dir),
        file_formats=["png", "pdf"]
    )
    
    # Create simulated convergence data
    convergence_data = []
    for _ in range(5):
        # Start with a value around 100 and decrease
        start_val = random.uniform(90, 110)
        history = [start_val]
        
        for i in range(49):  # 50 iterations total
            # Decay rate gets smaller as iterations increase
            decay = random.uniform(0.95, 0.99) 
            history.append(history[-1] * decay)
        
        convergence_data.append(history)
    
    # Generate convergence plot
    analysis.plot_convergence_history(
        convergence_data=convergence_data,
        title="Example Algorithm Convergence",
        output_dir=str(output_dir),
        file_formats=["png", "pdf"]
    )
    
    print(f"Performance analysis visualizations saved to {output_dir}")


def main():
    """Run all visualization examples."""
    print("POFJSP Visualization Examples")
    print("="*50)
    
    # Create figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    
    # Run examples
    dataset_name, loader = generate_dataset_visualizations()
    instances = generate_gantt_charts(dataset_name, loader)
    generate_comparative_visualization(instances)
    analyze_algorithm_performance()
    
    print(f"\n{'='*50}")
    print("Visualization Examples Completed")
    print(f"Results saved to {figures_dir}")


if __name__ == "__main__":
    main() 