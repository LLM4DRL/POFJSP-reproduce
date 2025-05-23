"""
Visualization runner for POFJSP.

This module provides functions to generate visualizations from datasets and results,
integrated with the Hydra configuration system.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import logging

from data_loader import POFJSPDataLoader
from algorithms import iaoa_gns_algorithm, ProblemInstance, Solution

from .gantt import visualize_solution, visualize_comparative_schedules
from .analysis import (
    visualize_dataset_distribution,
    plot_complexity_characteristics, 
    plot_algorithm_performance,
    plot_convergence_history,
    plot_reproduction_comparison
)


def visualize_dataset(cfg, data_dir: Path):
    """
    Generate dataset visualization based on configuration.
    
    Args:
        cfg: Hydra configuration object
        data_dir: Path to the data directory
    """
    # Load dataset metadata
    dataset_name = cfg.dataset.name
    
    # Initialize loader
    try:
        loader = POFJSPDataLoader(str(data_dir))
        metadata_df = loader.load_metadata(dataset_name)
    except Exception as e:
        print(f"Error loading dataset metadata: {e}")
        return
    
    # Configure output directory
    output_dir = Path(cfg.visualization.save_path) / "dataset" / dataset_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    visualize_dataset_distribution(
        metadata_df=metadata_df,
        output_dir=str(output_dir),
        file_formats=cfg.visualization.save_formats
    )
    
    plot_complexity_characteristics(
        metadata_df=metadata_df,
        output_dir=str(output_dir),
        file_formats=cfg.visualization.save_formats
    )
    
    print(f"Dataset visualizations saved to: {output_dir}")


def visualize_results(cfg, results_df: pd.DataFrame):
    """
    Generate result visualizations based on configuration.
    
    Args:
        cfg: Hydra configuration object
        results_df: DataFrame with algorithm results
    """
    # Configure output directory
    dataset_name = cfg.dataset.name
    output_dir = Path(cfg.visualization.save_path) / "results" / dataset_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    plot_algorithm_performance(
        results_df=results_df,
        output_dir=str(output_dir),
        file_formats=cfg.visualization.save_formats
    )
    
    # For reproduction mode, add comparison visualization
    if cfg.mode == "reproduce" and 'reported_makespan' in results_df.columns:
        plot_reproduction_comparison(
            results_df=results_df,
            output_dir=str(output_dir),
            file_formats=cfg.visualization.save_formats
        )
    
    print(f"Result visualizations saved to: {output_dir}")


def visualize_schedules(cfg, results_list: List[Dict], data_dir: Path):
    """
    Generate Gantt charts for schedules.
    
    Args:
        cfg: Hydra configuration object
        results_list: List of result dictionaries with instance_id and makespan
        data_dir: Path to the data directory
    """
    # Sample instances to visualize (limit to a reasonable number)
    max_to_visualize = min(cfg.visualization.max_gantt_charts, len(results_list))
    sample_results = results_list[:max_to_visualize]
    
    # Output directory for Gantt charts
    output_dir = Path(cfg.visualization.save_path) / "gantt" / cfg.dataset.name
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader
    loader = POFJSPDataLoader(str(data_dir))
    
    # Load problems dataframe
    try:
        problems_df = loader.load_problems_dataframe(cfg.dataset.name)
    except Exception as e:
        print(f"Error loading problems dataframe: {e}")
        return
    
    # Generate Gantt charts for each sample
    for result in sample_results:
        instance_id = result['instance_id']
        try:
            # Get problem instance
            problem_instance = loader.load_instance_from_dataframe(problems_df, instance_id)
            
            # Get solution (need to rerun algorithm)
            solution = iaoa_gns_algorithm(
                problem_instance=problem_instance,
                pop_size=cfg.algorithm.pop_size,
                max_iterations=cfg.algorithm.max_iterations,
                crossover_prob=cfg.algorithm.crossover_prob,
                mutation_prob=cfg.algorithm.mutation_prob
            )
            
            # Visualize solution
            visualize_solution(
                solution=solution,
                problem_instance=problem_instance,
                instance_id=instance_id,
                output_dir=str(output_dir),
                file_formats=cfg.visualization.save_formats
            )
            
        except Exception as e:
            print(f"Error generating Gantt chart for {instance_id}: {e}")
    
    print(f"Schedule visualizations saved to: {output_dir}")


def visualize_comparative_methods(cfg, data_dir: Path):
    """
    Generate comparative visualizations between different algorithm variants.
    
    Args:
        cfg: Hydra configuration object
        data_dir: Path to the data directory
    """
    # Check if comparative methods are defined
    if not hasattr(cfg, 'comparative_methods') or not cfg.comparative_methods:
        print("No comparative methods defined in configuration.")
        return
    
    # Output directory for comparison charts
    output_dir = Path(cfg.visualization.save_path) / "comparison" / cfg.dataset.name
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader
    loader = POFJSPDataLoader(str(data_dir))
    
    # Load a sample of instances to compare
    try:
        instances = loader.load_instances_by_criteria(
            dataset_name=cfg.dataset.name,
            size_config=cfg.dataset.size_filter,
            precedence_pattern=cfg.dataset.pattern_filter,
            max_instances=min(5, cfg.dataset.max_instances)  # Limit to 5 for comparison
        )
    except Exception as e:
        print(f"Error loading instances for comparison: {e}")
        return
    
    # For each instance, compare different methods
    for instance_id, problem_instance in instances:
        solutions = {}
        
        # Run each comparative method
        for method in cfg.comparative_methods:
            try:
                # Configure algorithm parameters based on method
                pop_size = method.pop_size
                max_iterations = method.max_iterations
                crossover_prob = method.crossover_prob
                mutation_prob = method.mutation_prob
                
                # Run algorithm
                solution = iaoa_gns_algorithm(
                    problem_instance=problem_instance,
                    pop_size=pop_size,
                    max_iterations=max_iterations,
                    crossover_prob=crossover_prob,
                    mutation_prob=mutation_prob
                )
                
                # Store solution with method name
                solutions[method.name] = solution
                
            except Exception as e:
                print(f"Error running {method.name} on {instance_id}: {e}")
        
        # Generate comparative visualization if we have multiple solutions
        if len(solutions) > 1:
            visualize_comparative_schedules(
                solutions=solutions,
                problem_instance=problem_instance,
                instance_id=instance_id,
                output_dir=str(output_dir),
                file_formats=cfg.visualization.save_formats
            )
    
    print(f"Comparative visualizations saved to: {output_dir}")


def run_visualization(cfg, results=None):
    """
    Run visualizations based on configuration.
    
    Args:
        cfg: Hydra configuration object
        results: Optional results DataFrame or list (if not provided, will load from file)
    """
    # Set up paths
    orig_dir = cfg.original_cwd if hasattr(cfg, 'original_cwd') else os.getcwd()
    data_dir = Path(orig_dir) / cfg.dataset.path
    
    # Configure matplotlib based on config
    plt.rcParams["figure.figsize"] = cfg.visualization.figsize
    
    # Visualize dataset statistics if requested
    if cfg.visualization.dataset_stats:
        visualize_dataset(cfg, data_dir)
    
    # Visualize algorithm results
    if results is not None:
        # Results provided directly
        if isinstance(results, pd.DataFrame):
            visualize_results(cfg, results)
        elif isinstance(results, list):
            results_df = pd.DataFrame(results)
            visualize_results(cfg, results_df)
            
            if cfg.visualization.gantt_charts:
                visualize_schedules(cfg, results, data_dir)
        else:
            print(f"Unsupported results format: {type(results)}")
    else:
        # Try to load results from file
        if cfg.mode == "evaluate":
            results_file = f"results_{cfg.dataset.name}.csv"
        elif cfg.mode == "reproduce":
            results_file = "reproduction_results.csv"
        else:
            results_file = None
        
        if results_file and os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            visualize_results(cfg, results_df)
            
            if cfg.visualization.gantt_charts:
                visualize_schedules(cfg, results_df.to_dict('records'), data_dir)
    
    # Generate comparative visualizations if requested
    if cfg.visualization.comparative_methods:
        visualize_comparative_methods(cfg, data_dir)
    
    print(f"All visualizations completed!")


if __name__ == "__main__":
    # Example usage
    print("This module should be imported and used with Hydra configuration.") 