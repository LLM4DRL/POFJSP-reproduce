#!/usr/bin/env python3
"""
Main entry point for POFJSP algorithm reproduction using Hydra for configuration management.

Usage:
    python main.py                     # Run with default config
    python main.py dataset=benchmark   # Override dataset config
    python main.py algorithm.pop_size=50 algorithm.max_iterations=100  # Override algorithm parameters
    python main.py --config-name=custom_experiment  # Use a different config file
    python main.py mode=visualize      # Run visualization only
"""

import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import POFJSPDataLoader
from data_generator import POFJSPDataGenerator
from algorithms import iaoa_gns_algorithm, ProblemInstance

# Import visualization module
from src.visualization import visualize


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for running POFJSP experiments.
    
    Args:
        cfg: Hydra configuration object
    """
    print(f"{'='*50}")
    print(f"POFJSP Algorithm Reproduction")
    print(f"{'='*50}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    print(f"{'='*50}")
    
    # Store original working directory for use by other modules
    original_cwd = hydra.utils.get_original_cwd()
    
    # Get mode from config
    mode = cfg.mode.mode if hasattr(cfg, 'mode') and hasattr(cfg.mode, 'mode') else "evaluate"
    
    # Ensure compatibility with original code paths
    data_dir = Path(original_cwd) / cfg.dataset.path
    
    # Determine operation mode
    if mode == "generate":
        instances = _generate_dataset(cfg)
        
        # Visualize the generated dataset if requested
        if cfg.visualization.dataset_stats:
            visualize.visualize_dataset(cfg, data_dir)
            
    elif mode == "evaluate":
        results = _evaluate_algorithm(cfg)
        
        # Visualize results if requested
        if cfg.visualization.enabled:
            visualize.run_visualization(cfg, results)
            
    elif mode == "reproduce":
        results = _reproduce_results(cfg)
        
        # Visualize comparison if requested
        if cfg.visualization.enabled:
            visualize.run_visualization(cfg, results)
            
    elif mode == "visualize":
        # Run only visualization for existing results
        visualize.run_visualization(cfg)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print(f"\n✅ Execution completed successfully!")
    return 0


def _generate_dataset(cfg: DictConfig):
    """Generate dataset based on configuration."""
    print(f"\n📦 Generating dataset: {cfg.dataset.name}")
    
    # Configure generator
    generator = POFJSPDataGenerator(random_seed=cfg.random_seed)
    
    # Prepare configurations
    configurations = []
    for config_item in cfg.dataset.configurations:
        configuration = {
            'name': config_item.name,
            'size_config': config_item.size_config,
            'precedence_pattern': config_item.precedence_pattern,
            'time_distribution': config_item.time_distribution,
            'machine_capability_prob': config_item.machine_capability_prob
        }
        configurations.append(configuration)
    
    # Generate dataset
    orig_dir = hydra.utils.get_original_cwd()
    output_dir = Path(orig_dir) / cfg.dataset.path / cfg.dataset.name
    
    instances = generator.generate_dataset(
        configurations=configurations,
        instances_per_config=cfg.dataset.instances_per_config,
        output_dir=str(output_dir)
    )
    
    # Print summary
    print(f"\n📊 Dataset Generation Summary:")
    print(f"  Total configurations: {len(configurations)}")
    print(f"  Total instances: {len(instances)}")
    print(f"  Output directory: {output_dir}")
    
    return instances


def _evaluate_algorithm(cfg: DictConfig):
    """Evaluate algorithm on specified dataset."""
    print(f"\n🧪 Evaluating algorithm on dataset: {cfg.dataset.name}")
    
    # Initialize data loader
    orig_dir = hydra.utils.get_original_cwd()
    data_dir = Path(orig_dir) / cfg.dataset.path
    loader = POFJSPDataLoader(str(data_dir))
    
    # Load instances
    instances = loader.load_instances_by_criteria(
        dataset_name=cfg.dataset.name,
        size_config=cfg.dataset.size_filter,
        precedence_pattern=cfg.dataset.pattern_filter,
        complexity_range=cfg.dataset.complexity_range if hasattr(cfg.dataset, 'complexity_range') else None,
        max_instances=cfg.dataset.max_instances
    )
    
    print(f"Loaded {len(instances)} instances for evaluation")
    
    # Prepare results storage
    results = []
    
    # Store convergence histories if requested
    convergence_histories = []
    
    # Run algorithm on each instance
    for idx, (instance_id, problem) in enumerate(instances):
        print(f"\n[{idx+1}/{len(instances)}] Evaluating instance: {instance_id}")
        
        # Configure algorithm parameters from config
        pop_size = cfg.algorithm.pop_size
        max_iterations = cfg.algorithm.max_iterations
        crossover_prob = cfg.algorithm.crossover_prob
        mutation_prob = cfg.algorithm.mutation_prob
        
        # Determine if we should track convergence
        track_convergence = cfg.visualization.enabled and cfg.visualization.convergence_plot
        
        # Measure execution time
        start_time = datetime.datetime.now()
        
        # Run algorithm
        solution = iaoa_gns_algorithm(
            problem=problem,
            pop_size=pop_size,
            max_iterations=max_iterations
        )
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Store results
        result = {
            'instance_id': instance_id,
            'makespan': solution.makespan,
            'execution_time': execution_time,
            'pop_size': pop_size,
            'max_iterations': max_iterations,
            'crossover_prob': crossover_prob,
            'mutation_prob': mutation_prob
        }
        
        results.append(result)
        print(f"  Makespan: {solution.makespan:.2f}, Time: {execution_time:.2f}s")
        
        # Store convergence history if available
        if track_convergence and hasattr(solution, 'convergence_history'):
            convergence_histories.append(solution.convergence_history)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = f"results_{cfg.dataset.name}.csv"
    results_df.to_csv(results_file, index=False)
    
    # Print summary
    print(f"\n📊 Evaluation Summary:")
    print(f"  Average makespan: {results_df['makespan'].mean():.2f}")
    print(f"  Average execution time: {results_df['execution_time'].mean():.2f}s")
    print(f"  Results saved to: {results_file}")
    
    # Plot convergence if requested and data available
    if cfg.visualization.enabled and cfg.visualization.convergence_plot and convergence_histories:
        from src.visualization.analysis import plot_convergence_history
        plot_convergence_history(
            convergence_data=convergence_histories,
            title=f"Algorithm Convergence on {cfg.dataset.name}",
            output_dir=f"figures/analysis/convergence/{cfg.dataset.name}",
            file_formats=cfg.visualization.save_formats
        )
    
    return results


def _reproduce_results(cfg: DictConfig):
    """Reproduce the original paper results."""
    print(f"\n🔍 Reproducing results from the paper")
    
    # Load specific benchmark instances
    orig_dir = hydra.utils.get_original_cwd()
    data_dir = Path(orig_dir) / cfg.dataset.path
    loader = POFJSPDataLoader(str(data_dir))
    
    # Load specific instances for reproduction
    instances = []
    for instance_spec in cfg.reproduction.instances:
        dataset_name = instance_spec.dataset
        instance_id = instance_spec.id
        
        try:
            problems_df = loader.load_problems_dataframe(dataset_name)
            problem = loader.load_instance_from_dataframe(problems_df, instance_id)
            instances.append((instance_id, problem))
            print(f"Loaded instance {instance_id} from {dataset_name}")
        except Exception as e:
            print(f"Failed to load instance {instance_id}: {e}")
    
    # Run algorithm with original settings
    results = []
    
    for instance_id, problem in instances:
        print(f"\nReproducing results for instance: {instance_id}")
        
        # Use original algorithm settings
        pop_size = cfg.reproduction.algorithm.pop_size
        max_iterations = cfg.reproduction.algorithm.max_iterations
        crossover_prob = cfg.reproduction.algorithm.crossover_prob
        mutation_prob = cfg.reproduction.algorithm.mutation_prob
        
        # Run algorithm
        solution = iaoa_gns_algorithm(
            problem=problem,
            pop_size=pop_size,
            max_iterations=max_iterations
        )
        
        # Record results
        result = {
            'instance_id': instance_id,
            'makespan': solution.makespan
        }
        
        # Compare with reported results if available
        for reported in cfg.reproduction.reported_results:
            if reported.instance_id == instance_id:
                result['reported_makespan'] = reported.makespan
                result['difference'] = (solution.makespan - reported.makespan) / reported.makespan * 100
                break
        
        results.append(result)
        
        if 'reported_makespan' in result:
            print(f"  Our makespan: {solution.makespan:.2f}")
            print(f"  Reported makespan: {result['reported_makespan']:.2f}")
            print(f"  Difference: {result['difference']:.2f}%")
        else:
            print(f"  Makespan: {solution.makespan:.2f}")
    
    # Save reproduction results
    results_df = pd.DataFrame(results)
    results_file = "reproduction_results.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n📊 Reproduction Summary:")
    if 'difference' in results_df.columns:
        print(f"  Average difference: {results_df['difference'].mean():.2f}%")
    print(f"  Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main() 