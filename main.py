#!/usr/bin/env python3
"""
Main entry point for POFJSP algorithm reproduction using Hydra for configuration management.

Usage:
    python main.py                     # Run with default config
    python main.py dataset=benchmark   # Override dataset config
    python main.py algorithm.pop_size=50 algorithm.max_iterations=100  # Override algorithm parameters
    python main.py --config-name=custom_experiment  # Use a different config file
    python main.py mode=visualize      # Run visualization only
    python main.py mode=rl             # Train and evaluate a PPO agent
    python main.py multiprocessing.enabled=true  # Enable parallel processing
"""

import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import pandas as pd
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import POFJSPDataLoader
from data_generator import POFJSPDataGenerator
from algorithms import iaoa_gns_algorithm, ProblemInstance

# Import visualization module
from src.visualization import visualize
from src.visualization.visualize import visualize_solution

# Import RL modules when needed
from src.rl_agent import train_ppo_agent, POFJSPAgent
from src.rl_env import POFJSPEnv


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
        
    elif mode == "rl":
        # Train and evaluate a PPO agent
        results = _run_rl_mode(cfg)
        
        # Visualize results if requested
        if cfg.rl.visualization.enabled:
            visualize.run_visualization(cfg, results)
            
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


def _process_single_instance(instance_id, problem, cfg):
    """
    Process a single problem instance.
    
    Args:
        instance_id: Identifier for the instance
        problem: The ProblemInstance object
        cfg: Configuration object
        
    Returns:
        Dictionary with results for this instance
    """
    # Configure algorithm parameters
    pop_size = cfg.algorithm.pop_size
    max_iterations = cfg.algorithm.max_iterations
    crossover_prob = cfg.algorithm.crossover_prob
    mutation_prob = cfg.algorithm.mutation_prob
    
    # Determine if we should track convergence
    track_convergence = cfg.visualization.enabled and cfg.visualization.convergence_plot
    
    # Measure execution time
    start_time = time.time()
    
    # Run algorithm
    solution = iaoa_gns_algorithm(
        problem=problem,
        pop_size=pop_size,
        max_iterations=max_iterations
    )
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Analyze job distribution across machines
    print("\nJOB DISTRIBUTION ANALYSIS:")
    job_machine_count = {}
    for j in range(problem.num_jobs):
        job_machine_count[j] = {m: 0 for m in range(problem.num_machines)}
    
    # First try to analyze using machine_schedules if available
    if hasattr(solution, 'machine_schedules') and solution.machine_schedules:
        # Count operations per job on each machine
        for machine_idx, schedule in enumerate(solution.machine_schedules):
            for op_data in schedule:
                if len(op_data) >= 3:
                    operation = op_data[2]
                    if hasattr(operation, 'job_idx'):
                        job_machine_count[operation.job_idx][machine_idx] += 1
    # Fall back to using schedule_details if machine_schedules isn't available or is empty
    elif hasattr(solution, 'schedule_details') and solution.schedule_details:
        for op, details in solution.schedule_details.items():
            if hasattr(op, 'job_idx') and 'machine' in details:
                job_idx = op.job_idx
                machine_idx = details['machine']
                job_machine_count[job_idx][machine_idx] += 1
    
    # Print distribution
    for job_idx, machine_counts in job_machine_count.items():
        if sum(machine_counts.values()) > 0:  # Skip jobs with no operations
            machines_used = [m for m, count in machine_counts.items() if count > 0]
            print(f"Job {job_idx}: {sum(machine_counts.values())} operations across {len(machines_used)} machines")
            for m, count in machine_counts.items():
                if count > 0:
                    print(f"  - Machine {m}: {count} operations")
    
    # Count if there are any jobs with all operations on one machine
    jobs_on_single_machine = 0
    for job_idx, machine_counts in job_machine_count.items():
        if sum(machine_counts.values()) > 0:  # Skip jobs with no operations
            machines_used = [m for m, count in machine_counts.items() if count > 0]
            if len(machines_used) == 1:
                jobs_on_single_machine += 1
    
    if jobs_on_single_machine > 0:
        print(f"\nWARNING: {jobs_on_single_machine} jobs have all operations on a single machine!")
    
    result = {
        'instance_id': instance_id,
        'makespan': solution.makespan,
        'execution_time': execution_time,
        'pop_size': pop_size,
        'max_iterations': max_iterations,
        'crossover_prob': crossover_prob,
        'mutation_prob': mutation_prob
    }
    
    # Add solution for Gantt charts etc.
    result['solution'] = solution
    
    print(f"  Processed {instance_id}: Makespan: {solution.makespan:.2f}, Time: {execution_time:.2f}s")
    
    return result


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
    
    # Check if multiprocessing is enabled
    use_multiprocessing = cfg.multiprocessing.enabled if hasattr(cfg, 'multiprocessing') else False
    
    if use_multiprocessing:
        try:
            from joblib import Parallel, delayed
            n_jobs = cfg.multiprocessing.n_jobs
            verbose = cfg.multiprocessing.verbose
            
            print(f"🔄 Using parallel processing with {n_jobs} workers")
            
            # Process instances in parallel
            results_with_solutions = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_process_single_instance)(instance_id, problem, cfg)
                for instance_id, problem in instances
            )
            
        except ImportError:
            print("⚠️ joblib not found, falling back to sequential processing")
            use_multiprocessing = False
    
    if not use_multiprocessing:
        # Sequential processing
        print("🔄 Using sequential processing")
        results_with_solutions = []
        
        # Process instances sequentially
        for idx, (instance_id, problem) in enumerate(instances):
            print(f"\n[{idx+1}/{len(instances)}] Evaluating instance: {instance_id}")
            result = _process_single_instance(instance_id, problem, cfg)
            results_with_solutions.append(result)
    
    # Gather convergence histories if needed
    convergence_histories = []
    
    # Extract solutions for visualization
    results = []
    for result in results_with_solutions:
        # Extract solution for convergence history if needed
        if cfg.visualization.enabled and cfg.visualization.convergence_plot:
            solution = result.pop('solution')  # Remove solution from result dict
            if hasattr(solution, 'convergence_history'):
                convergence_histories.append(solution.convergence_history)
        else:
            result.pop('solution', None)  # Remove solution if not needed
            
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
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


def _run_rl_mode(cfg: DictConfig):
    """
    Train and evaluate a PPO agent on POFJSP instances.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Dictionary with results
    """
    print(f"\n🤖 Running RL mode with PPO agent")
    
    # Load dataset
    dataset = POFJSPDataLoader(cfg.dataset)
    problem_instances = dataset.load_instances()
    
    if not problem_instances:
        print("No problem instances found! Check dataset configuration.")
        return {}
    
    print(f"Loaded {len(problem_instances)} problem instances.")
    
    # Prepare results container
    results = []
    
    # Get agent configuration
    agent_cfg = cfg.rl.agent
    training_cfg = cfg.rl.training
    
    # Process each problem instance
    for idx, (instance_id, problem) in enumerate(problem_instances.items()):
        print(f"\nProcessing instance {idx+1}/{len(problem_instances)}: {instance_id}")
        
        # Train PPO agent for this instance
        agent_save_path = None
        if training_cfg.save_models:
            # Create directory if it doesn't exist
            save_dir = Path(hydra.utils.get_original_cwd()) / training_cfg.models_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            agent_save_path = str(save_dir / f"{instance_id}_model.zip")
            
        # Measure training time
        start_time = time.time()
        
        # Train the agent
        agent, solution = train_ppo_agent(
            problem_instance=problem,
            total_timesteps=training_cfg.total_timesteps,
            n_envs=training_cfg.n_envs,
            save_path=agent_save_path
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare result
        result = {
            'instance_id': instance_id,
            'makespan': solution.makespan,
            'execution_time': execution_time,
            'algorithm': 'PPO',
            'solution': solution
        }
        
        # Add job distribution analysis to the results
        job_machine_count = {}
        for j in range(problem.num_jobs):
            job_machine_count[j] = {m: 0 for m in range(problem.num_machines)}
        
        # Count operations per job on each machine
        for op, details in solution.schedule_details.items():
            if hasattr(op, 'job_idx') and 'machine' in details:
                job_idx = op.job_idx
                machine_idx = details['machine']
                job_machine_count[job_idx][machine_idx] += 1
        
        # Add to result
        result['job_distribution'] = job_machine_count
        
        # Print job distribution
        print("\nJOB DISTRIBUTION ANALYSIS:")
        for job_idx, machine_counts in job_machine_count.items():
            if sum(machine_counts.values()) > 0:  # Skip jobs with no operations
                machines_used = [m for m, count in machine_counts.items() if count > 0]
                print(f"Job {job_idx}: {sum(machine_counts.values())} operations across {len(machines_used)} machines")
                for m, count in machine_counts.items():
                    if count > 0:
                        print(f"  - Machine {m}: {count} operations")
        
        # Add result to results list
        results.append(result)
        
        # Visualize solution if requested
        if cfg.rl.visualization.enabled:
            output_dir = Path(hydra.utils.get_original_cwd()) / cfg.rl.visualization.save_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Gantt chart
            visualize_solution(
                solution=solution,
                problem_instance=problem,
                instance_id=instance_id,
                save_dir=str(output_dir),
                title=f"PPO Agent Solution for {instance_id}",
                save_convergence=False
            )
    
    return {
        'results': results,
        'rl_mode': True
    }


if __name__ == "__main__":
    main() 