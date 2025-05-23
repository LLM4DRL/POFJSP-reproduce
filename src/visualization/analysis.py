"""
Analysis and visualization for POFJSP datasets and algorithm performance.

This module provides tools for visualizing dataset characteristics, 
algorithm convergence, and aggregated performance metrics across 
multiple problem instances.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
from pathlib import Path
import json

# Set style for consistent plots
sns.set_style("whitegrid")


def visualize_dataset_distribution(metadata_df: pd.DataFrame,
                                   output_dir: str = "figures/analysis",
                                   file_formats: List[str] = ["png", "pdf"]):
    """
    Visualize the distribution of problems in a dataset.
    
    Args:
        metadata_df: DataFrame containing metadata about problem instances
        output_dir: Directory to save visualizations
        file_formats: File formats to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for different distributions
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
    
    # 1. Distribution by size config
    if 'size_config' in metadata_df.columns:
        size_counts = metadata_df['size_config'].value_counts().sort_index()
        axes[0, 0].bar(size_counts.index, size_counts.values)
        axes[0, 0].set_title("Dataset Distribution by Problem Size")
        axes[0, 0].set_xlabel("Size Configuration")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Distribution by precedence pattern
    if 'precedence_pattern' in metadata_df.columns:
        pattern_counts = metadata_df['precedence_pattern'].value_counts()
        axes[0, 1].bar(pattern_counts.index, pattern_counts.values)
        axes[0, 1].set_title("Dataset Distribution by Precedence Pattern")
        axes[0, 1].set_xlabel("Pattern Type")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Distribution by complexity score
    if 'complexity_score' in metadata_df.columns:
        sns.histplot(metadata_df['complexity_score'], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("Distribution of Complexity Scores")
        axes[1, 0].set_xlabel("Complexity Score")
        axes[1, 0].set_ylabel("Count")
        
        # Add lines for mean and median
        mean_complexity = metadata_df['complexity_score'].mean()
        median_complexity = metadata_df['complexity_score'].median()
        
        axes[1, 0].axvline(mean_complexity, color='red', linestyle='--', 
                          label=f'Mean: {mean_complexity:.2f}')
        axes[1, 0].axvline(median_complexity, color='green', linestyle=':', 
                          label=f'Median: {median_complexity:.2f}')
        axes[1, 0].legend()
    
    # 4. Distribution by total operations
    if 'total_operations' in metadata_df.columns:
        sns.histplot(metadata_df['total_operations'], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title("Distribution of Total Operations")
        axes[1, 1].set_xlabel("Number of Operations")
        axes[1, 1].set_ylabel("Count")
        
        # Add lines for mean and median
        mean_ops = metadata_df['total_operations'].mean()
        median_ops = metadata_df['total_operations'].median()
        
        axes[1, 1].axvline(mean_ops, color='red', linestyle='--', 
                           label=f'Mean: {mean_ops:.1f}')
        axes[1, 1].axvline(median_ops, color='green', linestyle=':', 
                           label=f'Median: {median_ops:.1f}')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save figures
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"dataset_distribution.{fmt}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dataset distribution visualization to {save_path}")
    
    plt.close(fig)


def plot_complexity_characteristics(metadata_df: pd.DataFrame,
                                   output_dir: str = "figures/analysis",
                                   file_formats: List[str] = ["png", "pdf"]):
    """
    Create visualizations showing relationships between problem characteristics and complexity.
    
    Args:
        metadata_df: DataFrame containing metadata about problem instances
        output_dir: Directory to save visualizations
        file_formats: File formats to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with scatter plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
    
    # 1. Complexity vs Total Operations
    if 'complexity_score' in metadata_df.columns and 'total_operations' in metadata_df.columns:
        sns.scatterplot(
            data=metadata_df, 
            x='total_operations', 
            y='complexity_score',
            hue='precedence_pattern' if 'precedence_pattern' in metadata_df.columns else None,
            alpha=0.7,
            ax=axes[0, 0]
        )
        axes[0, 0].set_title("Complexity vs. Total Operations")
        axes[0, 0].set_xlabel("Number of Operations")
        axes[0, 0].set_ylabel("Complexity Score")
    
    # 2. Machine Utilization vs Complexity
    if ('complexity_score' in metadata_df.columns and 
        'machine_utilization' in metadata_df.columns):
        sns.scatterplot(
            data=metadata_df, 
            x='machine_utilization', 
            y='complexity_score',
            hue='size_config' if 'size_config' in metadata_df.columns else None,
            alpha=0.7,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title("Complexity vs. Machine Utilization")
        axes[0, 1].set_xlabel("Machine Utilization")
        axes[0, 1].set_ylabel("Complexity Score")
    
    # 3. Box plots of complexity by precedence pattern
    if ('complexity_score' in metadata_df.columns and 
        'precedence_pattern' in metadata_df.columns):
        sns.boxplot(
            data=metadata_df,
            x='precedence_pattern',
            y='complexity_score',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title("Complexity by Precedence Pattern")
        axes[1, 0].set_xlabel("Precedence Pattern")
        axes[1, 0].set_ylabel("Complexity Score")
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Box plots of complexity by size config
    if ('complexity_score' in metadata_df.columns and 
        'size_config' in metadata_df.columns):
        sns.boxplot(
            data=metadata_df,
            x='size_config',
            y='complexity_score',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title("Complexity by Size Configuration")
        axes[1, 1].set_xlabel("Size Configuration")
        axes[1, 1].set_ylabel("Complexity Score")
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figures
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"complexity_analysis.{fmt}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved complexity analysis to {save_path}")
    
    plt.close(fig)


def plot_algorithm_performance(results_df: pd.DataFrame,
                               output_dir: str = "figures/analysis",
                               file_formats: List[str] = ["png", "pdf"]):
    """
    Visualize algorithm performance across multiple instances.
    
    Args:
        results_df: DataFrame containing algorithm results
        output_dir: Directory to save visualizations
        file_formats: File formats to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a separate figure for each visualization
    
    # 1. Makespan distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(results_df['makespan'], kde=True, ax=ax1)
    ax1.set_title("Distribution of Makespan Values")
    ax1.set_xlabel("Makespan")
    ax1.set_ylabel("Count")
    
    # Add mean and median lines
    mean_makespan = results_df['makespan'].mean()
    median_makespan = results_df['makespan'].median()
    
    ax1.axvline(mean_makespan, color='red', linestyle='--', 
               label=f'Mean: {mean_makespan:.2f}')
    ax1.axvline(median_makespan, color='green', linestyle=':', 
               label=f'Median: {median_makespan:.2f}')
    ax1.legend()
    
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"makespan_distribution.{fmt}")
        fig1.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved makespan distribution to {save_path}")
    
    plt.close(fig1)
    
    # 2. Execution time vs makespan
    if 'execution_time' in results_df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=results_df,
            x='execution_time',
            y='makespan',
            ax=ax2
        )
        ax2.set_title("Makespan vs. Execution Time")
        ax2.set_xlabel("Execution Time (seconds)")
        ax2.set_ylabel("Makespan")
        
        for fmt in file_formats:
            save_path = os.path.join(output_dir, f"makespan_vs_time.{fmt}")
            fig2.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved makespan vs. time plot to {save_path}")
        
        plt.close(fig2)
    
    # 3. Parameter influence on makespan (if params are in results)
    param_columns = [col for col in results_df.columns 
                    if col in ['pop_size', 'max_iterations', 
                              'crossover_prob', 'mutation_prob']]
    
    if param_columns:
        fig3, axes = plt.subplots(nrows=len(param_columns), ncols=1, 
                                 figsize=(10, 5*len(param_columns)))
        
        # Handle case where there's only one parameter
        if len(param_columns) == 1:
            axes = [axes]
            
        for i, param in enumerate(param_columns):
            # Check if we have different parameter values to plot
            if results_df[param].nunique() > 1:
                sns.boxplot(
                    data=results_df,
                    x=param,
                    y='makespan',
                    ax=axes[i]
                )
                axes[i].set_title(f"Makespan vs. {param}")
                axes[i].set_xlabel(param)
                axes[i].set_ylabel("Makespan")
            else:
                axes[i].set_title(f"{param} has only one value: {results_df[param].iloc[0]}")
                axes[i].axis('off')
        
        plt.tight_layout()
        
        for fmt in file_formats:
            save_path = os.path.join(output_dir, f"parameter_influence.{fmt}")
            fig3.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved parameter influence plot to {save_path}")
        
        plt.close(fig3)


def plot_convergence_history(convergence_data: List[List[float]],
                            title: str = "Algorithm Convergence",
                            output_dir: str = "figures/analysis",
                            file_formats: List[str] = ["png", "pdf"]):
    """
    Plot convergence history from multiple runs.
    
    Args:
        convergence_data: List of convergence histories (list of fitness values per iteration)
        title: Plot title
        output_dir: Directory to save visualization
        file_formats: File formats to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each convergence history
    for i, history in enumerate(convergence_data):
        iterations = np.arange(1, len(history) + 1)
        ax.plot(iterations, history, alpha=0.3, linewidth=1, 
                label=f"Run {i+1}" if i < 10 else None)
    
    # If we have multiple runs, compute and plot average convergence
    if len(convergence_data) > 1:
        # Find max length
        max_len = max(len(h) for h in convergence_data)
        
        # Pad shorter histories with their final value
        padded_histories = []
        for h in convergence_data:
            if len(h) < max_len:
                padded = h + [h[-1]] * (max_len - len(h))
            else:
                padded = h
            padded_histories.append(padded)
        
        # Calculate average and std
        history_array = np.array(padded_histories)
        avg_history = np.mean(history_array, axis=0)
        std_history = np.std(history_array, axis=0)
        
        # Plot average with confidence interval
        iterations = np.arange(1, max_len + 1)
        ax.plot(iterations, avg_history, color='red', linewidth=2, 
                label="Average Convergence")
        ax.fill_between(iterations, 
                       avg_history - std_history, 
                       avg_history + std_history, 
                       color='red', alpha=0.3)
    
    # Log scale can be useful for convergence plots
    ax.set_yscale('log')
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Makespan (log scale)")
    ax.set_title(title)
    ax.grid(True)
    
    # Show legend if useful
    if len(convergence_data) > 1:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figures
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"convergence_plot.{fmt}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    
    plt.close(fig)


def plot_reproduction_comparison(results_df: pd.DataFrame,
                                output_dir: str = "figures/analysis",
                                file_formats: List[str] = ["png", "pdf"]):
    """
    Plot comparison between reproduced results and original paper results.
    
    Args:
        results_df: DataFrame with columns 'instance_id', 'makespan', 'reported_makespan'
        output_dir: Directory to save visualization
        file_formats: File formats to save
    """
    if 'reported_makespan' not in results_df.columns:
        print("No reported makespan values available for comparison")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))
    
    # 1. Bar chart comparison
    instances = results_df['instance_id']
    reproduced = results_df['makespan']
    reported = results_df['reported_makespan']
    
    x = np.arange(len(instances))
    width = 0.35
    
    axes[0].bar(x - width/2, reproduced, width, label='Reproduced')
    axes[0].bar(x + width/2, reported, width, label='Reported in Paper')
    
    axes[0].set_ylabel('Makespan')
    axes[0].set_title('Comparison of Reproduced vs. Reported Results')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(instances, rotation=45, ha='right')
    axes[0].legend()
    
    # 2. Difference percentage
    results_df['diff_percent'] = ((results_df['makespan'] - results_df['reported_makespan']) /
                                 results_df['reported_makespan'] * 100)
    
    sns.barplot(data=results_df, x='instance_id', y='diff_percent', ax=axes[1])
    axes[1].set_ylabel('Difference (%)')
    axes[1].set_title('Percentage Difference from Reported Results')
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # Add text with average difference
    avg_diff = results_df['diff_percent'].mean()
    axes[1].text(0.02, 0.95, f"Average difference: {avg_diff:.2f}%",
                transform=axes[1].transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figures
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"reproduction_comparison.{fmt}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reproduction comparison to {save_path}")
    
    plt.close(fig)
    

if __name__ == "__main__":
    # Example usage (for testing)
    
    # 1. Create sample metadata for dataset distribution
    metadata = {
        'instance_id': [f'inst_{i}' for i in range(50)],
        'size_config': np.random.choice(['tiny', 'small', 'medium', 'large'], 50),
        'precedence_pattern': np.random.choice(['sequential', 'parallel', 'assembly', 'mixed'], 50),
        'complexity_score': np.random.normal(100, 30, 50),
        'total_operations': np.random.randint(10, 100, 50),
        'machine_utilization': np.random.uniform(0.5, 0.9, 50)
    }
    
    metadata_df = pd.DataFrame(metadata)
    
    # 2. Create sample results for algorithm performance
    results = {
        'instance_id': [f'inst_{i}' for i in range(30)],
        'makespan': np.random.normal(50, 15, 30),
        'execution_time': np.random.uniform(1, 10, 30),
        'pop_size': np.random.choice([20, 50, 100], 30),
        'max_iterations': np.random.choice([50, 100, 200], 30),
        'crossover_prob': np.random.choice([0.7, 0.8, 0.9], 30),
        'mutation_prob': np.random.choice([0.1, 0.2, 0.3], 30)
    }
    
    results_df = pd.DataFrame(results)
    
    # 3. Create sample convergence histories
    convergence_data = []
    for _ in range(5):
        # Start with a value around 100 and decrease
        start_val = np.random.uniform(90, 110)
        history = [start_val]
        
        for i in range(49):  # 50 iterations total
            # Decay rate gets smaller as iterations increase
            decay = np.random.uniform(0.95, 0.99) 
            history.append(history[-1] * decay)
        
        convergence_data.append(history)
    
    # 4. Test visualization functions
    print("Testing dataset distribution visualization...")
    visualize_dataset_distribution(metadata_df)
    
    print("Testing complexity characteristics visualization...")
    plot_complexity_characteristics(metadata_df)
    
    print("Testing algorithm performance visualization...")
    plot_algorithm_performance(results_df)
    
    print("Testing convergence visualization...")
    plot_convergence_history(convergence_data) 