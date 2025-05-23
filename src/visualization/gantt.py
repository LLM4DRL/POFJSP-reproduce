"""
Gantt chart visualization for POFJSP solutions.

This module provides functions for creating Gantt charts to visualize 
job schedules produced by the IAOA+GNS algorithm.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

# Import from our algorithm module
from algorithms import Solution, ProblemInstance, Operation


def create_gantt_chart(solution: Solution, 
                       problem_instance: ProblemInstance, 
                       title: Optional[str] = None,
                       fig_size: Tuple[int, int] = (12, 8),
                       bar_height: float = 0.8,
                       save_path: Optional[str] = None,
                       show_critical_path: bool = True,
                       color_by: str = 'job',  # 'job', 'machine', or 'operation'
                       instance_id: Optional[str] = None) -> plt.Figure:
    """
    Create a Gantt chart visualization of a schedule solution.
    
    Args:
        solution: A Solution object containing the schedule
        problem_instance: The ProblemInstance used to generate the solution
        title: Custom title for the chart. If None, a default title is used.
        fig_size: Figure size as (width, height) in inches
        bar_height: Height of each bar in the chart
        save_path: Path to save the figure. If None, the figure is not saved.
        show_critical_path: Whether to highlight the critical path in the schedule
        color_by: How to color the bars ('job', 'machine', or 'operation')
        instance_id: Optional identifier for the instance being visualized
        
    Returns:
        A matplotlib figure object
    """
    # Extract necessary data from solution
    machine_schedules = solution.machine_schedules
    
    # Calculate total number of machines
    num_machines = problem_instance.num_machines
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Map of job indices to colors
    cmap = plt.cm.get_cmap('tab20', problem_instance.num_jobs)
    job_colors = {j: cmap(j) for j in range(problem_instance.num_jobs)}
    
    # Map of machine indices to colors
    cmap_machines = plt.cm.get_cmap('viridis', problem_instance.num_machines)
    machine_colors = {m: cmap_machines(m) for m in range(problem_instance.num_machines)}
    
    # Track the operations on the critical path if needed
    critical_ops = set()
    if show_critical_path and hasattr(solution, 'critical_path'):
        critical_ops = set(solution.critical_path)
    
    # Plot each operation as a rectangle
    for machine_idx, schedule in enumerate(machine_schedules):
        y_position = num_machines - machine_idx - 1  # Reverse order for visual clarity
        
        for op_start, op_end, operation in schedule:
            # Choose color based on job or machine
            if color_by == 'job':
                color = job_colors[operation.job_idx]
            elif color_by == 'machine':
                color = machine_colors[machine_idx]
            else:  # color by operation
                color_idx = (operation.job_idx * 10 + operation.op_idx_in_job) % 20
                color = plt.cm.tab20(color_idx)
            
            # Create rectangle
            rect = patches.Rectangle(
                (op_start, y_position - bar_height/2),
                op_end - op_start,
                bar_height,
                edgecolor='black',
                facecolor=color,
                alpha=0.8,
                linewidth=1
            )
            ax.add_patch(rect)
            
            # Add label in the middle of the rectangle
            ax.text(
                (op_start + op_end) / 2,
                y_position,
                f"J{operation.job_idx}-O{operation.op_idx_in_job}",
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='white'
            )
            
            # Highlight critical operations
            if operation in critical_ops:
                rect_critical = patches.Rectangle(
                    (op_start, y_position - bar_height/2),
                    op_end - op_start,
                    bar_height,
                    edgecolor='red',
                    fill=False,
                    linewidth=2,
                    linestyle='--'
                )
                ax.add_patch(rect_critical)
    
    # Set chart properties
    ax.set_xlim(0, solution.makespan * 1.05)  # Add some padding
    ax.set_ylim(-0.5, num_machines - 0.5 + bar_height/2)
    
    # Set y-axis ticks and labels
    ax.set_yticks(list(range(num_machines)))
    ax.set_yticklabels([f'Machine {num_machines - i - 1}' for i in range(num_machines)])
    
    # Set x-axis label
    ax.set_xlabel('Time')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        makespan_str = f"Makespan: {solution.makespan:.2f}"
        if instance_id:
            ax.set_title(f"Schedule for {instance_id}\n{makespan_str}")
        else:
            ax.set_title(f"Job Shop Schedule\n{makespan_str}")
    
    # Add grid lines
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Create legend for jobs
    job_patches = [patches.Patch(color=job_colors[j], label=f'Job {j}') 
                  for j in range(problem_instance.num_jobs)]
    
    # Add critical path to legend if shown
    if show_critical_path and hasattr(solution, 'critical_path'):
        critical_patch = patches.Patch(edgecolor='red', fill=False, 
                                     label='Critical Path', linestyle='--')
        job_patches.append(critical_patch)
    
    ax.legend(handles=job_patches, loc='upper right', bbox_to_anchor=(1.02, 1), 
             fontsize='small')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_solution(solution: Solution, 
                       problem_instance: ProblemInstance,
                       instance_id: str,
                       output_dir: str = 'figures/gantt',
                       file_formats: List[str] = ['png', 'pdf'],
                       show_fig: bool = False):
    """
    Visualize a solution and save to specified formats.
    
    Args:
        solution: Solution object
        problem_instance: Problem instance
        instance_id: Instance identifier
        output_dir: Directory to save visualizations
        file_formats: File formats to save (e.g., ['png', 'pdf'])
        show_fig: Whether to display the figure
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig = create_gantt_chart(
        solution=solution,
        problem_instance=problem_instance,
        instance_id=instance_id,
        show_critical_path=True
    )
    
    # Save in each requested format
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"{instance_id}_gantt.{fmt}")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Gantt chart to {save_path}")
    
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def visualize_comparative_schedules(solutions: Dict[str, Solution],
                                   problem_instance: ProblemInstance,
                                   instance_id: str,
                                   output_dir: str = 'figures/comparison',
                                   file_formats: List[str] = ['png', 'pdf']):
    """
    Create comparative visualization of multiple scheduling solutions.
    
    Args:
        solutions: Dictionary mapping algorithm/method names to Solution objects
        problem_instance: Problem instance
        instance_id: Instance identifier
        output_dir: Directory to save visualizations
        file_formats: File formats to save
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    num_solutions = len(solutions)
    if num_solutions == 0:
        return
    
    # Create a grid layout based on the number of solutions
    if num_solutions <= 2:
        nrows, ncols = 1, num_solutions
    else:
        nrows = (num_solutions + 1) // 2  # Ceiling division
        ncols = 2
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6*nrows))
    if num_solutions == 1:
        axes = [axes]  # Make iterable for consistency
    
    # Flatten axes if it's a multi-dimensional array
    if num_solutions > 2:
        axes = axes.flatten()
    
    # Map of job indices to colors
    cmap = plt.cm.get_cmap('tab20', problem_instance.num_jobs)
    job_colors = {j: cmap(j) for j in range(problem_instance.num_jobs)}
    
    max_makespan = max(sol.makespan for sol in solutions.values()) * 1.05
    
    for i, (method_name, solution) in enumerate(solutions.items()):
        if i >= len(axes):
            break  # Safety check
            
        ax = axes[i]
        machine_schedules = solution.machine_schedules
        num_machines = problem_instance.num_machines
        bar_height = 0.8
        
        # Plot each operation as a rectangle
        for machine_idx, schedule in enumerate(machine_schedules):
            y_position = num_machines - machine_idx - 1  # Reverse order for visual clarity
            
            for op_start, op_end, operation in schedule:
                # Color by job
                color = job_colors[operation.job_idx]
                
                # Create rectangle
                rect = patches.Rectangle(
                    (op_start, y_position - bar_height/2),
                    op_end - op_start,
                    bar_height,
                    edgecolor='black',
                    facecolor=color,
                    alpha=0.8,
                    linewidth=1
                )
                ax.add_patch(rect)
                
                # Add label if rectangle is wide enough
                if (op_end - op_start) > max_makespan * 0.05:  # Only add text if enough space
                    ax.text(
                        (op_start + op_end) / 2,
                        y_position,
                        f"J{operation.job_idx}-O{operation.op_idx_in_job}",
                        ha='center',
                        va='center',
                        fontsize=7,
                        fontweight='bold',
                        color='white'
                    )
        
        # Set chart properties
        ax.set_xlim(0, max_makespan)
        ax.set_ylim(-0.5, num_machines - 0.5 + bar_height/2)
        
        # Set y-axis ticks and labels
        ax.set_yticks(list(range(num_machines)))
        ax.set_yticklabels([f'M{num_machines - i - 1}' for i in range(num_machines)])
        
        # Set x-axis label
        ax.set_xlabel('Time')
        
        # Set title
        ax.set_title(f"{method_name}\nMakespan: {solution.makespan:.2f}")
        
        # Add grid lines
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Hide any unused subplot
    for i in range(num_solutions, len(axes)):
        axes[i].axis('off')
    
    # Add a common legend
    job_patches = [patches.Patch(color=job_colors[j], label=f'Job {j}') 
                  for j in range(problem_instance.num_jobs)]
    
    fig.legend(handles=job_patches, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=min(problem_instance.num_jobs, 5), fontsize='small')
    
    # Set a common super title
    plt.suptitle(f"Comparative Schedules for {instance_id}", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1)
    
    # Save in each requested format
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"{instance_id}_comparison.{fmt}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison chart to {save_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    # Example usage (for testing)
    from algorithms import iaoa_gns_algorithm
    
    # Create a simple test problem
    num_jobs = 3
    num_machines = 3
    num_ops_per_job = [2, 3, 2]
    processing_times = [
        np.array([[5.0, np.inf, 3.0], [np.inf, 2.0, 4.0]]),  # Job 0
        np.array([[1.0, 4.0, np.inf], [5.0, np.inf, 3.0], [np.inf, 2.0, 1.0]]),  # Job 1
        np.array([[2.0, 3.0, np.inf], [np.inf, 4.0, 2.0]]),  # Job 2
    ]
    
    # Create simple precedence constraints
    predecessors_map = {}
    successors_map = {}
    
    for j in range(num_jobs):
        for o in range(num_ops_per_job[j]):
            op = Operation(j, o)
            
            if o == 0:
                # First operation has no predecessors
                predecessors_map[op] = set()
            else:
                # Each operation depends on previous operation in same job
                predecessors_map[op] = {Operation(j, o-1)}
            
            if o == num_ops_per_job[j] - 1:
                # Last operation has no successors
                successors_map[op] = set()
            else:
                # Each operation is followed by next operation in same job
                successors_map[op] = {Operation(j, o+1)}
    
    # Create problem instance
    problem = ProblemInstance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_operations_per_job=num_ops_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )
    
    # Solve with algorithm
    solution = iaoa_gns_algorithm(
        problem_instance=problem, 
        pop_size=20, 
        max_iterations=10
    )
    
    # Visualize
    visualize_solution(
        solution=solution,
        problem_instance=problem,
        instance_id="test_instance",
        show_fig=True
    ) 