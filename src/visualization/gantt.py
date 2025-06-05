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
        title: Custom title for the chart. If None, a default title will be generated.
        fig_size: Figure size (width, height) in inches
        bar_height: Height of bars in the chart
        save_path: If provided, the chart will be saved to this path
        show_critical_path: Whether to highlight the critical path
        color_by: How to color the operations ('job', 'machine', or 'operation')
        instance_id: Optional instance ID to include in the title
        
    Returns:
        The matplotlib Figure object
    """
    # Extract necessary data from solution
    machine_schedules = solution.machine_schedules
    num_machines = problem_instance.num_machines
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get critical operations if available
    critical_ops = set()
    if show_critical_path and hasattr(solution, 'critical_path'):
        critical_ops = set(solution.critical_path)
    
    # Define color maps
    cmap = plt.cm.get_cmap('tab20', problem_instance.num_jobs)
    job_colors = {j: cmap(j) for j in range(problem_instance.num_jobs)}
    
    cmap_machines = plt.cm.get_cmap('viridis', problem_instance.num_machines)
    machine_colors = {m: cmap_machines(m) for m in range(problem_instance.num_machines)}
    
    # Count for summary warnings
    int_operations_count = 0
    invalid_format_count = 0
    
    # Build a mapping from operation indices to job indices
    # This helps when dealing with integer operations
    op_to_job_map = {}
    
    if hasattr(solution, 'operation_sequence') and hasattr(solution, 'machine_assignment'):
        # If solution has operation_sequence, we can build a more accurate mapping
        for i, op in enumerate(solution.operation_sequence):
            if hasattr(op, 'job_idx'):
                op_to_job_map[i] = op.job_idx
    
    # Job-to-machine count to analyze distribution
    job_machine_count = {}
    for j in range(problem_instance.num_jobs):
        job_machine_count[j] = {m: 0 for m in range(problem_instance.num_machines)}
    
    # First pass: count job distribution across machines
    for machine_idx, schedule in enumerate(machine_schedules):
        for operation_data in schedule:
            if len(operation_data) >= 3:
                _, _, operation = operation_data[0], operation_data[1], operation_data[2]
                
                job_idx = None
                if not isinstance(operation, int) and hasattr(operation, 'job_idx'):
                    job_idx = operation.job_idx
                elif isinstance(operation, int) and operation in op_to_job_map:
                    job_idx = op_to_job_map[operation]
                elif isinstance(operation, int):
                    # Try to infer job index from operation index for sequential operations
                    # Assuming each job has roughly equal number of operations
                    estimated_ops_per_job = problem_instance.total_operations // problem_instance.num_jobs
                    if estimated_ops_per_job > 0:
                        estimated_job_idx = operation // estimated_ops_per_job
                        if estimated_job_idx < problem_instance.num_jobs:
                            job_idx = estimated_job_idx
                            # Remember this mapping for future reference
                            op_to_job_map[operation] = job_idx
                
                if job_idx is not None:
                    job_machine_count[job_idx][machine_idx] += 1
    
    # Calculate job distribution statistics
    job_distribution_info = []
    jobs_on_single_machine = 0
    jobs_distributed = 0
    
    for j, machine_counts in job_machine_count.items():
        # Skip jobs with no operations
        if sum(machine_counts.values()) == 0:
            continue
            
        machines_used = sum(1 for count in machine_counts.values() if count > 0)
        total_ops = sum(machine_counts.values())
        
        if machines_used == 1 and total_ops > 1:
            # All operations of this job are on a single machine
            machine_idx = next(m for m, count in machine_counts.items() if count > 0)
            job_distribution_info.append(f"Job {j}: All {total_ops} ops on Machine {machine_idx}")
            jobs_on_single_machine += 1
        elif machines_used > 1:
            # Operations are distributed
            job_distribution_info.append(f"Job {j}: {total_ops} ops across {machines_used} machines")
            jobs_distributed += 1
    
    # Print job distribution info
    print(f"\nJOB DISTRIBUTION ANALYSIS FOR {instance_id if instance_id else 'SOLUTION'}:")
    if jobs_on_single_machine > 0:
        print(f"WARNING: {jobs_on_single_machine} jobs have all operations on a single machine!")
    print(f"GOOD: {jobs_distributed} jobs have operations distributed across multiple machines")
    
    for info in job_distribution_info:
        print(f"  {info}")
        
    # Plot each operation as a rectangle
    for machine_idx, schedule in enumerate(machine_schedules):
        y_position = num_machines - machine_idx - 1  # Reverse order for visual clarity
        
        for operation_data in schedule:
            # Fix the unpacking issue - ensure we always have at least op_start, op_end, operation
            if len(operation_data) >= 3:
                op_start, op_end, operation = operation_data[0], operation_data[1], operation_data[2]
            else:
                # Skip if we don't have enough data
                invalid_format_count += 1
                continue
            
            # Check if operation is an int (operation ID) instead of an Operation object
            if isinstance(operation, int):
                int_operations_count += 1
                
                # Try to determine the job this operation belongs to
                job_idx = op_to_job_map.get(operation)
                
                # Choose color based on job or machine
                if job_idx is not None and color_by == 'job':
                    color = job_colors[job_idx]
                else:  # Default to machine color if job can't be determined
                    color = machine_colors[machine_idx]
                
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
                
                # Add label
                label_text = f"Op {operation}"
                if job_idx is not None:
                    label_text = f"J{job_idx}-Op{operation}"
                
                ax.text(
                    (op_start + op_end) / 2,
                    y_position,
                    label_text,
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                    color='white'
                )
                continue
            
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
    
    # Print summary warnings instead of individual warnings
    if int_operations_count > 0:
        print(f"Note: Found {int_operations_count} integer operations instead of Operation objects in {instance_id}")
    
    if invalid_format_count > 0:
        print(f"Note: Found {invalid_format_count} operations with invalid data format in {instance_id}")
    
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
    
    # Map of machine indices to colors for integer operations
    cmap_machines = plt.cm.get_cmap('viridis', problem_instance.num_machines)
    machine_colors = {m: cmap_machines(m) for m in range(problem_instance.num_machines)}
    
    max_makespan = max(sol.makespan for sol in solutions.values()) * 1.05
    
    # Count for summary warnings
    total_int_operations = 0
    total_invalid_format = 0
    
    for i, (method_name, solution) in enumerate(solutions.items()):
        if i >= len(axes):
            break  # Safety check
            
        ax = axes[i]
        machine_schedules = solution.machine_schedules
        num_machines = problem_instance.num_machines
        bar_height = 0.8
        
        # Track issues per method
        int_operations_count = 0
        invalid_format_count = 0
        
        # Plot each operation as a rectangle
        for machine_idx, schedule in enumerate(machine_schedules):
            y_position = num_machines - machine_idx - 1  # Reverse order for visual clarity
            
            for operation_data in schedule:
                # Fix the unpacking issue - ensure we always have at least op_start, op_end, operation
                if len(operation_data) >= 3:
                    op_start, op_end, operation = operation_data[0], operation_data[1], operation_data[2]
                else:
                    # Skip if we don't have enough data
                    invalid_format_count += 1
                    continue
                
                # Check if operation is an int (operation ID) instead of an Operation object
                if isinstance(operation, int):
                    int_operations_count += 1
                    # Create a simple colored bar without job-specific information
                    color = machine_colors[machine_idx]
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
                    
                    # Add generic operation ID label if enough space
                    if (op_end - op_start) > max_makespan * 0.05:
                        ax.text(
                            (op_start + op_end) / 2,
                            y_position,
                            f"Op {operation}",
                            ha='center',
                            va='center',
                            fontsize=7,
                            fontweight='bold',
                            color='white'
                        )
                    continue
                
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
        
        # Update total counts
        total_int_operations += int_operations_count
        total_invalid_format += invalid_format_count
        
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
    
    # Print summary warnings
    if total_int_operations > 0:
        print(f"Note: Found {total_int_operations} integer operations instead of Operation objects in comparative visualization")
    
    if total_invalid_format > 0:
        print(f"Note: Found {total_invalid_format} operations with invalid format in comparative visualization")
    
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