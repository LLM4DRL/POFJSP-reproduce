"""
Simple example demonstrating IAOA+GNS algorithm usage.

This script shows how to:
1. Define a problem instance
2. Run the IAOA+GNS algorithm
3. Display and interpret results
"""

import sys
import os
import numpy as np
from collections import namedtuple

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms import (
    ProblemInstance, Solution, Operation,
    iaoa_gns_algorithm, decode_solution
)


def create_simple_problem():
    """Create a simple 2-job, 2-machine POFJSP instance."""
    
    # Problem parameters
    num_jobs = 2
    num_machines = 2
    num_operations_per_job = [2, 2]  # Each job has 2 operations
    
    # Processing times for each job
    # Job 0: Op0 can be done on M0(3) or M1(5), Op1 can be done on M0(6) or not on M1
    # Job 1: Op0 can be done on M0(4) or M1(2), Op1 cannot be done on M0 but M1(7)
    processing_times = [
        np.array([
            [3, 5],        # Job 0, Operation 0: M0=3, M1=5
            [6, np.inf]    # Job 0, Operation 1: M0=6, M1=cannot process
        ]),
        np.array([
            [4, 2],        # Job 1, Operation 0: M0=4, M1=2
            [np.inf, 7]    # Job 1, Operation 1: M0=cannot, M1=7
        ])
    ]
    
    # Precedence constraints (simple sequential within each job)
    predecessors_map = {
        Operation(0, 0): set(),                    # First op of job 0 has no predecessors
        Operation(0, 1): {Operation(0, 0)},        # Second op of job 0 needs first op
        Operation(1, 0): set(),                    # First op of job 1 has no predecessors
        Operation(1, 1): {Operation(1, 0)}         # Second op of job 1 needs first op
    }
    
    # Derive successors from predecessors
    successors_map = {
        Operation(0, 0): {Operation(0, 1)},        # First op of job 0 precedes second
        Operation(0, 1): set(),                    # Last op of job 0 has no successors
        Operation(1, 0): {Operation(1, 1)},        # First op of job 1 precedes second
        Operation(1, 1): set()                     # Last op of job 1 has no successors
    }
    
    return ProblemInstance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )


def create_complex_problem():
    """Create a more complex problem with parallel operations."""
    
    # 3 jobs, 3 machines
    num_jobs = 3
    num_machines = 3
    num_operations_per_job = [4, 3, 3]  # Job 0: 4 ops, Job 1: 3 ops, Job 2: 3 ops
    
    # Processing times (random-like but deterministic for reproducibility)
    processing_times = [
        # Job 0: 4 operations
        np.array([
            [2, 3, 4],      # Op 0
            [3, 2, 5],      # Op 1  
            [1, 4, 2],      # Op 2
            [4, 1, 3]       # Op 3
        ]),
        # Job 1: 3 operations
        np.array([
            [3, 1, 2],      # Op 0
            [2, 3, 1],      # Op 1
            [1, 2, 4]       # Op 2
        ]),
        # Job 2: 3 operations
        np.array([
            [1, 3, 2],      # Op 0
            [4, 1, 3],      # Op 1
            [2, 2, 1]       # Op 2
        ])
    ]
    
    # Complex precedence constraints with some parallel operations
    predecessors_map = {
        # Job 0: Op0 -> (Op1, Op2 in parallel) -> Op3
        Operation(0, 0): set(),
        Operation(0, 1): {Operation(0, 0)},
        Operation(0, 2): {Operation(0, 0)},
        Operation(0, 3): {Operation(0, 1), Operation(0, 2)},
        
        # Job 1: Sequential operations
        Operation(1, 0): set(),
        Operation(1, 1): {Operation(1, 0)},
        Operation(1, 2): {Operation(1, 1)},
        
        # Job 2: Sequential operations  
        Operation(2, 0): set(),
        Operation(2, 1): {Operation(2, 0)},
        Operation(2, 2): {Operation(2, 1)}
    }
    
    # Derive successors map
    successors_map = {op: set() for op in predecessors_map.keys()}
    for op, preds in predecessors_map.items():
        for pred_op in preds:
            successors_map[pred_op].add(op)
    
    return ProblemInstance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )


def print_problem_info(problem):
    """Print basic information about the problem instance."""
    print(f"Problem Instance:")
    print(f"  Jobs: {problem.num_jobs}")
    print(f"  Machines: {problem.num_machines}")
    print(f"  Total operations: {problem.total_operations}")
    print(f"  Operations per job: {problem.num_operations_per_job}")
    print()
    
    print("Processing Times:")
    for j in range(problem.num_jobs):
        print(f"  Job {j}:")
        for o in range(problem.num_operations_per_job[j]):
            times = problem.processing_times[j][o, :]
            times_str = [f"M{m}:{t}" if np.isfinite(t) else f"M{m}:--" 
                        for m, t in enumerate(times)]
            print(f"    Op {o}: {', '.join(times_str)}")
    print()
    
    print("Precedence Constraints:")
    for op, predecessors in problem.predecessors_map.items():
        if predecessors:
            pred_str = ', '.join(f"({p.job_idx},{p.op_idx_in_job})" for p in predecessors)
            print(f"  Op({op.job_idx},{op.op_idx_in_job}) needs: {pred_str}")
        else:
            print(f"  Op({op.job_idx},{op.op_idx_in_job}) has no prerequisites")
    print()


def print_solution_details(solution, problem):
    """Print detailed solution information."""
    if not solution.schedule_details:
        decode_solution(solution, problem)
    
    print(f"Solution Details:")
    print(f"  Makespan: {solution.makespan}")
    print()
    
    print("Operation Sequence and Machine Assignment:")
    for i, (op, machine) in enumerate(zip(solution.operation_sequence, solution.machine_assignment)):
        proc_time = problem.processing_times[op.job_idx][op.op_idx_in_job, machine]
        print(f"  {i+1:2d}. Op({op.job_idx},{op.op_idx_in_job}) -> M{machine} (time: {proc_time})")
    print()
    
    print("Schedule Timeline:")
    scheduled_ops = sorted(solution.schedule_details.items(), 
                          key=lambda x: x[1]['start_time'])
    for op, details in scheduled_ops:
        print(f"  Op({op.job_idx},{op.op_idx_in_job}): "
              f"Start={details['start_time']:4.1f}, "
              f"End={details['end_time']:4.1f}, "
              f"Machine={details['machine']}")
    print()
    
    print("Machine Schedules:")
    for m_idx in range(problem.num_machines):
        print(f"  Machine {m_idx}:")
        if solution.machine_schedules[m_idx]:
            for start_time, end_time, job_idx, op_idx in sorted(solution.machine_schedules[m_idx]):
                print(f"    {start_time:4.1f}-{end_time:4.1f}: Job {job_idx} Op {op_idx}")
        else:
            print("    (idle)")
    print()


def run_simple_example():
    """Run the simple 2-job, 2-machine example."""
    print("=" * 60)
    print("SIMPLE EXAMPLE: 2 Jobs, 2 Machines")
    print("=" * 60)
    
    # Create problem
    problem = create_simple_problem()
    print_problem_info(problem)
    
    # Run algorithm
    print("Running IAOA+GNS algorithm...")
    print("Parameters: pop_size=20, max_iterations=30")
    best_solution = iaoa_gns_algorithm(problem, pop_size=20, max_iterations=30)
    
    # Display results
    print_solution_details(best_solution, problem)


def run_complex_example():
    """Run the more complex example with parallel operations."""
    print("=" * 60)
    print("COMPLEX EXAMPLE: 3 Jobs, 3 Machines with Parallel Operations")
    print("=" * 60)
    
    # Create problem
    problem = create_complex_problem()
    print_problem_info(problem)
    
    # Run algorithm
    print("Running IAOA+GNS algorithm...")
    print("Parameters: pop_size=50, max_iterations=40")
    best_solution = iaoa_gns_algorithm(problem, pop_size=50, max_iterations=40)
    
    # Display results
    print_solution_details(best_solution, problem)


def compare_algorithms():
    """Compare algorithm performance on the same problem."""
    print("=" * 60)
    print("ALGORITHM COMPARISON: Multiple Runs")
    print("=" * 60)
    
    problem = create_complex_problem()
    print("Problem: 3 Jobs, 3 Machines with Parallel Operations")
    print(f"Total operations: {problem.total_operations}")
    print()
    
    # Run algorithm multiple times with different parameters
    configs = [
        {"pop_size": 30, "max_iterations": 20, "name": "Small (30x20)"},
        {"pop_size": 50, "max_iterations": 30, "name": "Medium (50x30)"},
        {"pop_size": 80, "max_iterations": 40, "name": "Large (80x40)"}
    ]
    
    results = []
    for config in configs:
        print(f"Running {config['name']}...")
        solution = iaoa_gns_algorithm(
            problem, 
            pop_size=config['pop_size'], 
            max_iterations=config['max_iterations']
        )
        results.append((config['name'], solution.makespan))
        print(f"  Result: {solution.makespan}")
    
    print("\nComparison Results:")
    print("-" * 40)
    for name, makespan in results:
        print(f"  {name:15s}: {makespan:6.2f}")
    
    best_config, best_makespan = min(results, key=lambda x: x[1])
    print(f"\nBest configuration: {best_config} with makespan {best_makespan}")


if __name__ == "__main__":
    print("POFJSP Algorithm Demonstration")
    print("IAOA+GNS: Improved Arithmetic Optimization Algorithm + Grade Neighborhood Search")
    print()
    
    # Run examples
    try:
        run_simple_example()
        run_complex_example()
        compare_algorithms()
        
        print("=" * 60)
        print("EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Try modifying the problem instances in this script")
        print("2. Experiment with different algorithm parameters")
        print("3. Create your own problem instances using the ProblemInstance class")
        print("4. Explore visualization tools (when available)")
        print("5. Run the test suite: python -m pytest tests/")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("This might be due to missing dependencies or algorithm issues.")
        print("Please check the requirements.txt and run the test suite.")
        import traceback
        traceback.print_exc() 