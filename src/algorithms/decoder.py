import numpy as np
import heapq
from typing import List, Dict, Tuple, Optional, Set
import logging

from exceptions import (
    InvalidMachineAssignmentError, PrecedenceConstraintViolationError,
    ValidationError
)

logger = logging.getLogger(__name__)

# Import validation utilities
from validation import (
    validate_inputs, Validators, validate_numeric_stability, SafeOperationWrapper
)

@SafeOperationWrapper("decode_solution")
@validate_inputs(
    solution=lambda x: x is not None,
    problem=lambda x: x is not None and hasattr(x, 'num_machines'),
    verbose=lambda x: isinstance(x, bool)
)
def decode_solution(solution, problem, verbose: bool = False) -> Tuple[float, Dict, List]:
    """
    Decodes a solution to calculate makespan and schedule details with validation.
    
    Args:
        solution: Solution object with operation sequence and machine assignments
        problem: Problem instance with constraints and processing times
        verbose: Enable detailed logging
        
    Returns:
        Tuple of (makespan, schedule_details, machine_schedules)
        
    Raises:
        ValidationError: If solution or problem is invalid
        InvalidMachineAssignmentError: If operation assigned to invalid machine
        PrecedenceConstraintViolationError: If precedence constraints violated
    """
    # Input validation
    if solution is None or problem is None:
        raise ValidationError("solution and problem cannot be None")
    
    if not hasattr(solution, 'operation_sequence') or not hasattr(solution, 'machine_assignment'):
        raise ValidationError("solution must have operation_sequence and machine_assignment")
    
    if len(solution.operation_sequence) != len(solution.machine_assignment):
        raise ValidationError(
            f"Sequence length {len(solution.operation_sequence)} != assignment length {len(solution.machine_assignment)}"
        )
    
    try:
        return _decode_solution_implementation(solution, problem, verbose)
    except Exception as e:
        logger.error(f"Decoding failed: {e}")
        # Return safe fallback values
        solution.makespan = float('inf')
        solution.schedule_details = {}
        solution.machine_schedules = [[] for _ in range(problem.num_machines)]
        return float('inf'), {}, [[] for _ in range(problem.num_machines)]


def _decode_solution_implementation(solution, problem, verbose: bool) -> Tuple[float, Dict, List]:
    # Use safe operation conversion with validation
    try:
        operation_objects = _convert_operations_safely(solution.operation_sequence, problem)
    except Exception as e:
        raise ValidationError(f"Failed to convert operations: {e}")
    
    # Validate all machine assignments
    for i, machine_idx in enumerate(solution.machine_assignment):
        if not isinstance(machine_idx, int) or not (0 <= machine_idx < problem.num_machines):
            op = operation_objects[i] if i < len(operation_objects) else f"op_{i}"
            raise InvalidMachineAssignmentError(op, machine_idx, "Invalid machine index")
    
    # schedule_details: {SimpleOperation: {'start_time', 'end_time', 'machine'}}
    schedule_details = {}
    # machine_schedules: list of lists, machine_schedules[m] = sorted list of (start_time, end_time)
    machine_schedules = [[] for _ in range(problem.num_machines)]

    # Keep track of completion times of operations for precedence constraints
    operation_completion_times = {}
    scheduled_operations = set()

    # Create a graph of operation dependencies
    in_degree = {op: 0 for op in operation_objects}
    op_to_predecessors = {}
    
    for op in operation_objects:
        op_key = (op.job_idx, op.op_idx_in_job)
        if op_key in problem.predecessors_map:
            op_to_predecessors[op] = problem.predecessors_map[op_key]
            in_degree[op] = len(problem.predecessors_map[op_key])
        else:
            op_to_predecessors[op] = []
    
    # Validate precedence constraints before processing
    _validate_precedence_constraints(operation_objects, problem)
    
    # Find operations with no predecessors
    ready_operations = [op for op in operation_objects if in_degree[op] == 0]
    
    if not ready_operations:
        raise PrecedenceConstraintViolationError(
            "No operations without predecessors", list(operation_objects)
        )
    
    # Process operations in topological order
    processed_ops = 0
    max_iterations = len(operation_objects) * 2  # Prevent infinite loops
    iterations = 0
    
    while ready_operations and processed_ops < len(operation_objects) and iterations < max_iterations:
        iterations += 1
        current_op = ready_operations.pop(0)
        
        if current_op in scheduled_operations:
            continue
        
        # Find its index in the operation sequence
        op_idx = operation_objects.index(current_op)
        assigned_machine = solution.machine_assignment[op_idx]
        proc_time = problem.processing_times[current_op.job_idx][current_op.op_idx_in_job, assigned_machine]

        # Validate machine assignment
        if not (0 <= assigned_machine < problem.num_machines):
            raise InvalidMachineAssignmentError(
                current_op, assigned_machine, "Machine index out of range"
            )
        
        if proc_time == np.inf or proc_time < 0:
            raise InvalidMachineAssignmentError(
                current_op, assigned_machine, proc_time
            )

        # Determine earliest start time based on predecessors
        earliest_start_due_to_predecessors = 0
        if current_op in op_to_predecessors:
            for pred_op_key in op_to_predecessors[current_op]:
                pred_job, pred_op = pred_op_key
                # Find the corresponding SimpleOperation
                pred_op_obj = None
                for op_obj in operation_objects:
                    if op_obj.job_idx == pred_job and op_obj.op_idx_in_job == pred_op:
                        pred_op_obj = op_obj
                        break
                if pred_op_obj and pred_op_obj in operation_completion_times:
                    earliest_start_due_to_predecessors = max(
                        earliest_start_due_to_predecessors,
                        operation_completion_times[pred_op_obj]
                    )
        
        # Sort machine schedule by start times to find gaps
        machine_schedules[assigned_machine].sort()

        # Try to insert in existing gaps
        last_finish_time_on_machine = 0
        inserted = False
        for j in range(len(machine_schedules[assigned_machine])):
            gap_start = last_finish_time_on_machine
            gap_end = machine_schedules[assigned_machine][j][0]
            
            possible_start_in_gap = max(earliest_start_due_to_predecessors, gap_start)
            if possible_start_in_gap + proc_time <= gap_end:
                op_start_time = possible_start_in_gap
                inserted = True
                break
            last_finish_time_on_machine = machine_schedules[assigned_machine][j][1]

        if not inserted:
            # If no suitable gap, schedule after the last operation on the machine
            op_start_time = max(earliest_start_due_to_predecessors, last_finish_time_on_machine)

        op_end_time = op_start_time + proc_time
        
        # Update schedules
        machine_schedules[assigned_machine].append((op_start_time, op_end_time))
        machine_schedules[assigned_machine].sort()

        schedule_details[current_op] = {
            'start_time': op_start_time,
            'end_time': op_end_time,
            'machine': assigned_machine
        }
        operation_completion_times[current_op] = op_end_time
        scheduled_operations.add(current_op)
        
        # Update successors' in_degree and add to ready queue
        op_key = (current_op.job_idx, current_op.op_idx_in_job)
        if op_key in problem.successors_map:
            for succ_op_key in problem.successors_map[op_key]:
                succ_job, succ_op = succ_op_key
                # Find the corresponding SimpleOperation
                succ_op_obj = None
                for op_obj in operation_objects:
                    if op_obj.job_idx == succ_job and op_obj.op_idx_in_job == succ_op:
                        succ_op_obj = op_obj
                        break
                if succ_op_obj and succ_op_obj in in_degree:
                    in_degree[succ_op_obj] -= 1
                    if in_degree[succ_op_obj] == 0:
                        ready_operations.append(succ_op_obj)
        
        processed_ops += 1
    
    # Check if all operations were processed
    if processed_ops < len(operation_objects):
        if verbose:
            print(f"WARNING: Not all operations were processed.")
    
    makespan = 0
    if operation_completion_times:
        makespan = max(operation_completion_times.values())
    
    # For GNS, it's useful to have machine schedules also store op info
    final_machine_schedules_detailed = [[] for _ in range(problem.num_machines)]
    for op, details in schedule_details.items():
        final_machine_schedules_detailed[details['machine']].append(
            (details['start_time'], details['end_time'], op)
        )
    for m_idx in range(problem.num_machines):
        final_machine_schedules_detailed[m_idx].sort()

    solution.makespan = makespan
    solution.schedule_details = schedule_details
    solution.machine_schedules = final_machine_schedules_detailed
    return makespan, schedule_details, final_machine_schedules_detailed


class SimpleOperation:
    """Simple operation class for consistent handling."""
    
    def __init__(self, job_idx: int, op_idx_in_job: int):
        self.job_idx = job_idx
        self.op_idx_in_job = op_idx_in_job
    
    def __hash__(self):
        return hash((self.job_idx, self.op_idx_in_job))
    
    def __eq__(self, other):
        if not isinstance(other, SimpleOperation):
            return False
        return (self.job_idx, self.op_idx_in_job) == (other.job_idx, other.op_idx_in_job)
    
    def __repr__(self):
        return f"Operation({self.job_idx}, {self.op_idx_in_job})"


def _convert_operations_safely(operation_sequence, problem) -> List[SimpleOperation]:
    """Convert operation sequence to SimpleOperation objects with validation."""
    operation_objects = []
    
    for i, op in enumerate(operation_sequence):
        try:
            if hasattr(op, 'job_idx') and hasattr(op, 'op_idx_in_job'):
                job_idx, op_idx = op.job_idx, op.op_idx_in_job
            elif isinstance(op, (tuple, list)) and len(op) == 2:
                job_idx, op_idx = op[0], op[1]
            else:
                raise ValidationError(f"Invalid operation format at index {i}: {op}")
            
            # Validate indices
            if not isinstance(job_idx, int) or not isinstance(op_idx, int):
                raise ValidationError(f"Operation indices must be integers: {op}")
            
            if not (0 <= job_idx < problem.num_jobs):
                raise ValidationError(f"Invalid job index {job_idx} at position {i}")
            
            if not (0 <= op_idx < problem.num_operations_per_job[job_idx]):
                raise ValidationError(f"Invalid operation index {op_idx} for job {job_idx} at position {i}")
            
            operation_objects.append(SimpleOperation(job_idx, op_idx))
            
        except Exception as e:
            raise ValidationError(f"Failed to convert operation at index {i}: {e}")
    
    return operation_objects


def _validate_precedence_constraints(operation_objects: List[SimpleOperation], problem) -> None:
    """Validate that precedence constraints can be satisfied."""
    operation_set = {(op.job_idx, op.op_idx_in_job) for op in operation_objects}
    
    # Check if all required operations are present
    expected_operations = {(op.job_idx, op.op_idx_in_job) for op in problem.all_operations}
    
    if operation_set != expected_operations:
        missing = expected_operations - operation_set
        extra = operation_set - expected_operations
        
        error_msg = []
        if missing:
            error_msg.append(f"Missing operations: {missing}")
        if extra:
            error_msg.append(f"Extra operations: {extra}")
        
        raise ValidationError("; ".join(error_msg))
    
    # Check for unsatisfiable precedence constraints
    for op_key, predecessors in problem.predecessors_map.items():
        if op_key not in operation_set:
            continue
        
        for pred_key in predecessors:
            if pred_key not in operation_set:
                raise PrecedenceConstraintViolationError(
                    f"Operation {op_key}", 
                    [f"Missing predecessor {pred_key}"]
                )