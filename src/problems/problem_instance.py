import numpy as np
from collections import namedtuple
from typing import List, Dict, Set, Optional, Tuple
import json
import logging

from src.exceptions import (
    InvalidProblemError, ValidationError, PrecedenceConstraintViolationError,
    validate_positive_int
)

logger = logging.getLogger(__name__)

# Import validation utilities
from src.validation import (
    validate_problem_instance_inputs, validate_solution_inputs,
    validate_numeric_stability, SafeOperationWrapper
)

# --- Data Structures ---
Operation = namedtuple('Operation', ['job_idx', 'op_idx_in_job'])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ProblemInstance:
    def __init__(self, 
                 num_jobs: int, 
                 num_machines: int, 
                 num_operations_per_job: List[int], 
                 processing_times: List[np.ndarray], 
                 predecessors_map: Dict[Operation, Set[Operation]], 
                 successors_map: Dict[Operation, Set[Operation]]):
        """
        Initializes the problem instance with comprehensive validation.
        
        Args:
            num_jobs: Total number of jobs
            num_machines: Total number of machines
            num_operations_per_job: Number of operations for each job
            processing_times: Processing times for each operation on each machine
            predecessors_map: Precedence constraints (predecessors)
            successors_map: Precedence constraints (successors)
            
        Raises:
            ValidationError: If input parameters are invalid
            InvalidProblemError: If problem structure is inconsistent
            PrecedenceConstraintViolationError: If precedence constraints form cycles
        """
        # Validate basic parameters
        self.num_jobs = validate_positive_int(num_jobs, "num_jobs")
        self.num_machines = validate_positive_int(num_machines, "num_machines")
        
        # Validate operations per job
        if not isinstance(num_operations_per_job, list) or len(num_operations_per_job) != num_jobs:
            raise ValidationError(f"num_operations_per_job must be list of length {num_jobs}")
        
        for i, ops in enumerate(num_operations_per_job):
            validate_positive_int(ops, f"num_operations_per_job[{i}]")
        
        self.num_operations_per_job = num_operations_per_job
        self.total_operations = sum(num_operations_per_job)
        
        # Validate processing times
        self._validate_processing_times(processing_times)
        self.processing_times = processing_times
        
        # Create all operations
        self.all_operations = []
        for j in range(num_jobs):
            for o in range(num_operations_per_job[j]):
                self.all_operations.append(Operation(j, o))
        
        # Validate and set precedence constraints
        self._validate_precedence_maps(predecessors_map, successors_map)
        self.predecessors_map = predecessors_map
        self.successors_map = successors_map
        
        # Check for cycles in precedence constraints
        self._check_precedence_cycles()
        
        # Validate problem feasibility
        self._validate_feasibility()
        
        logger.info(f"Created problem instance: {num_jobs} jobs, {num_machines} machines, {self.total_operations} operations")
    
    def _validate_processing_times(self, processing_times: List[np.ndarray]) -> None:
        """Validate processing times structure and values."""
        if not isinstance(processing_times, list) or len(processing_times) != self.num_jobs:
            raise ValidationError(f"processing_times must be list of length {self.num_jobs}")
        
        for job_idx, job_times in enumerate(processing_times):
            if not isinstance(job_times, np.ndarray):
                raise ValidationError(f"processing_times[{job_idx}] must be numpy array")
            
            expected_ops = self.num_operations_per_job[job_idx]
            if job_times.shape != (expected_ops, self.num_machines):
                raise ValidationError(
                    f"processing_times[{job_idx}] has shape {job_times.shape}, "
                    f"expected ({expected_ops}, {self.num_machines})"
                )
            
            # Check for negative processing times
            finite_times = job_times[np.isfinite(job_times)]
            if np.any(finite_times < 0):
                raise ValidationError(f"processing_times[{job_idx}] contains negative values")
    
    def _validate_precedence_maps(self, 
                                predecessors_map: Dict[Operation, Set[Operation]], 
                                successors_map: Dict[Operation, Set[Operation]]) -> None:
        """Validate precedence constraint maps."""
        if not isinstance(predecessors_map, dict) or not isinstance(successors_map, dict):
            raise ValidationError("Precedence maps must be dictionaries")
        
        all_ops_set = set(self.all_operations)
        
        # Validate all operations in maps exist
        for op in predecessors_map.keys():
            if op not in all_ops_set:
                raise InvalidProblemError("precedence", f"Unknown operation {op} in predecessors_map")
        
        for op in successors_map.keys():
            if op not in all_ops_set:
                raise InvalidProblemError("precedence", f"Unknown operation {op} in successors_map")
        
        # Validate consistency between predecessor and successor maps
        for op, preds in predecessors_map.items():
            for pred in preds:
                if pred not in all_ops_set:
                    raise InvalidProblemError("precedence", f"Unknown predecessor {pred} for operation {op}")
                
                # Check if successor map is consistent
                if pred in successors_map and op not in successors_map[pred]:
                    raise InvalidProblemError(
                        "precedence", 
                        f"Inconsistent precedence: {pred} -> {op} in predecessors but not in successors"
                    )
    
    def _check_precedence_cycles(self) -> None:
        """Check for cycles in precedence constraints using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {op: WHITE for op in self.all_operations}
        
        def dfs(op: Operation) -> None:
            if colors[op] == GRAY:
                raise PrecedenceConstraintViolationError(
                    op, [op for op, color in colors.items() if color == GRAY]
                )
            
            if colors[op] == WHITE:
                colors[op] = GRAY
                for successor in self.successors_map.get(op, set()):
                    dfs(successor)
                colors[op] = BLACK
        
        for op in self.all_operations:
            if colors[op] == WHITE:
                dfs(op)
    
    def _validate_feasibility(self) -> None:
        """Check if problem has at least one feasible solution."""
        # Check if each operation can be processed on at least one machine
        for job_idx, job_times in enumerate(self.processing_times):
            for op_idx in range(self.num_operations_per_job[job_idx]):
                if np.all(job_times[op_idx, :] == np.inf):
                    op = Operation(job_idx, op_idx)
                    raise InvalidProblemError(
                        "infeasible", 
                        f"Operation {op} cannot be processed on any machine"
                    )
    
    def get_valid_machines(self, operation: Operation) -> List[int]:
        """Get list of machines that can process the given operation."""
        if operation not in set(self.all_operations):
            raise ValidationError(f"Operation {operation} not in problem instance")
        
        job_times = self.processing_times[operation.job_idx]
        valid_machines = []
        
        for machine_idx in range(self.num_machines):
            if job_times[operation.op_idx_in_job, machine_idx] != np.inf:
                valid_machines.append(machine_idx)
        
        return valid_machines
    
    def get_processing_time(self, operation: Operation, machine_idx: int) -> float:
        """Get processing time for operation on specific machine."""
        if operation not in set(self.all_operations):
            raise ValidationError(f"Operation {operation} not in problem instance")
        
        if not 0 <= machine_idx < self.num_machines:
            raise ValidationError(f"Machine index {machine_idx} not in range [0, {self.num_machines})")
        
        return self.processing_times[operation.job_idx][operation.op_idx_in_job, machine_idx]

    @classmethod 
    def from_dict(cls, data: dict) -> 'ProblemInstance':
        """Create problem instance from dictionary data.
        
        Args:
            data: Dictionary containing problem data
            
        Returns:
            ProblemInstance object
            
        Raises:
            ValidationError: If data format is invalid
            InvalidProblemError: If problem data is inconsistent
        """
        # Validate required fields
        required_fields = ['num_jobs', 'num_machines', 'jobs']
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        num_jobs = data['num_jobs']
        num_machines = data['num_machines']
        
        # Validate basic parameters
        validate_positive_int(num_jobs, 'num_jobs')
        validate_positive_int(num_machines, 'num_machines')
        
        # Process jobs data
        if len(data['jobs']) != num_jobs:
            raise ValidationError(f"Expected {num_jobs} jobs, got {len(data['jobs'])}")
        
        num_operations_per_job = []
        processing_times = []
        predecessors_map = {}
        successors_map = {}
        all_operations = []
        
        # Build operations and precedence constraints
        op_id_to_operation = {}  # Map from operation ID to Operation object
        
        for job_data in data['jobs']:
            job_id = job_data['id']
            operations = job_data['operations']
            num_operations_per_job.append(len(operations))
            
            job_processing_times = np.zeros((len(operations), num_machines))
            
            for op_idx, op_data in enumerate(operations):
                op_id = op_data['id']
                proc_times = op_data['processing_times']
                precedence = op_data.get('precedence', [])
                
                # Create operation
                op = Operation(job_id, op_idx)
                all_operations.append(op)
                op_id_to_operation[op_id] = op
                
                # Set processing times
                if len(proc_times) != num_machines:
                    raise ValidationError(f"Operation {op_id} processing times length mismatch")
                
                for machine_idx, proc_time in enumerate(proc_times):
                    if proc_time > 0:  # Only set positive processing times
                        job_processing_times[op_idx, machine_idx] = proc_time
                
                # Initialize maps
                predecessors_map[op] = set()
                successors_map[op] = set()
            
            processing_times.append(job_processing_times)
        
        # Build precedence constraints after all operations are created
        for job_data in data['jobs']:
            for op_data in job_data['operations']:
                op_id = op_data['id']
                precedence = op_data.get('precedence', [])
                
                op = op_id_to_operation[op_id]
                
                for pred_id in precedence:
                    if pred_id in op_id_to_operation:
                        pred_op = op_id_to_operation[pred_id]
                        predecessors_map[op].add(pred_op)
                        successors_map[pred_op].add(op)
        
        return cls(num_jobs, num_machines, num_operations_per_job, 
                  processing_times, predecessors_map, successors_map)

    @classmethod
    def from_json(cls, json_path: str) -> 'ProblemInstance':
        """Load problem instance from JSON file with validation.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            ProblemInstance object
            
        Raises:
            ValidationError: If file format is invalid
            InvalidProblemError: If problem data is inconsistent
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValidationError(f"Error loading JSON file {json_path}: {e}")
        
        # Handle both legacy format (keys at root) and new format (keys under 'problem')
        if 'problem' in data:
            problem_data = data['problem']
        else:
            problem_data = data
        
        # Validate required fields exist
        required_fields = ['num_jobs', 'num_machines', 'num_operations_per_job', 
                          'processing_times', 'predecessors_map']
        for field in required_fields:
            if field not in problem_data:
                raise ValidationError(f"Missing required field '{field}' in JSON data")
        
        try:
            num_jobs = int(problem_data['num_jobs'])
            num_machines = int(problem_data['num_machines'])
            num_operations_per_job = [int(x) for x in problem_data['num_operations_per_job']]
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid numeric data in JSON file: {e}")
        
        # Convert processing times to numpy arrays with validation
        processing_times = []
        try:
            for job_idx, job_proc_times in enumerate(problem_data['processing_times']):
                arr = np.array(job_proc_times, dtype=float)
                arr[arr == float('inf')] = np.inf
                processing_times.append(arr)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid processing times data: {e}")
        
        # Convert predecessors map with validation
        predecessors_map = {}
        successors_map = {}
        
        try:
            for op_key_str, pred_list in problem_data['predecessors_map'].items():
                # Parse operation key
                job_idx, op_idx = map(int, op_key_str.strip('()').split(','))
                op = Operation(job_idx, op_idx)
                
                # Validate operation is valid
                if not (0 <= job_idx < num_jobs):
                    raise InvalidProblemError("precedence", f"Invalid job index {job_idx}")
                if not (0 <= op_idx < num_operations_per_job[job_idx]):
                    raise InvalidProblemError("precedence", f"Invalid operation index {op_idx} for job {job_idx}")
                
                # Parse predecessors
                predecessors = set()
                for pred_str in pred_list:
                    pred_job, pred_op = map(int, pred_str.strip('()').split(','))
                    pred_operation = Operation(pred_job, pred_op)
                    
                    # Validate predecessor is valid
                    if not (0 <= pred_job < num_jobs):
                        raise InvalidProblemError("precedence", f"Invalid predecessor job index {pred_job}")
                    if not (0 <= pred_op < num_operations_per_job[pred_job]):
                        raise InvalidProblemError("precedence", f"Invalid predecessor operation index {pred_op}")
                    
                    predecessors.add(pred_operation)
                
                predecessors_map[op] = predecessors
        except (ValueError, KeyError, IndexError) as e:
            raise ValidationError(f"Error parsing precedence constraints: {e}")
        
        # Build successors map from predecessors map
        all_operations = []
        for j in range(num_jobs):
            for o in range(num_operations_per_job[j]):
                all_operations.append(Operation(j, o))
        
        successors_map = {op: set() for op in all_operations}
        for op, preds in predecessors_map.items():
            for pred in preds:
                successors_map[pred].add(op)
        
        logger.info(f"Loading problem instance from {json_path}")
        return cls(num_jobs, num_machines, num_operations_per_job, 
                  processing_times, predecessors_map, successors_map)
    
    @classmethod
    def from_fjsp_file(cls, file_path: str) -> 'ProblemInstance':
        """Load problem instance from FJSP format file.
        
        Args:
            file_path: Path to FJSP format file
            
        Raises:
            NotImplementedError: FJSP format not yet supported
        """
        raise NotImplementedError("FJSP format loading not implemented yet")
    
    def to_dict(self) -> Dict:
        """Convert problem instance to dictionary for serialization."""
        return {
            'num_jobs': self.num_jobs,
            'num_machines': self.num_machines,
            'num_operations_per_job': self.num_operations_per_job,
            'processing_times': [times.tolist() for times in self.processing_times],
            'predecessors_map': {
                f"({op.job_idx}, {op.op_idx_in_job})": [
                    f"({pred.job_idx}, {pred.op_idx_in_job})" for pred in preds
                ] for op, preds in self.predecessors_map.items()
            }
        }
    
    def __repr__(self) -> str:
        return (f"ProblemInstance(jobs={self.num_jobs}, machines={self.num_machines}, "
                f"operations={self.total_operations})")

class Solution:
    def __init__(self, operation_sequence: List[Operation], machine_assignment: List[int]):
        """
        Represents a solution with validation.
        
        Args:
            operation_sequence: Ordered list of operations
            machine_assignment: Machine assignments for each operation
            
        Raises:
            ValidationError: If solution structure is invalid
        """
        if not isinstance(operation_sequence, list) or not isinstance(machine_assignment, list):
            raise ValidationError("operation_sequence and machine_assignment must be lists")
        
        if len(operation_sequence) != len(machine_assignment):
            raise ValidationError(
                f"Length mismatch: operation_sequence={len(operation_sequence)}, "
                f"machine_assignment={len(machine_assignment)}"
            )
        
        self.operation_sequence = operation_sequence 
        self.machine_assignment = machine_assignment
        self.makespan = float('inf')
        self.schedule_details = {}  # {Operation: {'start_time', 'end_time', 'machine'}}
        self.machine_schedules = []  # Will be initialized in decode_solution
        self.is_valid = True
        self.validation_errors = []
    
    def validate(self, problem: 'ProblemInstance') -> bool:
        """Validate solution against problem constraints."""
        self.validation_errors = []
        self.is_valid = True
        
        # Check if all operations are present
        expected_ops = set(problem.all_operations)
        actual_ops = set(self.operation_sequence)
        
        if expected_ops != actual_ops:
            missing = expected_ops - actual_ops
            extra = actual_ops - expected_ops
            if missing:
                self.validation_errors.append(f"Missing operations: {missing}")
            if extra:
                self.validation_errors.append(f"Extra operations: {extra}")
            self.is_valid = False
        
        # Check machine assignments
        for i, (op, machine) in enumerate(zip(self.operation_sequence, self.machine_assignment)):
            if not isinstance(machine, int) or not 0 <= machine < problem.num_machines:
                self.validation_errors.append(f"Invalid machine {machine} for operation {op} at position {i}")
                self.is_valid = False
                continue
            
            # Check if operation can be processed on assigned machine
            proc_time = problem.get_processing_time(op, machine)
            if proc_time == np.inf:
                self.validation_errors.append(f"Operation {op} cannot be processed on machine {machine}")
                self.is_valid = False
        
        return self.is_valid
    
    def copy(self) -> 'Solution':
        """Create a deep copy of the solution."""
        new_solution = Solution(
            operation_sequence=self.operation_sequence.copy(),
            machine_assignment=self.machine_assignment.copy()
        )
        new_solution.makespan = self.makespan
        new_solution.schedule_details = self.schedule_details.copy()
        new_solution.machine_schedules = [schedule.copy() for schedule in self.machine_schedules]
        return new_solution

    def __lt__(self, other: 'Solution') -> bool:
        """For sorting solutions by makespan."""
        return self.makespan < other.makespan
    
    def __repr__(self) -> str:
        return f"Solution(makespan={self.makespan:.2f}, operations={len(self.operation_sequence)})"