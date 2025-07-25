"""
Comprehensive tests for problem instance validation and functionality.

These tests ensure problem instances are created correctly with
proper validation and constraint checking.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from src.problems.problem_instance import ProblemInstance, Operation, Solution
from src.exceptions import (
    ValidationError, InvalidProblemError, PrecedenceConstraintViolationError
)


class TestProblemInstanceCreation:
    """Test problem instance creation and validation."""
    
    def test_valid_problem_creation(self):
        """Test creation of valid problem instance."""
        num_jobs = 2
        num_machines = 2
        num_operations_per_job = [2, 2]
        
        processing_times = [
            np.array([[1.0, 2.0], [3.0, 1.0]]),  # Job 0
            np.array([[2.0, 1.0], [1.0, 3.0]])   # Job 1
        ]
        
        predecessors_map = {
            Operation(0, 1): {Operation(0, 0)},
            Operation(1, 1): {Operation(1, 0)}
        }
        
        successors_map = {
            Operation(0, 0): {Operation(0, 1)},
            Operation(1, 0): {Operation(1, 1)}
        }
        
        problem = ProblemInstance(
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_operations_per_job=num_operations_per_job,
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
        
        assert problem.num_jobs == 2
        assert problem.num_machines == 2
        assert problem.total_operations == 4
        assert len(problem.all_operations) == 4
    
    def test_invalid_num_jobs(self):
        """Test validation of invalid number of jobs."""
        with pytest.raises(ValidationError, match="num_jobs"):
            ProblemInstance(
                num_jobs=0,  # Invalid
                num_machines=2,
                num_operations_per_job=[2],
                processing_times=[np.array([[1.0, 2.0]])],
                predecessors_map={},
                successors_map={}
            )
    
    def test_invalid_num_machines(self):
        """Test validation of invalid number of machines."""
        with pytest.raises(ValidationError, match="num_machines"):
            ProblemInstance(
                num_jobs=1,
                num_machines=-1,  # Invalid
                num_operations_per_job=[2],
                processing_times=[np.array([[1.0, 2.0]])],
                predecessors_map={},
                successors_map={}
            )
    
    def test_invalid_operations_per_job_length(self):
        """Test validation of operations per job list length."""
        with pytest.raises(ValidationError, match="num_operations_per_job"):
            ProblemInstance(
                num_jobs=2,
                num_machines=2,
                num_operations_per_job=[2],  # Length doesn't match num_jobs
                processing_times=[np.array([[1.0, 2.0]])],
                predecessors_map={},
                successors_map={}
            )
    
    def test_invalid_operations_per_job_values(self):
        """Test validation of operations per job values."""
        with pytest.raises(ValidationError, match="num_operations_per_job"):
            ProblemInstance(
                num_jobs=1,
                num_machines=2,
                num_operations_per_job=[0],  # Invalid: must be positive
                processing_times=[np.array([[1.0, 2.0]])],
                predecessors_map={},
                successors_map={}
            )
    
    def test_invalid_processing_times_length(self):
        """Test validation of processing times list length."""
        with pytest.raises(ValidationError, match="processing_times"):
            ProblemInstance(
                num_jobs=2,
                num_machines=2,
                num_operations_per_job=[2, 2],
                processing_times=[np.array([[1.0, 2.0]])],  # Length doesn't match num_jobs
                predecessors_map={},
                successors_map={}
            )
    
    def test_invalid_processing_times_shape(self):
        """Test validation of processing times matrix shape."""
        with pytest.raises(ValidationError, match="processing_times"):
            ProblemInstance(
                num_jobs=1,
                num_machines=2,
                num_operations_per_job=[2],
                processing_times=[np.array([[1.0, 2.0, 3.0]])],  # Wrong shape: should be 2x2
                predecessors_map={},
                successors_map={}
            )
    
    def test_negative_processing_times(self):
        """Test validation of negative processing times."""
        with pytest.raises(ValidationError, match="negative values"):
            ProblemInstance(
                num_jobs=1,
                num_machines=2,
                num_operations_per_job=[2],
                processing_times=[np.array([[1.0, -2.0], [3.0, 1.0]])],  # Negative time
                predecessors_map={},
                successors_map={}
            )
    
    def test_precedence_cycle_detection(self):
        """Test detection of cycles in precedence constraints."""
        processing_times = [np.array([[1.0, 2.0], [3.0, 1.0]])]
        
        # Create a cycle: op(0,0) -> op(0,1) -> op(0,0)
        predecessors_map = {
            Operation(0, 1): {Operation(0, 0)},
            Operation(0, 0): {Operation(0, 1)}  # Creates cycle
        }
        
        successors_map = {
            Operation(0, 0): {Operation(0, 1)},
            Operation(0, 1): {Operation(0, 0)}
        }
        
        with pytest.raises(PrecedenceConstraintViolationError):
            ProblemInstance(
                num_jobs=1,
                num_machines=2,
                num_operations_per_job=[2],
                processing_times=processing_times,
                predecessors_map=predecessors_map,
                successors_map=successors_map
            )
    
    def test_infeasible_operation_detection(self):
        """Test detection of operations that cannot be processed."""
        processing_times = [
            np.array([[np.inf, np.inf], [1.0, 2.0]])  # First operation cannot be processed
        ]
        
        with pytest.raises(InvalidProblemError, match="cannot be processed"):
            ProblemInstance(
                num_jobs=1,
                num_machines=2,
                num_operations_per_job=[2],
                processing_times=processing_times,
                predecessors_map={},
                successors_map={}
            )
    
    def test_unknown_operation_in_predecessors(self):
        """Test validation of unknown operations in precedence maps."""
        processing_times = [np.array([[1.0, 2.0]])]
        
        # Reference unknown operation
        predecessors_map = {
            Operation(0, 0): {Operation(1, 0)}  # Job 1 doesn't exist
        }
        
        with pytest.raises(InvalidProblemError, match="Unknown operation"):
            ProblemInstance(
                num_jobs=1,
                num_machines=2,
                num_operations_per_job=[1],
                processing_times=processing_times,
                predecessors_map=predecessors_map,
                successors_map={}
            )


class TestProblemInstanceMethods:
    """Test problem instance methods."""
    
    def test_get_valid_machines(self, simple_problem_instance):
        """Test getting valid machines for an operation."""
        op = Operation(0, 0)
        valid_machines = simple_problem_instance.get_valid_machines(op)
        
        # All machines should be valid for this operation (no inf times)
        assert len(valid_machines) == 3
        assert valid_machines == [0, 1, 2]
    
    def test_get_valid_machines_with_inf(self):
        """Test getting valid machines when some have infinite processing time."""
        processing_times = [
            np.array([[1.0, np.inf, 2.0]])  # Machine 1 cannot process this operation
        ]
        
        problem = ProblemInstance(
            num_jobs=1,
            num_machines=3,
            num_operations_per_job=[1],
            processing_times=processing_times,
            predecessors_map={},
            successors_map={}
        )
        
        op = Operation(0, 0)
        valid_machines = problem.get_valid_machines(op)
        
        assert valid_machines == [0, 2]  # Machine 1 excluded
    
    def test_get_valid_machines_invalid_operation(self, simple_problem_instance):
        """Test error handling for invalid operation."""
        invalid_op = Operation(10, 0)  # Job 10 doesn't exist
        
        with pytest.raises(ValidationError, match="not in problem instance"):
            simple_problem_instance.get_valid_machines(invalid_op)
    
    def test_get_processing_time(self, simple_problem_instance):
        """Test getting processing time for operation on machine."""
        op = Operation(0, 0)
        proc_time = simple_problem_instance.get_processing_time(op, 0)
        
        assert proc_time == 1.0
    
    def test_get_processing_time_invalid_machine(self, simple_problem_instance):
        """Test error handling for invalid machine index."""
        op = Operation(0, 0)
        
        with pytest.raises(ValidationError, match="Machine index"):
            simple_problem_instance.get_processing_time(op, 10)
    
    def test_to_dict(self, simple_problem_instance):
        """Test converting problem instance to dictionary."""
        problem_dict = simple_problem_instance.to_dict()
        
        assert 'num_jobs' in problem_dict
        assert 'num_machines' in problem_dict
        assert 'num_operations_per_job' in problem_dict
        assert 'processing_times' in problem_dict
        assert 'predecessors_map' in problem_dict
        
        assert problem_dict['num_jobs'] == 3
        assert problem_dict['num_machines'] == 3


class TestJSONLoading:
    """Test loading problem instances from JSON files."""
    
    def test_load_valid_json(self, temp_dir):
        """Test loading valid JSON problem instance."""
        problem_data = {
            "num_jobs": 2,
            "num_machines": 2,
            "num_operations_per_job": [2, 2],
            "processing_times": [
                [[1.0, 2.0], [3.0, 1.0]],
                [[2.0, 1.0], [1.0, 3.0]]
            ],
            "predecessors_map": {
                "(0, 1)": ["(0, 0)"],
                "(1, 1)": ["(1, 0)"]
            }
        }
        
        json_file = temp_dir / "test_problem.json"
        with open(json_file, 'w') as f:
            json.dump(problem_data, f)
        
        problem = ProblemInstance.from_json(str(json_file))
        
        assert problem.num_jobs == 2
        assert problem.num_machines == 2
        assert problem.total_operations == 4
    
    def test_load_json_with_nested_problem(self, temp_dir):
        """Test loading JSON with nested 'problem' key."""
        data = {
            "problem": {
                "num_jobs": 1,
                "num_machines": 2,
                "num_operations_per_job": [1],
                "processing_times": [[[1.0, 2.0]]],
                "predecessors_map": {}
            }
        }
        
        json_file = temp_dir / "nested_problem.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        problem = ProblemInstance.from_json(str(json_file))
        assert problem.num_jobs == 1
    
    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(ValidationError, match="Error loading JSON"):
            ProblemInstance.from_json("nonexistent_file.json")
    
    def test_load_invalid_json(self, temp_dir):
        """Test error handling for invalid JSON."""
        json_file = temp_dir / "invalid.json"
        with open(json_file, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(ValidationError, match="Invalid YAML"):
            ProblemInstance.from_json(str(json_file))
    
    def test_load_missing_required_fields(self, temp_dir):
        """Test error handling for missing required fields."""
        problem_data = {
            "num_jobs": 2,
            # Missing num_machines and other required fields
        }
        
        json_file = temp_dir / "incomplete.json"
        with open(json_file, 'w') as f:
            json.dump(problem_data, f)
        
        with pytest.raises(ValidationError, match="Missing required field"):
            ProblemInstance.from_json(str(json_file))
    
    def test_load_invalid_precedence_constraints(self, temp_dir):
        """Test error handling for invalid precedence constraints in JSON."""
        problem_data = {
            "num_jobs": 1,
            "num_machines": 2,
            "num_operations_per_job": [1],
            "processing_times": [[[1.0, 2.0]]],
            "predecessors_map": {
                "(0, 0)": ["(1, 0)"]  # References non-existent job 1
            }
        }
        
        json_file = temp_dir / "invalid_precedence.json"
        with open(json_file, 'w') as f:
            json.dump(problem_data, f)
        
        with pytest.raises(InvalidProblemError, match="Invalid predecessor job index"):
            ProblemInstance.from_json(str(json_file))


class TestSolution:
    """Test solution class functionality."""
    
    def test_solution_creation(self, simple_problem_instance):
        """Test creation of valid solution."""
        operation_sequence = [
            Operation(0, 0), Operation(1, 0), Operation(2, 0),
            Operation(0, 1), Operation(1, 1), Operation(2, 1)
        ]
        machine_assignment = [0, 1, 2, 1, 0, 2]
        
        solution = Solution(operation_sequence, machine_assignment)
        
        assert len(solution.operation_sequence) == 6
        assert len(solution.machine_assignment) == 6
        assert solution.makespan == float('inf')  # Not decoded yet
        assert solution.is_valid is True
    
    def test_solution_validation_length_mismatch(self):
        """Test solution validation with mismatched lengths."""
        with pytest.raises(ValidationError, match="Length mismatch"):
            Solution([Operation(0, 0)], [0, 1])  # Different lengths
    
    def test_solution_validation_invalid_types(self):
        """Test solution validation with invalid types."""
        with pytest.raises(ValidationError, match="must be lists"):
            Solution("not a list", [0])
    
    def test_solution_validate_against_problem(self, simple_problem_instance, valid_solution):
        """Test solution validation against problem instance."""
        assert valid_solution.validate(simple_problem_instance) is True
        assert len(valid_solution.validation_errors) == 0
    
    def test_solution_validate_missing_operations(self, simple_problem_instance):
        """Test solution validation with missing operations."""
        # Create solution missing some operations
        operation_sequence = [Operation(0, 0), Operation(1, 0)]  # Missing others
        machine_assignment = [0, 1]
        
        solution = Solution(operation_sequence, machine_assignment)
        
        assert solution.validate(simple_problem_instance) is False
        assert len(solution.validation_errors) > 0
        assert "Missing operations" in solution.validation_errors[0]
    
    def test_solution_validate_invalid_machine(self, simple_problem_instance):
        """Test solution validation with invalid machine assignment."""
        operation_sequence = list(simple_problem_instance.all_operations)
        machine_assignment = [10] * len(operation_sequence)  # Invalid machines
        
        solution = Solution(operation_sequence, machine_assignment)
        
        assert solution.validate(simple_problem_instance) is False
        assert any("Invalid machine" in error for error in solution.validation_errors)
    
    def test_solution_copy(self, valid_solution):
        """Test solution copying."""
        copied = valid_solution.copy()
        
        assert copied.operation_sequence == valid_solution.operation_sequence
        assert copied.machine_assignment == valid_solution.machine_assignment
        assert copied.makespan == valid_solution.makespan
        
        # Check they are separate objects
        copied.makespan = 100.0
        assert valid_solution.makespan != 100.0
    
    def test_solution_comparison(self, simple_problem_instance):
        """Test solution comparison for sorting."""
        sol1 = Solution([Operation(0, 0)], [0])
        sol1.makespan = 10.0
        
        sol2 = Solution([Operation(0, 0)], [1])
        sol2.makespan = 5.0
        
        assert sol2 < sol1  # sol2 has better (smaller) makespan
    
    def test_solution_repr(self, valid_solution):
        """Test solution string representation."""
        valid_solution.makespan = 12.34
        repr_str = repr(valid_solution)
        
        assert "Solution" in repr_str
        assert "12.34" in repr_str
        assert str(len(valid_solution.operation_sequence)) in repr_str


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_job_single_machine(self):
        """Test problem with single job and single machine."""
        problem = ProblemInstance(
            num_jobs=1,
            num_machines=1,
            num_operations_per_job=[1],
            processing_times=[np.array([[5.0]])],
            predecessors_map={},
            successors_map={}
        )
        
        assert problem.num_jobs == 1
        assert problem.num_machines == 1
        assert problem.total_operations == 1
    
    def test_job_with_many_operations(self):
        """Test job with many operations."""
        num_operations = 10
        processing_times = [np.random.rand(num_operations, 3)]
        
        # Create chain of precedence constraints
        predecessors_map = {}
        successors_map = {}
        
        for i in range(1, num_operations):
            predecessors_map[Operation(0, i)] = {Operation(0, i-1)}
            successors_map[Operation(0, i-1)] = {Operation(0, i)}
        
        problem = ProblemInstance(
            num_jobs=1,
            num_machines=3,
            num_operations_per_job=[num_operations],
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
        
        assert problem.total_operations == num_operations
    
    def test_large_problem_instance(self):
        """Test reasonably large problem instance."""
        num_jobs = 20
        num_machines = 10
        num_operations_per_job = [5] * num_jobs
        
        processing_times = []
        for _ in range(num_jobs):
            # Some operations can't be processed on some machines
            job_times = np.random.rand(5, num_machines) * 10
            # Make 30% of assignments infeasible
            mask = np.random.rand(5, num_machines) < 0.3
            job_times[mask] = np.inf
            
            # Ensure at least one machine can process each operation
            for op_idx in range(5):
                if np.all(job_times[op_idx] == np.inf):
                    job_times[op_idx, 0] = np.random.rand() * 10
            
            processing_times.append(job_times)
        
        # Simple precedence: within each job, operations must be sequential
        predecessors_map = {}
        successors_map = {}
        
        for job_idx in range(num_jobs):
            for op_idx in range(1, 5):
                predecessors_map[Operation(job_idx, op_idx)] = {Operation(job_idx, op_idx-1)}
                successors_map[Operation(job_idx, op_idx-1)] = {Operation(job_idx, op_idx)}
        
        # Should create successfully
        problem = ProblemInstance(
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_operations_per_job=num_operations_per_job,
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
        
        assert problem.total_operations == 100
        assert len(problem.all_operations) == 100