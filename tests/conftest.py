"""
Pytest configuration and fixtures for POFJSP testing.

Provides common fixtures and testing utilities to ensure
consistent test behavior across all modules.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from src.problems.problem_instance import ProblemInstance, Operation, Solution
from src.algorithms.iaoa_gns import IAOAGNSAlgorithm, IAOAConfig


@pytest.fixture
def device():
    """Provide device for testing (prefer CPU for consistency)."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def set_seed(seed):
    """Set random seeds for reproducible testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def simple_problem_instance():
    """Create a simple 3x3 problem instance for testing."""
    # 3 jobs, 3 machines, 2 operations per job
    num_jobs = 3
    num_machines = 3
    num_operations_per_job = [2, 2, 2]
    
    # Processing times: job 0: op0=[1,2,3], op1=[2,1,4]
    processing_times = [
        np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]]),  # Job 0
        np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]]),  # Job 1  
        np.array([[2.0, 3.0, 1.0], [4.0, 2.0, 1.0]])   # Job 2
    ]
    
    # Precedence constraints: within each job, op0 -> op1
    predecessors_map = {
        Operation(0, 1): {Operation(0, 0)},
        Operation(1, 1): {Operation(1, 0)},
        Operation(2, 1): {Operation(2, 0)}
    }
    
    successors_map = {
        Operation(0, 0): {Operation(0, 1)},
        Operation(1, 0): {Operation(1, 1)},
        Operation(2, 0): {Operation(2, 1)}
    }
    
    return ProblemInstance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )


@pytest.fixture
def valid_solution(simple_problem_instance):
    """Create a valid solution for the simple problem."""
    # Topologically valid sequence
    operation_sequence = [
        Operation(0, 0), Operation(1, 0), Operation(2, 0),  # First ops
        Operation(0, 1), Operation(1, 1), Operation(2, 1)   # Second ops
    ]
    
    # Machine assignments (using fastest machines)
    machine_assignment = [0, 1, 2, 1, 0, 2]
    
    return Solution(operation_sequence, machine_assignment)


@pytest.fixture
def iaoa_config():
    """Default IAOA+GNS configuration for testing."""
    return IAOAConfig(
        pop_size=10,  # Small for fast testing
        max_iterations=5  # Small for fast testing
    )


@pytest.fixture
def performance_baseline():
    """Performance baseline for regression testing."""
    return {
        'simple_problem_makespan': 6.0,  # Expected makespan for simple problem
        'iaoa_convergence_iterations': 5,  # Max iterations to converge
        'memory_usage_mb': 100  # Maximum memory usage in MB
    }


class MockPPOAgent:
    """Mock PPO agent for testing without full RL dependencies."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.trained = False
    
    def get_action(self, *args, **kwargs):
        # Return deterministic actions for testing
        return torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0])
    
    def update(self):
        self.trained = True
        return {'total_loss': 0.1, 'policy_loss': 0.05, 'value_loss': 0.03, 'entropy_loss': 0.02}


@pytest.fixture
def mock_ppo_agent():
    """Provide mock PPO agent for testing."""
    return MockPPOAgent()


def assert_solution_valid(solution: Solution, problem: ProblemInstance) -> None:
    """Assert that a solution is valid for the given problem."""
    # Check basic structure
    assert len(solution.operation_sequence) == problem.total_operations
    assert len(solution.machine_assignment) == problem.total_operations
    
    # Check all operations are present
    expected_ops = set(problem.all_operations)
    actual_ops = set(solution.operation_sequence)
    assert expected_ops == actual_ops, f"Missing operations: {expected_ops - actual_ops}"
    
    # Check machine assignments are valid
    for i, (op, machine) in enumerate(zip(solution.operation_sequence, solution.machine_assignment)):
        assert 0 <= machine < problem.num_machines, f"Invalid machine {machine} at position {i}"
        
        # Check operation can be processed on assigned machine
        proc_time = problem.processing_times[op.job_idx][op.op_idx_in_job, machine]
        assert proc_time != np.inf, f"Operation {op} cannot be processed on machine {machine}"


def assert_makespan_reasonable(solution: Solution, expected_range: tuple = (1.0, 1000.0)) -> None:
    """Assert that solution makespan is within reasonable bounds."""
    assert expected_range[0] <= solution.makespan <= expected_range[1], \
        f"Makespan {solution.makespan} outside expected range {expected_range}"
    assert solution.makespan != float('inf'), "Solution has infinite makespan"


def assert_no_performance_regression(current_value: float, baseline_value: float, tolerance: float = 0.1) -> None:
    """Assert that performance hasn't regressed beyond tolerance."""
    regression = (current_value - baseline_value) / baseline_value
    assert regression <= tolerance, \
        f"Performance regression: {regression:.2%} > {tolerance:.2%} tolerance"


# Regression testing markers
pytest.mark.slow = pytest.mark.filterwarnings("ignore::UserWarning")
pytest.mark.integration = pytest.mark.filterwarnings("ignore::DeprecationWarning")
pytest.mark.performance = pytest.mark.filterwarnings("ignore")