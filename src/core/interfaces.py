"""
Core Interfaces and Base Classes for POFJSP

This module defines the fundamental interfaces and base classes used throughout
the POFJSP system to ensure consistency and enable polymorphism.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

from src.problems.problem_instance import ProblemInstance
from src.algorithms.baseline_algorithms import AlgorithmResult


@runtime_checkable
class AlgorithmInterface(Protocol):
    """Protocol defining the algorithm interface."""
    
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        ...
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        """Solve the problem and return results."""
        ...


class BaseSchedulingAlgorithm(ABC):
    """Enhanced base class for scheduling algorithms with validation."""
    
    @abstractmethod
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        """Solve the problem and return results."""
        pass
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        pass
    
    def validate_problem(self, problem: ProblemInstance) -> bool:
        """Validate that the problem can be solved by this algorithm."""
        if problem.num_jobs <= 0:
            return False
        if problem.num_machines <= 0:
            return False
        if len(problem.all_operations) == 0:
            return False
        return True
    
    def get_algorithm_info(self) -> dict:
        """Return algorithm information and parameters."""
        return {
            'name': self.algorithm_name,
            'type': self.__class__.__name__,
            'parameters': self.__dict__
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for algorithm execution."""
    execution_time: float
    memory_used_mb: float
    cpu_percent: float
    iterations_completed: int = 0
    convergence_achieved: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'execution_time': self.execution_time,
            'memory_used_mb': self.memory_used_mb,
            'cpu_percent': self.cpu_percent,
            'iterations_completed': self.iterations_completed,
            'convergence_achieved': self.convergence_achieved
        }


class EnhancedAlgorithmResult(AlgorithmResult):
    """Enhanced algorithm result with performance metrics."""
    
    def __init__(self, algorithm_name: str, makespan: float, execution_time: float,
                 solution=None, additional_metrics=None, performance_metrics=None):
        super().__init__(algorithm_name, makespan, execution_time, solution, additional_metrics)
        self.performance_metrics = performance_metrics or PerformanceMetrics(
            execution_time, 0.0, 0.0
        )
    
    def get_quality_score(self) -> float:
        """Calculate quality score based on makespan and execution time."""
        # Lower is better for both makespan and time
        if self.makespan == float('inf'):
            return 0.0
        
        # Normalize by execution time (prefer faster solutions with similar quality)
        time_penalty = max(1.0, self.execution_time / 60.0)  # Penalty after 1 minute
        return 1000.0 / (self.makespan * time_penalty)
    
    def is_feasible(self) -> bool:
        """Check if the result represents a feasible solution."""
        return (self.makespan != float('inf') and 
                self.solution is not None and
                self.execution_time > 0)


@runtime_checkable
class ConfigurableAlgorithm(Protocol):
    """Protocol for algorithms that can be configured."""
    
    def set_parameters(self, **kwargs) -> None:
        """Set algorithm parameters."""
        ...
    
    def get_parameters(self) -> dict:
        """Get current algorithm parameters."""
        ...
    
    def reset_to_defaults(self) -> None:
        """Reset algorithm to default parameters."""
        ...


@runtime_checkable
class MonitorableAlgorithm(Protocol):
    """Protocol for algorithms that support monitoring."""
    
    def set_progress_callback(self, callback) -> None:
        """Set progress callback function."""
        ...
    
    def get_progress(self) -> float:
        """Get current progress (0.0 to 1.0)."""
        ...
    
    def should_terminate(self) -> bool:
        """Check if algorithm should terminate early."""
        ...


class AlgorithmCategory:
    """Categories for algorithm classification."""
    EXACT = "exact"
    HEURISTIC = "heuristic"
    METAHEURISTIC = "metaheuristic"
    HYBRID = "hybrid"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class AlgorithmCapability:
    """Capabilities that algorithms can have."""
    PARALLEL_EXECUTION = "parallel_execution"
    ONLINE_OPTIMIZATION = "online_optimization"
    MULTI_OBJECTIVE = "multi_objective"
    CONSTRAINT_HANDLING = "constraint_handling"
    INCREMENTAL_LEARNING = "incremental_learning"