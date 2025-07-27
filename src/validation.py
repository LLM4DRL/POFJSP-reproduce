"""
Comprehensive Validation System

Provides decorators and validators to prevent runtime errors
throughout the POFJSP codebase while maintaining performance.
"""

import functools
import inspect
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import logging

from src.exceptions import ValidationError, InvalidProblemError

logger = logging.getLogger(__name__)


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Mapping of parameter names to validator functions
        
    Example:
        @validate_inputs(
            num_jobs=lambda x: x > 0,
            processing_times=lambda x: isinstance(x, list)
        )
        def create_problem(num_jobs, processing_times):
            pass
    """
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to parameter names
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each specified parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            raise ValidationError(f"Validation failed for parameter '{param_name}' with value {value}")
                    except Exception as e:
                        raise ValidationError(f"Validation error for parameter '{param_name}': {e}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_tensor_inputs(**validators):
    """
    Decorator specifically for tensor input validation.
    
    Example:
        @validate_tensor_inputs(
            x=lambda t: t.dim() == 2,
            edge_index=lambda t: t.dtype == torch.long
        )
        def forward(x, edge_index):
            pass
    """
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if torch.is_tensor(value):
                        try:
                            if not validator(value):
                                raise ValidationError(f"Tensor validation failed for '{param_name}'")
                        except Exception as e:
                            raise ValidationError(f"Tensor validation error for '{param_name}': {e}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Common validator functions
class Validators:
    """Collection of common validation functions."""
    
    @staticmethod
    def positive_int(value: Any) -> bool:
        """Validate positive integer."""
        return isinstance(value, int) and value > 0
    
    @staticmethod
    def non_negative_int(value: Any) -> bool:
        """Validate non-negative integer."""
        return isinstance(value, int) and value >= 0
    
    @staticmethod
    def positive_float(value: Any) -> bool:
        """Validate positive float."""
        return isinstance(value, (int, float)) and value > 0
    
    @staticmethod
    def probability(value: Any) -> bool:
        """Validate probability (0 <= x <= 1)."""
        return isinstance(value, (int, float)) and 0 <= value <= 1
    
    @staticmethod
    def non_empty_list(value: Any) -> bool:
        """Validate non-empty list."""
        return isinstance(value, list) and len(value) > 0
    
    @staticmethod
    def numpy_array_2d(value: Any) -> bool:
        """Validate 2D numpy array."""
        return isinstance(value, np.ndarray) and value.ndim == 2
    
    @staticmethod
    def finite_values(value: Any) -> bool:
        """Validate that array contains finite values (except allowed inf)."""
        if isinstance(value, np.ndarray):
            finite_values = value[np.isfinite(value)]
            return np.all(finite_values >= 0)  # Non-negative finite values
        return True
    
    @staticmethod
    def tensor_2d(value: torch.Tensor) -> bool:
        """Validate 2D tensor."""
        return value.dim() == 2
    
    @staticmethod
    def tensor_1d(value: torch.Tensor) -> bool:
        """Validate 1D tensor."""
        return value.dim() == 1
    
    @staticmethod
    def tensor_long_type(value: torch.Tensor) -> bool:
        """Validate tensor has long dtype."""
        return value.dtype == torch.long
    
    @staticmethod
    def tensor_float_type(value: torch.Tensor) -> bool:
        """Validate tensor has float dtype."""
        return value.dtype in [torch.float32, torch.float64]
    
    @staticmethod
    def tensor_non_negative(value: torch.Tensor) -> bool:
        """Validate tensor has non-negative values."""
        return torch.all(value >= 0)


def validate_problem_instance_inputs(func: Callable) -> Callable:
    """Specialized validator for problem instance creation."""
    @functools.wraps(func)
    def wrapper(self, num_jobs: int, num_machines: int, num_operations_per_job: List[int], 
                processing_times: List[np.ndarray], predecessors_map: Dict, successors_map: Dict):
        
        # Validate basic parameters
        if not isinstance(num_jobs, int) or num_jobs <= 0:
            raise ValidationError(f"num_jobs must be positive integer, got {num_jobs}")
        
        if not isinstance(num_machines, int) or num_machines <= 0:
            raise ValidationError(f"num_machines must be positive integer, got {num_machines}")
        
        # Validate operations per job
        if not isinstance(num_operations_per_job, list):
            raise ValidationError("num_operations_per_job must be a list")
        
        if len(num_operations_per_job) != num_jobs:
            raise ValidationError(f"num_operations_per_job length {len(num_operations_per_job)} != num_jobs {num_jobs}")
        
        for i, ops in enumerate(num_operations_per_job):
            if not isinstance(ops, int) or ops <= 0:
                raise ValidationError(f"num_operations_per_job[{i}] must be positive integer, got {ops}")
        
        # Validate processing times
        if not isinstance(processing_times, list):
            raise ValidationError("processing_times must be a list")
        
        if len(processing_times) != num_jobs:
            raise ValidationError(f"processing_times length {len(processing_times)} != num_jobs {num_jobs}")
        
        for job_idx, job_times in enumerate(processing_times):
            if not isinstance(job_times, np.ndarray):
                raise ValidationError(f"processing_times[{job_idx}] must be numpy array")
            
            expected_shape = (num_operations_per_job[job_idx], num_machines)
            if job_times.shape != expected_shape:
                raise ValidationError(
                    f"processing_times[{job_idx}] shape {job_times.shape} != expected {expected_shape}"
                )
            
            # Check for negative finite values
            finite_mask = np.isfinite(job_times)
            finite_values = job_times[finite_mask]
            if np.any(finite_values < 0):
                raise ValidationError(f"processing_times[{job_idx}] contains negative values")
        
        # Validate precedence maps
        if not isinstance(predecessors_map, dict):
            raise ValidationError("predecessors_map must be a dictionary")
        
        if not isinstance(successors_map, dict):
            raise ValidationError("successors_map must be a dictionary")
        
        return func(self, num_jobs, num_machines, num_operations_per_job, 
                    processing_times, predecessors_map, successors_map)
    
    return wrapper


def validate_solution_inputs(func: Callable) -> Callable:
    """Specialized validator for solution creation."""
    @functools.wraps(func)
    def wrapper(self, operation_sequence: List, machine_assignment: List[int]):
        
        if not isinstance(operation_sequence, list):
            raise ValidationError("operation_sequence must be a list")
        
        if not isinstance(machine_assignment, list):
            raise ValidationError("machine_assignment must be a list")
        
        if len(operation_sequence) != len(machine_assignment):
            raise ValidationError(
                f"Length mismatch: operation_sequence={len(operation_sequence)}, "
                f"machine_assignment={len(machine_assignment)}"
            )
        
        # Validate machine assignments are integers
        for i, machine in enumerate(machine_assignment):
            if not isinstance(machine, int):
                raise ValidationError(f"machine_assignment[{i}] must be integer, got {type(machine)}")
        
        return func(self, operation_sequence, machine_assignment)
    
    return wrapper


def validate_ppo_inputs(func: Callable) -> Callable:
    """Specialized validator for PPO agent methods."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get method name for specific validation
        method_name = func.__name__
        
        if method_name == 'get_action':
            # Validate get_action inputs
            if len(args) >= 6:  # self, x, edge_index, batch, job_masks, machine_masks
                x, edge_index, batch, job_masks, machine_masks = args[1:6]
                
                if not torch.is_tensor(x):
                    raise ValidationError("x must be a tensor")
                if not torch.is_tensor(edge_index):
                    raise ValidationError("edge_index must be a tensor")
                if not torch.is_tensor(batch):
                    raise ValidationError("batch must be a tensor")
                if not torch.is_tensor(job_masks):
                    raise ValidationError("job_masks must be a tensor")
                if not torch.is_tensor(machine_masks):
                    raise ValidationError("machine_masks must be a tensor")
                
                # Check tensor properties
                if edge_index.dtype != torch.long:
                    raise ValidationError("edge_index must have long dtype")
                if batch.dtype != torch.long:
                    raise ValidationError("batch must have long dtype")
                if job_masks.dtype != torch.bool:
                    raise ValidationError("job_masks must have bool dtype")
                if machine_masks.dtype != torch.bool:
                    raise ValidationError("machine_masks must have bool dtype")
                
                # Check dimensions
                if x.dim() != 2:
                    raise ValidationError(f"x must be 2D tensor, got {x.dim()}D")
                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    raise ValidationError(f"edge_index must be 2xN tensor, got {edge_index.shape}")
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_algorithm_inputs(func: Callable) -> Callable:
    """Specialized validator for algorithm methods."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__
        
        if method_name == 'solve':
            # Validate solve method inputs
            if len(args) >= 2:  # self, problem
                problem = args[1]
                
                if problem is None:
                    raise ValidationError("problem cannot be None")
                
                # Check if it has required attributes
                required_attrs = ['num_jobs', 'num_machines', 'all_operations', 'processing_times']
                for attr in required_attrs:
                    if not hasattr(problem, attr):
                        raise ValidationError(f"problem must have attribute '{attr}'")
                
                # Check basic problem properties
                if problem.num_jobs <= 0:
                    raise ValidationError(f"problem.num_jobs must be positive, got {problem.num_jobs}")
                if problem.num_machines <= 0:
                    raise ValidationError(f"problem.num_machines must be positive, got {problem.num_machines}")
        
        return func(*args, **kwargs)
    
    return wrapper


class SafeOperationWrapper:
    """Wrapper for safe execution of operations with automatic error handling."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError:
                # Re-raise validation errors as-is
                raise
            except Exception as e:
                # Wrap other exceptions with context
                logger.error(f"Error in {self.operation_name}: {e}")
                raise ValidationError(f"Operation '{self.operation_name}' failed: {e}")
        
        return wrapper


# Pre-configured decorators for common use cases
validate_iaoa_inputs = validate_inputs(
    pop_size=Validators.positive_int,
    max_iterations=Validators.positive_int
)

validate_tensor_forward = validate_tensor_inputs(
    x=lambda t: t.dim() >= 2,
    edge_index=lambda t: t.dtype == torch.long and t.dim() == 2
)

validate_rl_episode = validate_inputs(
    max_episode_steps=Validators.positive_int,
    reward_type=lambda x: isinstance(x, str) and x in ['makespan', 'utilization', 'combined']
)


def validate_numeric_stability(value: Union[float, torch.Tensor, np.ndarray], 
                              name: str, 
                              max_value: float = 1e6) -> None:
    """Validate numeric stability of values."""
    if isinstance(value, torch.Tensor):
        if torch.any(torch.isnan(value)):
            raise ValidationError(f"{name} contains NaN values")
        if torch.any(torch.isinf(value)):
            # Allow positive infinity for processing times
            if not (name == "processing_times" and torch.all(torch.isinf(value) == (value == float('inf')))):
                raise ValidationError(f"{name} contains invalid infinity values")
        if torch.any(torch.abs(value[torch.isfinite(value)]) > max_value):
            raise ValidationError(f"{name} contains values too large for numeric stability")
    
    elif isinstance(value, np.ndarray):
        if np.any(np.isnan(value)):
            raise ValidationError(f"{name} contains NaN values")
        finite_mask = np.isfinite(value)
        if np.any(np.abs(value[finite_mask]) > max_value):
            raise ValidationError(f"{name} contains values too large for numeric stability")
    
    elif isinstance(value, (int, float)):
        if np.isnan(value):
            raise ValidationError(f"{name} is NaN")
        if np.isinf(value) and name != "processing_time":
            raise ValidationError(f"{name} is infinite")
        if abs(value) > max_value:
            raise ValidationError(f"{name} value {value} too large for numeric stability")


def create_type_validator(expected_type: type, allow_none: bool = False):
    """Create a type validator function."""
    def validator(value: Any) -> bool:
        if allow_none and value is None:
            return True
        return isinstance(value, expected_type)
    return validator


# Example usage and testing
if __name__ == "__main__":
    # Test validator functions
    assert Validators.positive_int(5) is True
    assert Validators.positive_int(0) is False
    assert Validators.positive_int(-1) is False
    
    assert Validators.probability(0.5) is True
    assert Validators.probability(0.0) is True
    assert Validators.probability(1.0) is True
    assert Validators.probability(1.5) is False
    
    print("All validator tests passed!")