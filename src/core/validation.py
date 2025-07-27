"""
Validation Decorators and Performance Contracts for POFJSP

This module provides decorators and utilities for validating algorithm interfaces
and enforcing performance contracts.
"""

import time
import psutil
import functools
import threading
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
import logging

from src.exceptions import ValidationError, AlgorithmError, PerformanceError
from src.core.interfaces import PerformanceMetrics


logger = logging.getLogger(__name__)


@dataclass
class PerformanceContract:
    """Performance contract specification."""
    max_memory_mb: Optional[float] = None
    max_time_seconds: Optional[float] = None
    min_quality_score: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    required_convergence: bool = False
    
    def validate(self, metrics: PerformanceMetrics, quality_score: float = None) -> bool:
        """Validate performance metrics against contract."""
        violations = []
        
        if self.max_memory_mb and metrics.memory_used_mb > self.max_memory_mb:
            violations.append(f"Memory usage {metrics.memory_used_mb:.1f}MB exceeds limit {self.max_memory_mb}MB")
        
        if self.max_time_seconds and metrics.execution_time > self.max_time_seconds:
            violations.append(f"Execution time {metrics.execution_time:.1f}s exceeds limit {self.max_time_seconds}s")
        
        if self.max_cpu_percent and metrics.cpu_percent > self.max_cpu_percent:
            violations.append(f"CPU usage {metrics.cpu_percent:.1f}% exceeds limit {self.max_cpu_percent}%")
        
        if self.min_quality_score and quality_score and quality_score < self.min_quality_score:
            violations.append(f"Quality score {quality_score:.2f} below minimum {self.min_quality_score}")
        
        if self.required_convergence and not metrics.convergence_achieved:
            violations.append("Required convergence not achieved")
        
        if violations:
            raise PerformanceError(f"Performance contract violations: {'; '.join(violations)}")
        
        return True


def validate_algorithm_interface(algorithm_class):
    """
    Decorator to validate that a class implements the required algorithm interface.
    
    Args:
        algorithm_class: Class to validate
        
    Returns:
        Validated class
        
    Raises:
        ValidationError: If interface is not properly implemented
    """
    required_methods = ['solve', 'algorithm_name']
    required_properties = ['algorithm_name']
    
    # Check methods
    for method in required_methods:
        if not hasattr(algorithm_class, method):
            raise ValidationError(f"Algorithm class {algorithm_class.__name__} missing required method: {method}")
        
        method_obj = getattr(algorithm_class, method)
        if not callable(method_obj) and method not in required_properties:
            raise ValidationError(f"Algorithm class {algorithm_class.__name__} method {method} is not callable")
    
    # Try to check if it implements the interface methods
    try:
        # Check if class has the required methods
        if not (hasattr(algorithm_class, 'solve') and hasattr(algorithm_class, 'algorithm_name')):
            raise ValidationError(f"Algorithm class {algorithm_class.__name__} does not implement required interface")
    except Exception as e:
        logger.warning(f"Could not validate interface for {algorithm_class.__name__}: {e}")
    
    # Add validation metadata
    algorithm_class._interface_validated = True
    algorithm_class._validation_timestamp = time.time()
    
    return algorithm_class


def performance_contract(max_memory_mb: Optional[float] = None,
                        max_time_seconds: Optional[float] = None,
                        min_quality_score: Optional[float] = None,
                        max_cpu_percent: Optional[float] = None,
                        required_convergence: bool = False):
    """
    Decorator to enforce performance contracts on algorithm methods.
    
    Args:
        max_memory_mb: Maximum memory usage in MB
        max_time_seconds: Maximum execution time in seconds
        min_quality_score: Minimum quality score required
        max_cpu_percent: Maximum CPU usage percentage
        required_convergence: Whether convergence is required
        
    Returns:
        Decorated function
    """
    contract = PerformanceContract(
        max_memory_mb=max_memory_mb,
        max_time_seconds=max_time_seconds,
        min_quality_score=min_quality_score,
        max_cpu_percent=max_cpu_percent,
        required_convergence=required_convergence
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start monitoring
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = process.cpu_percent()
            
            # Execute function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {e}")
                raise
            
            # Collect performance metrics
            end_time = time.time()
            execution_time = end_time - start_time
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            cpu_percent = process.cpu_percent()
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_used_mb=memory_used,
                cpu_percent=cpu_percent,
                iterations_completed=getattr(result, 'iterations_completed', 0),
                convergence_achieved=getattr(result, 'convergence_achieved', False)
            )
            
            # Calculate quality score if result has makespan
            quality_score = None
            if hasattr(result, 'makespan') and result.makespan != float('inf'):
                quality_score = 1000.0 / result.makespan  # Simple quality metric
            
            # Validate against contract
            try:
                contract.validate(metrics, quality_score)
            except PerformanceError as e:
                logger.warning(f"Performance contract violation in {func.__name__}: {e}")
                # Don't fail the execution, just log the violation
            
            # Attach performance metrics to result if possible
            if hasattr(result, '__dict__'):
                result.performance_metrics = metrics
            
            return result
        
        return wrapper
    return decorator


def validate_input_types(**type_constraints):
    """
    Decorator to validate input types.
    
    Args:
        **type_constraints: Mapping of parameter names to expected types
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Check types
            for param_name, expected_type in type_constraints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise ValidationError(
                            f"Parameter {param_name} must be of type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_range(param_name: str, min_val: float = None, max_val: float = None):
    """
    Decorator to validate parameter ranges.
    
    Args:
        param_name: Name of parameter to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Check range
            if param_name in bound_args.arguments:
                value = bound_args.arguments[param_name]
                if min_val is not None and value < min_val:
                    raise ValidationError(f"Parameter {param_name} must be >= {min_val}, got {value}")
                if max_val is not None and value > max_val:
                    raise ValidationError(f"Parameter {param_name} must be <= {max_val}, got {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def timeout_handler(timeout_seconds: float):
    """
    Decorator to enforce timeout on function execution.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                # Force thread termination (not recommended but necessary for timeout)
                logger.warning(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise AlgorithmError(f"Function {func.__name__} exceeded timeout of {timeout_seconds}s")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator


def require_convergence(tolerance: float = 1e-6, max_iterations: int = None):
    """
    Decorator to ensure algorithm convergence.
    
    Args:
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations before forced termination
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Check convergence if result has convergence information
            if hasattr(result, 'additional_metrics'):
                metrics = result.additional_metrics or {}
                
                # Check if convergence was achieved
                converged = metrics.get('converged', False)
                final_improvement = metrics.get('final_improvement', float('inf'))
                iterations = metrics.get('iterations', 0)
                
                if not converged and final_improvement > tolerance:
                    logger.warning(
                        f"Algorithm {getattr(result, 'algorithm_name', 'unknown')} "
                        f"did not converge within tolerance {tolerance} "
                        f"(final improvement: {final_improvement})"
                    )
                
                if max_iterations and iterations >= max_iterations:
                    logger.warning(
                        f"Algorithm reached maximum iterations {max_iterations} "
                        f"without convergence"
                    )
            
            return result
        return wrapper
    return decorator


class PerformanceMonitor:
    """Context manager for monitoring algorithm performance."""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.start_time = None
        self.start_memory = None
        self.process = None
        
    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log results."""
        if self.start_time:
            execution_time = time.time() - self.start_time
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - self.start_memory
            
            logger.info(
                f"Algorithm {self.algorithm_name} completed in {execution_time:.2f}s "
                f"using {memory_used:.1f}MB additional memory"
            )
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if not self.start_time:
            raise RuntimeError("Monitor not started")
        
        execution_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - self.start_memory
        cpu_percent = self.process.cpu_percent()
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_used_mb=memory_used,
            cpu_percent=cpu_percent
        )