"""
POFJSP Exception Hierarchy

Custom exceptions for Partial Order Flexible Job Shop Problem operations.
Provides structured error handling across all modules.
"""


class POFJSPError(Exception):
    """Base exception for all POFJSP-related errors."""
    pass


class ValidationError(POFJSPError):
    """Raised when input validation fails."""
    pass


class InvalidProblemError(ValidationError):
    """Raised when problem instance is invalid or malformed."""
    
    def __init__(self, problem_type: str, details: str):
        self.problem_type = problem_type
        self.details = details
        super().__init__(f"Invalid {problem_type} problem: {details}")


class InvalidMachineAssignmentError(POFJSPError):
    """Raised when an operation is assigned to an incompatible machine."""
    
    def __init__(self, operation, machine_id: int, processing_time: float):
        self.operation = operation
        self.machine_id = machine_id
        self.processing_time = processing_time
        super().__init__(
            f"Operation {operation} cannot be processed on machine {machine_id} "
            f"(processing time: {processing_time})"
        )


class PrecedenceConstraintViolationError(POFJSPError):
    """Raised when precedence constraints are violated."""
    
    def __init__(self, operation, violated_predecessors: list):
        self.operation = operation
        self.violated_predecessors = violated_predecessors
        super().__init__(
            f"Operation {operation} scheduled before required predecessors: {violated_predecessors}"
        )


class InfeasibleProblemError(POFJSPError):
    """Raised when no feasible solution exists for the problem."""
    
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Problem is infeasible: {reason}")


class AlgorithmError(POFJSPError):
    """Base class for algorithm-specific errors."""
    pass


class ConvergenceError(AlgorithmError):
    """Raised when algorithm fails to converge."""
    
    def __init__(self, algorithm_name: str, iterations: int, best_fitness: float):
        self.algorithm_name = algorithm_name
        self.iterations = iterations
        self.best_fitness = best_fitness
        super().__init__(
            f"{algorithm_name} failed to converge after {iterations} iterations "
            f"(best fitness: {best_fitness})"
        )


class RLTrainingError(AlgorithmError):
    """Raised during RL training failures."""
    
    def __init__(self, stage: str, details: str):
        self.stage = stage
        self.details = details
        super().__init__(f"RL training error in {stage}: {details}")


class MemoryError(POFJSPError):
    """Raised when memory allocation fails."""
    
    def __init__(self, operation: str, memory_used: str):
        self.operation = operation
        self.memory_used = memory_used
        super().__init__(f"Memory error during {operation}. Memory used: {memory_used}")


class ConfigurationError(POFJSPError):
    """Raised when configuration is invalid."""
    
    def __init__(self, parameter: str, value, expected: str):
        self.parameter = parameter
        self.value = value
        self.expected = expected
        super().__init__(f"Invalid {parameter}={value}, expected {expected}")


def handle_gpu_memory_error(func):
    """Decorator to handle GPU memory errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise MemoryError(
                    operation=func.__name__,
                    memory_used=f"GPU memory exhausted: {e}"
                )
            else:
                raise e
    return wrapper


def validate_positive_int(value: int, name: str) -> int:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value}")
    return value


def validate_probability(value: float, name: str) -> float:
    """Validate that a value is a valid probability."""
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValidationError(f"{name} must be between 0 and 1, got {value}")
    return float(value)


def validate_tensor_shape(tensor, expected_shape: tuple, name: str):
    """Validate tensor shape."""
    if tensor.shape != expected_shape:
        raise ValidationError(
            f"{name} has shape {tensor.shape}, expected {expected_shape}"
        )