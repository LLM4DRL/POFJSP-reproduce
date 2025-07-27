"""
POFJSP - Partially Ordered Flexible Job Shop Scheduling

This package provides tools and algorithms for solving POFJSP problems.
"""

"""
POFJSP - Partially Ordered Flexible Job Shop Scheduling

This package provides tools and algorithms for solving POFJSP problems.
"""

__version__ = "1.0.0"
__author__ = "POFJSP Research Team"

# Lazy imports to avoid circular dependencies
def get_algorithm_factory():
    """Get the algorithm factory."""
    from src.algorithms.factory import AlgorithmFactory
    return AlgorithmFactory

def get_problem_instance():
    """Get the ProblemInstance class."""
    from src.problems.problem_instance import ProblemInstance, Solution
    return ProblemInstance, Solution

def get_rl_components():
    """Get RL components."""
    from src.rl.models.ppo_agent import PPOAgent
    from src.rl.environments.pofjsp_env import POFJSPEnv
    return PPOAgent, POFJSPEnv

__all__ = [
    'get_algorithm_factory',
    'get_problem_instance', 
    'get_rl_components'
]