"""
Algorithm Factory for POFJSP

This module provides a factory pattern for creating scheduling algorithms
with consistent interfaces and configuration management.
"""

from typing import Dict, Type, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.algorithms.baseline_algorithms import (
    BaseSchedulingAlgorithm, GeneticAlgorithm, SimulatedAnnealing, 
    RandomAlgorithm, GreedyAlgorithm, DispatchingRulesAlgorithm,
    AntColonyOptimization, ParticleSwarmOptimization
)
from src.algorithms.metaheuristics import (
    DifferentialEvolution, VariableNeighborhoodSearch, TabuSearch,
    HybridGeneticLocalSearch, MemorybasedSimulatedAnnealing
)
from src.algorithms.iaoa_gns import IAOAGNSAlgorithm
from src.exceptions import AlgorithmError, ValidationError


@dataclass
class AlgorithmConfig:
    """Configuration for algorithms."""
    algorithm_type: str
    parameters: Dict[str, Any]
    timeout: float = 300.0
    
    def validate(self):
        """Validate configuration parameters."""
        if self.timeout <= 0:
            raise ValidationError("Timeout must be positive")
        
        required_params = ALGORITHM_REQUIREMENTS.get(self.algorithm_type, {})
        for param, param_type in required_params.items():
            if param not in self.parameters:
                raise ValidationError(f"Missing required parameter '{param}' for {self.algorithm_type}")
            if not isinstance(self.parameters[param], param_type):
                raise ValidationError(f"Parameter '{param}' must be of type {param_type.__name__}")


# Algorithm parameter requirements
ALGORITHM_REQUIREMENTS = {
    'genetic': {
        'pop_size': int,
        'generations': int,
        'crossover_rate': float,
        'mutation_rate': float
    },
    'simulated_annealing': {
        'initial_temp': float,
        'cooling_rate': float,
        'min_temp': float,
        'max_iterations': int
    },
    'aco': {
        'n_ants': int,
        'max_iterations': int,
        'alpha': float,
        'beta': float,
        'rho': float
    },
    'pso': {
        'n_particles': int,
        'max_iterations': int,
        'w': float,
        'c1': float,
        'c2': float
    },
    'iaoa_gns': {
        'pop_size': int,
        'max_iterations': int
    }
}

# Default parameter sets
DEFAULT_PARAMETERS = {
    'genetic': {
        'pop_size': 50,
        'generations': 100,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2
    },
    'simulated_annealing': {
        'initial_temp': 1000.0,
        'cooling_rate': 0.95,
        'min_temp': 1.0,
        'max_iterations': 10000
    },
    'aco': {
        'n_ants': 20,
        'max_iterations': 50,
        'alpha': 1.0,
        'beta': 2.0,
        'rho': 0.1,
        'q0': 0.9
    },
    'pso': {
        'n_particles': 30,
        'max_iterations': 50,
        'w': 0.7,
        'c1': 2.0,
        'c2': 2.0
    },
    'differential_evolution': {
        'population_size': 30,
        'max_iterations': 50,
        'F': 0.8,
        'CR': 0.9
    },
    'vns': {
        'max_iterations': 100,
        'k_max': 3
    },
    'tabu_search': {
        'max_iterations': 100,
        'tabu_tenure': 7
    },
    'hybrid_ga_ls': {
        'population_size': 50,
        'max_generations': 100,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'local_search_rate': 0.3
    },
    'memory_sa': {
        'initial_temp': 1000.0,
        'cooling_rate': 0.95,
        'min_temp': 1.0,
        'max_iterations': 10000,
        'memory_size': 20
    },
    'iaoa_gns': {
        'pop_size': 80,
        'max_iterations': 60
    },
    'random': {
        'num_trials': 1000
    },
    'dispatching': {
        'rule': 'SPT'
    }
}


class AlgorithmFactory:
    """Factory for creating scheduling algorithms with consistent interface."""
    
    _algorithms: Dict[str, Type[BaseSchedulingAlgorithm]] = {
        'genetic': GeneticAlgorithm,
        'simulated_annealing': SimulatedAnnealing,
        'aco': AntColonyOptimization,
        'pso': ParticleSwarmOptimization,
        'differential_evolution': DifferentialEvolution,
        'vns': VariableNeighborhoodSearch,
        'tabu_search': TabuSearch,
        'hybrid_ga_ls': HybridGeneticLocalSearch,
        'memory_sa': MemorybasedSimulatedAnnealing,
        'iaoa_gns': IAOAGNSAlgorithm,
        'random': RandomAlgorithm,
        'greedy': GreedyAlgorithm,
        'dispatching': DispatchingRulesAlgorithm
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_type: str, 
                        parameters: Optional[Dict[str, Any]] = None,
                        **kwargs) -> BaseSchedulingAlgorithm:
        """
        Create an algorithm instance.
        
        Args:
            algorithm_type: Type of algorithm to create
            parameters: Algorithm-specific parameters
            **kwargs: Additional parameters
            
        Returns:
            Algorithm instance
            
        Raises:
            AlgorithmError: If algorithm type is unknown
            ValidationError: If parameters are invalid
        """
        if algorithm_type not in cls._algorithms:
            available = ', '.join(cls._algorithms.keys())
            raise AlgorithmError(f"Unknown algorithm type: {algorithm_type}. Available: {available}")
        
        # Merge parameters
        final_params = DEFAULT_PARAMETERS.get(algorithm_type, {}).copy()
        if parameters:
            final_params.update(parameters)
        final_params.update(kwargs)
        
        # Validate parameters
        config = AlgorithmConfig(algorithm_type, final_params)
        config.validate()
        
        # Create algorithm instance
        algorithm_class = cls._algorithms[algorithm_type]
        
        try:
            # Special case for IAOA+GNS which expects a config object
            if algorithm_type == 'iaoa_gns':
                from src.algorithms.iaoa_gns import IAOAConfig
                config = IAOAConfig(**final_params)
                return algorithm_class(config)
            else:
                return algorithm_class(**final_params)
        except TypeError as e:
            raise AlgorithmError(f"Invalid parameters for {algorithm_type}: {e}")
    
    @classmethod
    def get_available_algorithms(cls) -> Dict[str, str]:
        """Get list of available algorithms with descriptions."""
        descriptions = {
            'genetic': 'Genetic Algorithm - Evolutionary approach with crossover and mutation',
            'simulated_annealing': 'Simulated Annealing - Physics-inspired optimization',
            'aco': 'Ant Colony Optimization - Swarm intelligence based on ant behavior',
            'pso': 'Particle Swarm Optimization - Population-based optimization',
            'differential_evolution': 'Differential Evolution - Real-parameter evolutionary algorithm',
            'vns': 'Variable Neighborhood Search - Local search with systematic neighborhood changes',
            'tabu_search': 'Tabu Search - Memory-based local search with tabu restrictions',
            'hybrid_ga_ls': 'Hybrid Genetic Algorithm with Local Search - GA enhanced with local improvement',
            'memory_sa': 'Memory-based Simulated Annealing - SA with elite solution memory',
            'iaoa_gns': 'IAOA+GNS - Improved Adaptive Optimization Algorithm with Grade Neighborhood Search',
            'random': 'Random Search - Baseline random sampling approach',
            'greedy': 'Greedy Algorithm - Constructive heuristic with earliest completion time',
            'dispatching': 'Dispatching Rules - Simple priority-based scheduling rules'
        }
        return descriptions
    
    @classmethod
    def get_algorithm_parameters(cls, algorithm_type: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm."""
        if algorithm_type not in cls._algorithms:
            raise AlgorithmError(f"Unknown algorithm type: {algorithm_type}")
        return DEFAULT_PARAMETERS.get(algorithm_type, {}).copy()
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type[BaseSchedulingAlgorithm], 
                          default_params: Optional[Dict[str, Any]] = None):
        """
        Register a new algorithm type.
        
        Args:
            name: Algorithm name
            algorithm_class: Algorithm class
            default_params: Default parameters
        """
        if not issubclass(algorithm_class, BaseSchedulingAlgorithm):
            raise AlgorithmError("Algorithm class must extend BaseSchedulingAlgorithm")
        
        cls._algorithms[name] = algorithm_class
        if default_params:
            DEFAULT_PARAMETERS[name] = default_params


class AlgorithmBuilder:
    """Builder pattern for creating configured algorithms."""
    
    def __init__(self, algorithm_type: str):
        self.algorithm_type = algorithm_type
        self.parameters = {}
        self.timeout = 300.0
    
    def with_parameter(self, name: str, value: Any) -> 'AlgorithmBuilder':
        """Add a parameter."""
        self.parameters[name] = value
        return self
    
    def with_parameters(self, **kwargs) -> 'AlgorithmBuilder':
        """Add multiple parameters."""
        self.parameters.update(kwargs)
        return self
    
    def with_timeout(self, timeout: float) -> 'AlgorithmBuilder':
        """Set timeout."""
        self.timeout = timeout
        return self
    
    def build(self) -> BaseSchedulingAlgorithm:
        """Build the algorithm."""
        return AlgorithmFactory.create_algorithm(self.algorithm_type, self.parameters)


def create_algorithm_suite() -> Dict[str, BaseSchedulingAlgorithm]:
    """Create a suite of algorithms with default configurations for comparison."""
    suite = {}
    
    # Fast algorithms for quick comparison
    suite['spt'] = AlgorithmFactory.create_algorithm('dispatching', {'rule': 'SPT'})
    suite['lpt'] = AlgorithmFactory.create_algorithm('dispatching', {'rule': 'LPT'})
    suite['greedy'] = AlgorithmFactory.create_algorithm('greedy')
    
    # Metaheuristics with reasonable parameters
    suite['ga_quick'] = AlgorithmFactory.create_algorithm('genetic', {
        'pop_size': 30, 'generations': 50
    })
    suite['sa_quick'] = AlgorithmFactory.create_algorithm('simulated_annealing', {
        'initial_temp': 500.0, 'max_iterations': 5000
    })
    suite['aco_quick'] = AlgorithmFactory.create_algorithm('aco', {
        'n_ants': 15, 'max_iterations': 30
    })
    suite['pso_quick'] = AlgorithmFactory.create_algorithm('pso', {
        'n_particles': 20, 'max_iterations': 30
    })
    
    # High-quality algorithms for final comparison
    suite['iaoa_gns'] = AlgorithmFactory.create_algorithm('iaoa_gns')
    suite['hybrid_ga'] = AlgorithmFactory.create_algorithm('hybrid_ga_ls')
    suite['tabu_search'] = AlgorithmFactory.create_algorithm('tabu_search')
    
    return suite


def create_benchmark_suite() -> Dict[str, BaseSchedulingAlgorithm]:
    """Create comprehensive benchmark suite with all algorithms."""
    suite = {}
    
    for algo_type in AlgorithmFactory.get_available_algorithms():
        try:
            suite[algo_type] = AlgorithmFactory.create_algorithm(algo_type)
        except Exception as e:
            print(f"Warning: Could not create {algo_type}: {e}")
    
    return suite


# Convenience functions for common algorithm configurations
def create_genetic_algorithm(pop_size: int = 50, generations: int = 100) -> GeneticAlgorithm:
    """Create genetic algorithm with specified parameters."""
    return AlgorithmFactory.create_algorithm('genetic', {
        'pop_size': pop_size, 'generations': generations
    })


def create_simulated_annealing(initial_temp: float = 1000.0, 
                             cooling_rate: float = 0.95) -> SimulatedAnnealing:
    """Create simulated annealing with specified parameters."""
    return AlgorithmFactory.create_algorithm('simulated_annealing', {
        'initial_temp': initial_temp, 'cooling_rate': cooling_rate
    })


def create_ant_colony(n_ants: int = 20, max_iterations: int = 50) -> AntColonyOptimization:
    """Create ACO algorithm with specified parameters."""
    return AlgorithmFactory.create_algorithm('aco', {
        'n_ants': n_ants, 'max_iterations': max_iterations
    })


def create_particle_swarm(n_particles: int = 30, max_iterations: int = 50) -> ParticleSwarmOptimization:
    """Create PSO algorithm with specified parameters."""
    return AlgorithmFactory.create_algorithm('pso', {
        'n_particles': n_particles, 'max_iterations': max_iterations
    })


def create_iaoa_gns(pop_size: int = 80, max_iterations: int = 60) -> IAOAGNSAlgorithm:
    """Create IAOA+GNS algorithm with specified parameters."""
    return AlgorithmFactory.create_algorithm('iaoa_gns', {
        'pop_size': pop_size, 'max_iterations': max_iterations
    })