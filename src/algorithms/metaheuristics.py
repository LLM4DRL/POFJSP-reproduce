"""
Advanced Metaheuristic and Hybrid Algorithms for POFJSP

This module implements additional metaheuristic algorithms and hybrid approaches
including Differential Evolution, Variable Neighborhood Search, and various
hybrid combinations.
"""

import numpy as np
import random
import copy
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from src.algorithms.baseline_algorithms import BaseSchedulingAlgorithm, AlgorithmResult, GeneticAlgorithm
from src.problems.problem_instance import ProblemInstance, Solution, Operation
from src.algorithms.decoder import decode_solution
from src.exceptions import AlgorithmError, ValidationError


class DifferentialEvolution(BaseSchedulingAlgorithm):
    """Differential Evolution algorithm for POFJSP."""
    
    def __init__(self, population_size: int = 30, max_iterations: int = 50,
                 F: float = 0.8, CR: float = 0.9):
        """
        Initialize DE algorithm.
        
        Args:
            population_size: Size of population
            max_iterations: Maximum number of iterations
            F: Differential weight
            CR: Crossover probability
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.F = F
        self.CR = CR
        
    @property
    def algorithm_name(self) -> str:
        return "DifferentialEvolution"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            n_ops = len(problem.all_operations)
            
            # Initialize population
            population = []
            fitness = []
            
            # Try to create initial population with at least some valid solutions
            max_init_attempts = self.population_size * 3
            init_attempts = 0
            
            while len(population) < self.population_size and init_attempts < max_init_attempts:
                init_attempts += 1
                individual = np.random.rand(n_ops)
                solution, makespan = self._individual_to_solution(individual, problem)
                
                population.append(individual)
                fitness.append(makespan)
                
                # If we have a valid solution, we can proceed
                if makespan != float('inf'):
                    break
            
            # Fill remaining population slots if needed
            while len(population) < self.population_size:
                individual = np.random.rand(n_ops)
                population.append(individual)
                fitness.append(float('inf'))
            
            # Find best individual
            finite_fitness = [f for f in fitness if f != float('inf')]
            if not finite_fitness:
                # All solutions are invalid, return failure
                execution_time = time.time() - start_time
                return AlgorithmResult(
                    algorithm_name=self.algorithm_name,
                    makespan=float('inf'),
                    execution_time=execution_time,
                    additional_metrics={"error": "Failed to generate any valid initial solutions"}
                )
            
            best_idx = np.argmin(fitness)
            best_solution, best_fitness = self._individual_to_solution(population[best_idx], problem)
            
            for iteration in range(self.max_iterations):
                if time.time() - start_time > timeout:
                    break
                
                for i in range(self.population_size):
                    # Select three different random individuals
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    
                    if len(indices) < 3:
                        continue  # Need at least 3 other individuals
                        
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    
                    # Mutation
                    mutant = population[a] + self.F * (population[b] - population[c])
                    mutant = np.clip(mutant, 0, 1)
                    
                    # Crossover
                    trial = population[i].copy()
                    for j in range(n_ops):
                        if np.random.rand() < self.CR:
                            trial[j] = mutant[j]
                    
                    # Selection
                    trial_solution, trial_fitness = self._individual_to_solution(trial, problem)
                    
                    # Only replace if the trial solution is better (and handle infinite fitness)
                    if (trial_fitness < fitness[i] and trial_fitness != float('inf')) or \
                       (fitness[i] == float('inf') and trial_fitness != float('inf')):
                        population[i] = trial
                        fitness[i] = trial_fitness
                        
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best_solution = trial_solution
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=best_fitness,
                execution_time=execution_time,
                solution=best_solution
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )
    
    def _individual_to_solution(self, individual: np.ndarray, problem: ProblemInstance) -> Tuple[Solution, float]:
        """Convert DE individual to scheduling solution."""
        operations = list(problem.all_operations)
        
        # Schedule operations respecting precedence constraints
        scheduled_ops = []
        scheduled_set = set()
        machine_assignment = []
        
        # Sort by priority but respect precedence
        remaining_ops = operations.copy()
        max_iterations = len(operations) * 2  # Prevent infinite loops
        iterations = 0
        
        while remaining_ops and iterations < max_iterations:
            iterations += 1
            
            # Find schedulable operations
            schedulable = []
            for op in remaining_ops:
                predecessors = problem.predecessors_map.get(op, set())
                if predecessors.issubset(scheduled_set):
                    schedulable.append(op)
            
            if not schedulable:
                # No operations can be scheduled - there might be a circular dependency
                # or missing operations. Return invalid solution.
                return Solution([], []), float('inf')
            
            # Select highest priority schedulable operation
            op_idx_map = {op: i for i, op in enumerate(operations)}
            best_op = max(schedulable, key=lambda op: individual[op_idx_map[op]])
            
            # Get valid machines for this operation
            try:
                valid_machines = problem.get_valid_machines(best_op)
                if not valid_machines:
                    # No valid machines for this operation
                    return Solution([], []), float('inf')
                
                # Select best machine (lowest processing time)
                best_machine = min(valid_machines,
                                 key=lambda m: problem.get_processing_time(best_op, m))
                
                # Check if processing time is valid
                proc_time = problem.get_processing_time(best_op, best_machine)
                if proc_time == np.inf or proc_time < 0:
                    # Invalid processing time, try next best machine
                    valid_proc_times = [(m, problem.get_processing_time(best_op, m)) 
                                      for m in valid_machines 
                                      if problem.get_processing_time(best_op, m) < np.inf and 
                                         problem.get_processing_time(best_op, m) >= 0]
                    
                    if not valid_proc_times:
                        # No machine has valid processing time
                        return Solution([], []), float('inf')
                    
                    best_machine = min(valid_proc_times, key=lambda x: x[1])[0]
                
            except Exception:
                # Error getting machine information
                return Solution([], []), float('inf')
            
            scheduled_ops.append(best_op)
            machine_assignment.append(best_machine)
            scheduled_set.add(best_op)
            remaining_ops.remove(best_op)
        
        # Check if all operations were scheduled
        if remaining_ops:
            # Some operations couldn't be scheduled
            return Solution([], []), float('inf')
        
        # Validate that we have complete solution
        if len(scheduled_ops) != len(operations) or len(machine_assignment) != len(operations):
            return Solution([], []), float('inf')
        
        try:
            solution = Solution(scheduled_ops, machine_assignment)
            makespan, _, _ = decode_solution(solution, problem)
            
            # Additional validation - ensure makespan is finite and positive
            if makespan == float('inf') or makespan < 0 or np.isnan(makespan):
                return Solution([], []), float('inf')
                
            return solution, makespan
        except Exception:
            # Error in solution creation or decoding
            return Solution([], []), float('inf')


class VariableNeighborhoodSearch(BaseSchedulingAlgorithm):
    """Variable Neighborhood Search algorithm for POFJSP."""
    
    def __init__(self, max_iterations: int = 100, k_max: int = 3):
        """
        Initialize VNS algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            k_max: Maximum neighborhood structure index
        """
        self.max_iterations = max_iterations
        self.k_max = k_max
        
    @property
    def algorithm_name(self) -> str:
        return "VariableNeighborhoodSearch"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Initialize with random solution
            operations = list(problem.all_operations)
            random.shuffle(operations)
            machines = [random.choice(problem.get_valid_machines(op)) for op in operations]
            current_solution = Solution(operations, machines)
            decode_solution(current_solution, problem)
            
            best_solution = copy.deepcopy(current_solution)
            
            for iteration in range(self.max_iterations):
                if time.time() - start_time > timeout:
                    break
                
                k = 1
                while k <= self.k_max:
                    # Shaking: generate random solution in k-th neighborhood
                    neighbor = self._shake(current_solution, k, problem)
                    
                    # Local search
                    improved_neighbor = self._local_search(neighbor, problem)
                    
                    # Move or not
                    if improved_neighbor.makespan < current_solution.makespan:
                        current_solution = improved_neighbor
                        k = 1  # Reset to first neighborhood
                        
                        if current_solution.makespan < best_solution.makespan:
                            best_solution = copy.deepcopy(current_solution)
                    else:
                        k += 1
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=best_solution.makespan,
                execution_time=execution_time,
                solution=best_solution
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )
    
    def _shake(self, solution: Solution, k: int, problem: ProblemInstance) -> Solution:
        """Shake the solution in k-th neighborhood."""
        neighbor = copy.deepcopy(solution)
        
        # Apply k random moves
        for _ in range(k):
            if random.random() < 0.5:  # Swap operations
                if len(neighbor.operation_sequence) > 1:
                    i, j = random.sample(range(len(neighbor.operation_sequence)), 2)
                    neighbor.operation_sequence[i], neighbor.operation_sequence[j] = \
                        neighbor.operation_sequence[j], neighbor.operation_sequence[i]
            else:  # Change machine assignment
                idx = random.randint(0, len(neighbor.machine_assignment) - 1)
                op = neighbor.operation_sequence[idx]
                valid_machines = problem.get_valid_machines(op)
                neighbor.machine_assignment[idx] = random.choice(valid_machines)
        
        decode_solution(neighbor, problem)
        return neighbor
    
    def _local_search(self, solution: Solution, problem: ProblemInstance) -> Solution:
        """Perform local search to improve solution."""
        current = copy.deepcopy(solution)
        improved = True
        
        while improved:
            improved = False
            best_neighbor = current
            
            # Try all possible swaps
            for i in range(len(current.operation_sequence)):
                for j in range(i + 1, len(current.operation_sequence)):
                    # Swap operations
                    neighbor = copy.deepcopy(current)
                    neighbor.operation_sequence[i], neighbor.operation_sequence[j] = \
                        neighbor.operation_sequence[j], neighbor.operation_sequence[i]
                    decode_solution(neighbor, problem)
                    
                    if neighbor.makespan < best_neighbor.makespan:
                        best_neighbor = neighbor
                        improved = True
                
                # Try different machine assignments
                op = current.operation_sequence[i]
                for machine in problem.get_valid_machines(op):
                    if machine != current.machine_assignment[i]:
                        neighbor = copy.deepcopy(current)
                        neighbor.machine_assignment[i] = machine
                        decode_solution(neighbor, problem)
                        
                        if neighbor.makespan < best_neighbor.makespan:
                            best_neighbor = neighbor
                            improved = True
            
            current = best_neighbor
        
        return current


class TabuSearch(BaseSchedulingAlgorithm):
    """Tabu Search algorithm for POFJSP."""
    
    def __init__(self, max_iterations: int = 100, tabu_tenure: int = 7):
        """
        Initialize Tabu Search algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            tabu_tenure: Tabu list tenure
        """
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        
    @property
    def algorithm_name(self) -> str:
        return "TabuSearch"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Initialize with random solution
            operations = list(problem.all_operations)
            random.shuffle(operations)
            machines = [random.choice(problem.get_valid_machines(op)) for op in operations]
            current_solution = Solution(operations, machines)
            decode_solution(current_solution, problem)
            
            best_solution = copy.deepcopy(current_solution)
            tabu_list = []
            
            for iteration in range(self.max_iterations):
                if time.time() - start_time > timeout:
                    break
                
                # Generate neighborhood
                neighbors = self._generate_neighbors(current_solution, problem)
                
                # Select best non-tabu move
                best_neighbor = None
                best_move = None
                best_makespan = float('inf')
                
                for neighbor, move in neighbors:
                    if move not in tabu_list or neighbor.makespan < best_solution.makespan:
                        if neighbor.makespan < best_makespan:
                            best_neighbor = neighbor
                            best_move = move
                            best_makespan = neighbor.makespan
                
                if best_neighbor is None:
                    break
                
                # Update current solution
                current_solution = best_neighbor
                
                # Update best solution
                if current_solution.makespan < best_solution.makespan:
                    best_solution = copy.deepcopy(current_solution)
                
                # Update tabu list
                tabu_list.append(best_move)
                if len(tabu_list) > self.tabu_tenure:
                    tabu_list.pop(0)
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=best_solution.makespan,
                execution_time=execution_time,
                solution=best_solution
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )
    
    def _generate_neighbors(self, solution: Solution, problem: ProblemInstance) -> List[Tuple[Solution, Tuple]]:
        """Generate neighborhood solutions."""
        neighbors = []
        
        # Swap-based neighbors
        for i in range(len(solution.operation_sequence)):
            for j in range(i + 1, len(solution.operation_sequence)):
                neighbor = copy.deepcopy(solution)
                neighbor.operation_sequence[i], neighbor.operation_sequence[j] = \
                    neighbor.operation_sequence[j], neighbor.operation_sequence[i]
                decode_solution(neighbor, problem)
                move = ('swap', i, j)
                neighbors.append((neighbor, move))
        
        # Machine assignment neighbors
        for i in range(len(solution.machine_assignment)):
            op = solution.operation_sequence[i]
            for machine in problem.get_valid_machines(op):
                if machine != solution.machine_assignment[i]:
                    neighbor = copy.deepcopy(solution)
                    neighbor.machine_assignment[i] = machine
                    decode_solution(neighbor, problem)
                    move = ('machine', i, machine)
                    neighbors.append((neighbor, move))
        
        return neighbors


class HybridGeneticLocalSearch(GeneticAlgorithm):
    """Hybrid Genetic Algorithm with Local Search for POFJSP."""
    
    def __init__(self, population_size: int = 50, max_generations: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 local_search_rate: float = 0.3):
        """
        Initialize Hybrid GA-LS algorithm.
        
        Args:
            population_size: Size of population
            max_generations: Maximum number of generations
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability
            local_search_rate: Probability of applying local search
        """
        super().__init__(population_size, max_generations, crossover_rate, mutation_rate)
        self.local_search_rate = local_search_rate
        
    @property
    def algorithm_name(self) -> str:
        return "HybridGeneticLocalSearch"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Initialize population
            population = self._initialize_population(problem)
            
            # Apply local search to some initial individuals
            for i in range(len(population)):
                if np.random.rand() < self.local_search_rate:
                    population[i] = self._local_search(population[i], problem)
            
            best_solution = min(population, key=lambda s: s.makespan)
            
            for generation in range(self.generations):
                if time.time() - start_time > timeout:
                    break
                
                # Create new population
                new_population = []
                
                # Elitism: keep best individuals
                elite_size = max(1, self.pop_size // 10)
                sorted_pop = sorted(population, key=lambda s: s.makespan)
                new_population.extend(copy.deepcopy(sorted_pop[:elite_size]))
                
                # Generate offspring
                while len(new_population) < self.pop_size:
                    # Selection
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Crossover
                    if np.random.rand() < self.crossover_rate:
                        child1, child2 = self._crossover(parent1, parent2, problem)
                    else:
                        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                    
                    # Mutation
                    if np.random.rand() < self.mutation_rate:
                        self._mutate(child1, problem)
                    if np.random.rand() < self.mutation_rate:
                        self._mutate(child2, problem)
                    
                    # Local search
                    if np.random.rand() < self.local_search_rate:
                        child1 = self._local_search(child1, problem)
                    if np.random.rand() < self.local_search_rate:
                        child2 = self._local_search(child2, problem)
                    
                    new_population.extend([child1, child2])
                
                population = new_population[:self.pop_size]
                current_best = min(population, key=lambda s: s.makespan)
                
                if current_best.makespan < best_solution.makespan:
                    best_solution = copy.deepcopy(current_best)
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=best_solution.makespan,
                execution_time=execution_time,
                solution=best_solution
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )
    
    def _local_search(self, individual: Solution, problem: ProblemInstance) -> Solution:
        """Apply local search to improve individual."""
        current = copy.deepcopy(individual)
        improved = True
        
        while improved:
            improved = False
            best_neighbor = current
            
            # Try swapping adjacent operations
            for i in range(len(current.operation_sequence) - 1):
                neighbor = copy.deepcopy(current)
                neighbor.operation_sequence[i], neighbor.operation_sequence[i+1] = \
                    neighbor.operation_sequence[i+1], neighbor.operation_sequence[i]
                
                # Check if still valid (respects precedence)
                if self._is_valid_sequence(neighbor.operation_sequence, problem):
                    decode_solution(neighbor, problem)
                    if neighbor.makespan < best_neighbor.makespan:
                        best_neighbor = neighbor
                        improved = True
            
            current = best_neighbor
        
        return current
    
    def _is_valid_sequence(self, sequence: List[Operation], problem: ProblemInstance) -> bool:
        """Check if operation sequence respects precedence constraints."""
        scheduled_set = set()
        
        for op in sequence:
            predecessors = problem.predecessors_map.get(op, set())
            if not predecessors.issubset(scheduled_set):
                return False
            scheduled_set.add(op)
        
        return True


class MemorybasedSimulatedAnnealing(BaseSchedulingAlgorithm):
    """Memory-based Simulated Annealing with diversification."""
    
    def __init__(self, initial_temp: float = 1000.0, cooling_rate: float = 0.95, 
                 min_temp: float = 1.0, max_iterations: int = 10000,
                 memory_size: int = 20):
        """
        Initialize memory-based SA.
        
        Args:
            initial_temp: Initial temperature
            cooling_rate: Cooling rate
            min_temp: Minimum temperature
            max_iterations: Maximum iterations
            memory_size: Size of elite solution memory
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.memory_size = memory_size
        
    @property
    def algorithm_name(self) -> str:
        return "MemoryBasedSimulatedAnnealing"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Initialize with random solution
            operations = list(problem.all_operations)
            random.shuffle(operations)
            machines = [random.choice(problem.get_valid_machines(op)) for op in operations]
            current_solution = Solution(operations, machines)
            decode_solution(current_solution, problem)
            
            best_solution = copy.deepcopy(current_solution)
            temperature = self.initial_temp
            memory = [copy.deepcopy(current_solution)]  # Elite solution memory
            
            for iteration in range(self.max_iterations):
                if time.time() - start_time > timeout or temperature < self.min_temp:
                    break
                
                # Generate neighbor
                if random.random() < 0.1 and memory:  # Diversification
                    # Start from a random solution in memory
                    current_solution = copy.deepcopy(random.choice(memory))
                
                neighbor = self._generate_neighbor(current_solution, problem)
                
                # Accept or reject
                delta = neighbor.makespan - current_solution.makespan
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_solution = neighbor
                    
                    # Update best solution
                    if current_solution.makespan < best_solution.makespan:
                        best_solution = copy.deepcopy(current_solution)
                        
                        # Update memory
                        memory.append(copy.deepcopy(current_solution))
                        if len(memory) > self.memory_size:
                            # Remove worst solution from memory
                            memory.sort(key=lambda s: s.makespan)
                            memory = memory[:self.memory_size]
                
                # Cool down
                temperature *= self.cooling_rate
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=best_solution.makespan,
                execution_time=execution_time,
                solution=best_solution,
                additional_metrics={"iterations": iteration + 1, "memory_size": len(memory)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )
    
    def _generate_neighbor(self, solution: Solution, problem: ProblemInstance) -> Solution:
        """Generate neighbor solution."""
        neighbor = copy.deepcopy(solution)
        
        if random.random() < 0.5:  # Swap operations
            if len(neighbor.operation_sequence) > 1:
                i, j = random.sample(range(len(neighbor.operation_sequence)), 2)
                neighbor.operation_sequence[i], neighbor.operation_sequence[j] = \
                    neighbor.operation_sequence[j], neighbor.operation_sequence[i]
        else:  # Change machine assignment
            idx = random.randint(0, len(neighbor.machine_assignment) - 1)
            op = neighbor.operation_sequence[idx]
            valid_machines = problem.get_valid_machines(op)
            neighbor.machine_assignment[idx] = random.choice(valid_machines)
        
        decode_solution(neighbor, problem)
        return neighbor