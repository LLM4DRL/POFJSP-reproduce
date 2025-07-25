"""
Baseline Scheduling Algorithms for POFJSP Comparison

Implements various scheduling algorithms including exact algorithms, heuristics,
metaheuristics, and hybrid approaches for comparing against IAOA+GNS.
"""

import numpy as np
import random
import copy
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from problems.problem_instance import ProblemInstance, Solution, Operation
from algorithms.decoder import decode_solution
from exceptions import AlgorithmError, ValidationError


@dataclass
class AlgorithmResult:
    """Result from running a scheduling algorithm."""
    algorithm_name: str
    makespan: float
    execution_time: float
    solution: Optional[Solution] = None
    additional_metrics: Dict = None


class BaseSchedulingAlgorithm(ABC):
    """Abstract base class for scheduling algorithms."""
    
    @abstractmethod
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        """Solve the problem and return results."""
        pass
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        pass


class DispatchingRulesAlgorithm(BaseSchedulingAlgorithm):
    """Dispatching rules-based heuristic algorithms."""
    
    def __init__(self, rule: str = "SPT"):
        """
        Initialize with dispatching rule.
        
        Args:
            rule: Dispatching rule ('SPT', 'LPT', 'EST', 'LST', 'FIFO')
        """
        self.rule = rule.upper()
        if self.rule not in ['SPT', 'LPT', 'EST', 'LST', 'FIFO']:
            raise ValidationError(f"Unknown dispatching rule: {rule}")
    
    @property
    def algorithm_name(self) -> str:
        return f"Dispatching_{self.rule}"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Build operation sequence using dispatching rule
            available_ops = list(problem.all_operations)
            scheduled_ops = []
            scheduled_set = set()
            
            while available_ops:
                # Find schedulable operations
                schedulable = []
                for op in available_ops:
                    predecessors = problem.predecessors_map.get(op, set())
                    if predecessors.issubset(scheduled_set):
                        schedulable.append(op)
                
                if not schedulable:
                    break
                
                # Select operation based on rule
                selected_op = self._select_operation(schedulable, problem)
                scheduled_ops.append(selected_op)
                scheduled_set.add(selected_op)
                available_ops.remove(selected_op)
            
            # Assign machines using load balancing
            machine_assignment = self._assign_machines(scheduled_ops, problem)
            
            # Create and evaluate solution
            solution = Solution(scheduled_ops, machine_assignment)
            makespan, _, _ = decode_solution(solution, problem)
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=makespan,
                execution_time=execution_time,
                solution=solution
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )
    
    def _select_operation(self, operations: List[Operation], problem: ProblemInstance) -> Operation:
        """Select operation based on dispatching rule."""
        if self.rule == "SPT":  # Shortest Processing Time
            return min(operations, key=lambda op: min(
                problem.get_processing_time(op, m) 
                for m in problem.get_valid_machines(op)
            ))
        elif self.rule == "LPT":  # Longest Processing Time
            return max(operations, key=lambda op: min(
                problem.get_processing_time(op, m) 
                for m in problem.get_valid_machines(op)
            ))
        elif self.rule == "EST":  # Earliest Start Time (simplified)
            return operations[0]  # First available
        elif self.rule == "LST":  # Latest Start Time (simplified)
            return operations[-1]  # Last available
        else:  # FIFO
            return operations[0]
    
    def _assign_machines(self, operations: List[Operation], problem: ProblemInstance) -> List[int]:
        """Assign machines using load balancing."""
        assignment = []
        machine_loads = [0.0] * problem.num_machines
        
        for op in operations:
            valid_machines = problem.get_valid_machines(op)
            best_machine = min(valid_machines, 
                             key=lambda m: machine_loads[m] + problem.get_processing_time(op, m))
            assignment.append(best_machine)
            machine_loads[best_machine] += problem.get_processing_time(op, best_machine)
        
        return assignment


class GeneticAlgorithm(BaseSchedulingAlgorithm):
    """Genetic Algorithm for POFJSP."""
    
    def __init__(self, pop_size: int = 50, generations: int = 100, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.2):
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    @property
    def algorithm_name(self) -> str:
        return "GeneticAlgorithm"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Initialize population
            population = self._initialize_population(problem)
            best_solution = min(population, key=lambda s: s.makespan)
            
            for generation in range(self.generations):
                if time.time() - start_time > timeout:
                    break
                
                # Selection, crossover, and mutation
                new_population = []
                
                for _ in range(self.pop_size // 2):
                    # Tournament selection
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Crossover
                    if random.random() < self.crossover_rate:
                        child1, child2 = self._crossover(parent1, parent2, problem)
                    else:
                        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                    
                    # Mutation
                    if random.random() < self.mutation_rate:
                        self._mutate(child1, problem)
                    if random.random() < self.mutation_rate:
                        self._mutate(child2, problem)
                    
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
                solution=best_solution,
                additional_metrics={"generations": generation + 1}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )
    
    def _initialize_population(self, problem: ProblemInstance) -> List[Solution]:
        """Initialize random population."""
        population = []
        for _ in range(self.pop_size):
            operations = list(problem.all_operations)
            random.shuffle(operations)
            machines = [random.choice(problem.get_valid_machines(op)) for op in operations]
            solution = Solution(operations, machines)
            decode_solution(solution, problem)
            population.append(solution)
        return population
    
    def _tournament_selection(self, population: List[Solution], tournament_size: int = 3) -> Solution:
        """Tournament selection."""
        tournament = random.choices(population, k=tournament_size)
        return min(tournament, key=lambda s: s.makespan)
    
    def _crossover(self, parent1: Solution, parent2: Solution, 
                   problem: ProblemInstance) -> Tuple[Solution, Solution]:
        """Order crossover for operation sequences."""
        size = len(parent1.operation_sequence)
        start, end = sorted(random.choices(range(size), k=2))
        
        # Child 1
        child1_ops = [None] * size
        child1_ops[start:end] = parent1.operation_sequence[start:end]
        
        p2_remaining = [op for op in parent2.operation_sequence if op not in child1_ops]
        j = 0
        for i in range(size):
            if child1_ops[i] is None:
                child1_ops[i] = p2_remaining[j]
                j += 1
        
        # Child 2 (symmetric)
        child2_ops = [None] * size
        child2_ops[start:end] = parent2.operation_sequence[start:end]
        
        p1_remaining = [op for op in parent1.operation_sequence if op not in child2_ops]
        j = 0
        for i in range(size):
            if child2_ops[i] is None:
                child2_ops[i] = p1_remaining[j]
                j += 1
        
        # Machine assignment crossover
        child1_machines = []
        child2_machines = []
        for i in range(size):
            if random.random() < 0.5:
                child1_machines.append(parent1.machine_assignment[i])
                child2_machines.append(parent2.machine_assignment[i])
            else:
                child1_machines.append(parent2.machine_assignment[i])
                child2_machines.append(parent1.machine_assignment[i])
        
        child1 = Solution(child1_ops, child1_machines)
        child2 = Solution(child2_ops, child2_machines)
        
        decode_solution(child1, problem)
        decode_solution(child2, problem)
        
        return child1, child2
    
    def _mutate(self, solution: Solution, problem: ProblemInstance):
        """Mutation operator."""
        # Swap two operations
        if len(solution.operation_sequence) > 1:
            i, j = random.choices(range(len(solution.operation_sequence)), k=2)
            solution.operation_sequence[i], solution.operation_sequence[j] = \
                solution.operation_sequence[j], solution.operation_sequence[i]
        
        # Change machine assignment
        for i in range(len(solution.machine_assignment)):
            if random.random() < 0.1:  # 10% chance per operation
                op = solution.operation_sequence[i]
                valid_machines = problem.get_valid_machines(op)
                solution.machine_assignment[i] = random.choice(valid_machines)
        
        decode_solution(solution, problem)


class SimulatedAnnealing(BaseSchedulingAlgorithm):
    """Simulated Annealing algorithm for POFJSP."""
    
    def __init__(self, initial_temp: float = 1000.0, cooling_rate: float = 0.95, 
                 min_temp: float = 1.0, max_iterations: int = 10000):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
    
    @property
    def algorithm_name(self) -> str:
        return "SimulatedAnnealing"
    
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
            
            for iteration in range(self.max_iterations):
                if time.time() - start_time > timeout or temperature < self.min_temp:
                    break
                
                # Generate neighbor
                neighbor = self._generate_neighbor(current_solution, problem)
                
                # Accept or reject
                delta = neighbor.makespan - current_solution.makespan
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_solution = neighbor
                    
                    if current_solution.makespan < best_solution.makespan:
                        best_solution = copy.deepcopy(current_solution)
                
                # Cool down
                temperature *= self.cooling_rate
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=best_solution.makespan,
                execution_time=execution_time,
                solution=best_solution,
                additional_metrics={"iterations": iteration + 1, "final_temp": temperature}
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
                i, j = random.choices(range(len(neighbor.operation_sequence)), k=2)
                neighbor.operation_sequence[i], neighbor.operation_sequence[j] = \
                    neighbor.operation_sequence[j], neighbor.operation_sequence[i]
        else:  # Change machine assignment
            idx = random.randint(0, len(neighbor.machine_assignment) - 1)
            op = neighbor.operation_sequence[idx]
            valid_machines = problem.get_valid_machines(op)
            neighbor.machine_assignment[idx] = random.choice(valid_machines)
        
        decode_solution(neighbor, problem)
        return neighbor


class RandomAlgorithm(BaseSchedulingAlgorithm):
    """Random algorithm for baseline comparison."""
    
    def __init__(self, num_trials: int = 1000):
        self.num_trials = num_trials
    
    @property
    def algorithm_name(self) -> str:
        return "RandomSearch"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            best_makespan = float('inf')
            best_solution = None
            
            for trial in range(self.num_trials):
                if time.time() - start_time > timeout:
                    break
                
                # Generate random solution
                operations = list(problem.all_operations)
                random.shuffle(operations)
                machines = [random.choice(problem.get_valid_machines(op)) for op in operations]
                solution = Solution(operations, machines)
                makespan, _, _ = decode_solution(solution, problem)
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_solution = copy.deepcopy(solution)
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=best_makespan,
                execution_time=execution_time,
                solution=best_solution,
                additional_metrics={"trials": trial + 1}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )


class GreedyAlgorithm(BaseSchedulingAlgorithm):
    """Greedy algorithm using earliest completion time."""
    
    @property
    def algorithm_name(self) -> str:
        return "GreedyScheduling"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Track machine availability
            machine_available_time = [0.0] * problem.num_machines
            operation_sequence = []
            machine_assignment = []
            
            # Get all operations and sort by job order
            available_ops = list(problem.all_operations)
            scheduled_set = set()
            
            while available_ops:
                # Find schedulable operations
                schedulable = []
                for op in available_ops:
                    predecessors = problem.predecessors_map.get(op, set())
                    if predecessors.issubset(scheduled_set):
                        schedulable.append(op)
                
                if not schedulable:
                    break
                
                # Select operation and machine with earliest completion
                best_completion = float('inf')
                best_op = None
                best_machine = None
                
                for op in schedulable:
                    for machine in problem.get_valid_machines(op):
                        proc_time = problem.get_processing_time(op, machine)
                        completion_time = machine_available_time[machine] + proc_time
                        
                        if completion_time < best_completion:
                            best_completion = completion_time
                            best_op = op
                            best_machine = machine
                
                # Schedule the best operation
                operation_sequence.append(best_op)
                machine_assignment.append(best_machine)
                machine_available_time[best_machine] = best_completion
                scheduled_set.add(best_op)
                available_ops.remove(best_op)
            
            # Create and evaluate solution
            solution = Solution(operation_sequence, machine_assignment)
            makespan, _, _ = decode_solution(solution, problem)
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=makespan,
                execution_time=execution_time,
                solution=solution
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )