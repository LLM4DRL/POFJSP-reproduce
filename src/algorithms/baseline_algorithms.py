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

from src.problems.problem_instance import ProblemInstance, Solution, Operation
from src.algorithms.decoder import decode_solution
from src.exceptions import AlgorithmError, ValidationError


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


class AntColonyOptimization(BaseSchedulingAlgorithm):
    """Ant Colony Optimization algorithm for POFJSP."""
    
    def __init__(self, n_ants: int = 20, max_iterations: int = 50, 
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, q0: float = 0.9):
        """
        Initialize ACO algorithm.
        
        Args:
            n_ants: Number of ants
            max_iterations: Maximum number of iterations
            alpha: Pheromone importance
            beta: Heuristic information importance
            rho: Pheromone evaporation rate
            q0: Exploitation probability
        """
        self.n_ants = n_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        
    @property
    def algorithm_name(self) -> str:
        return "ACO"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Initialize pheromone matrix
            n_ops = len(problem.all_operations)
            pheromone = np.ones((n_ops, n_ops)) * 0.1
            
            best_solution = None
            best_makespan = float('inf')
            
            for iteration in range(self.max_iterations):
                if time.time() - start_time > timeout:
                    break
                    
                # Generate solutions with ants
                ant_solutions = []
                ant_makespans = []
                
                for ant in range(self.n_ants):
                    solution, makespan = self._construct_solution(problem, pheromone)
                    ant_solutions.append(solution)
                    ant_makespans.append(makespan)
                    
                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_solution = solution
                
                # Update pheromones
                self._update_pheromones(pheromone, ant_solutions, ant_makespans, problem)
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=best_makespan,
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
    
    def _construct_solution(self, problem: ProblemInstance, pheromone: np.ndarray) -> Tuple[Solution, float]:
        """Construct a solution using ant colony construction."""
        operations = list(problem.all_operations)
        op_to_idx = {op: i for i, op in enumerate(operations)}
        
        available_ops = set(operations)
        scheduled_ops = []
        scheduled_set = set()
        machine_assignment = []
        
        while available_ops:
            # Find schedulable operations
            schedulable = []
            for op in available_ops:
                predecessors = problem.predecessors_map.get(op, set())
                if predecessors.issubset(scheduled_set):
                    schedulable.append(op)
            
            if not schedulable:
                break
            
            # Select operation using ACO probabilities
            if len(scheduled_ops) == 0:
                selected_op = random.choice(schedulable)
            else:
                last_op_idx = op_to_idx[scheduled_ops[-1]]
                probabilities = []
                
                for op in schedulable:
                    op_idx = op_to_idx[op]
                    # Heuristic: prefer operations with shorter processing time
                    min_proc_time = min(problem.get_processing_time(op, m) 
                                      for m in problem.get_valid_machines(op))
                    heuristic = 1.0 / (min_proc_time + 1e-6)
                    
                    prob = (pheromone[last_op_idx][op_idx] ** self.alpha) * (heuristic ** self.beta)
                    probabilities.append(prob)
                
                # Normalize probabilities
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                    # Convert probabilities to numpy array and use indices for selection
                    prob_array = np.array(probabilities)
                    selected_idx = np.random.choice(len(schedulable), p=prob_array)
                    selected_op = schedulable[selected_idx]
                else:
                    selected_op = random.choice(schedulable)
            
            # Select machine with minimum completion time
            best_machine = min(problem.get_valid_machines(selected_op),
                             key=lambda m: problem.get_processing_time(selected_op, m))
            
            scheduled_ops.append(selected_op)
            machine_assignment.append(best_machine)
            scheduled_set.add(selected_op)
            available_ops.remove(selected_op)
        
        solution = Solution(scheduled_ops, machine_assignment)
        makespan, _, _ = decode_solution(solution, problem)
        return solution, makespan
    
    def _update_pheromones(self, pheromone: np.ndarray, solutions: List[Solution], makespans: List[float], problem: ProblemInstance):
        """Update pheromone matrix."""
        # Evaporation
        pheromone *= (1 - self.rho)
        
        # Add pheromone from best solutions
        best_idx = np.argmin(makespans)
        best_solution = solutions[best_idx]
        best_makespan = makespans[best_idx]
        
        # Use the original operations list to maintain consistent indexing
        all_operations = list(problem.all_operations)
        op_to_idx = {op: i for i, op in enumerate(all_operations)}
        
        operations = list(best_solution.operation_sequence)
        
        # Add pheromone along the best path
        for i in range(len(operations) - 1):
            if operations[i] in op_to_idx and operations[i+1] in op_to_idx:
                idx1 = op_to_idx[operations[i]]
                idx2 = op_to_idx[operations[i+1]]
                pheromone[idx1][idx2] += 1.0 / best_makespan


class ParticleSwarmOptimization(BaseSchedulingAlgorithm):
    """Particle Swarm Optimization algorithm for POFJSP."""
    
    def __init__(self, n_particles: int = 30, max_iterations: int = 50,
                 w: float = 0.7, c1: float = 2.0, c2: float = 2.0):
        """
        Initialize PSO algorithm.
        
        Args:
            n_particles: Number of particles
            max_iterations: Maximum number of iterations
            w: Inertia weight
            c1: Cognitive component
            c2: Social component
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
    @property
    def algorithm_name(self) -> str:
        return "PSO"
    
    def solve(self, problem: ProblemInstance, timeout: float = 300.0) -> AlgorithmResult:
        start_time = time.time()
        
        try:
            # Initialize particles
            particles = []
            velocities = []
            personal_best = []
            personal_best_fitness = []
            
            n_ops = len(problem.all_operations)
            
            for _ in range(self.n_particles):
                # Initialize position (priority values for operations)
                position = np.random.rand(n_ops)
                velocity = np.random.rand(n_ops) * 0.1
                
                particles.append(position)
                velocities.append(velocity)
                personal_best.append(position.copy())
                personal_best_fitness.append(float('inf'))
            
            global_best = None
            global_best_fitness = float('inf')
            
            for iteration in range(self.max_iterations):
                if time.time() - start_time > timeout:
                    break
                
                for i in range(self.n_particles):
                    # Convert position to solution
                    solution, makespan = self._position_to_solution(particles[i], problem)
                    
                    # Update personal best
                    if makespan < personal_best_fitness[i]:
                        personal_best_fitness[i] = makespan
                        personal_best[i] = particles[i].copy()
                    
                    # Update global best
                    if makespan < global_best_fitness:
                        global_best_fitness = makespan
                        global_best = solution
                
                # Update velocities and positions
                for i in range(self.n_particles):
                    r1, r2 = np.random.rand(), np.random.rand()
                    
                    velocities[i] = (self.w * velocities[i] + 
                                   self.c1 * r1 * (personal_best[i] - particles[i]) +
                                   self.c2 * r2 * (personal_best[np.argmin(personal_best_fitness)] - particles[i]))
                    
                    particles[i] += velocities[i]
                    particles[i] = np.clip(particles[i], 0, 1)
            
            execution_time = time.time() - start_time
            
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=global_best_fitness,
                execution_time=execution_time,
                solution=global_best
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AlgorithmResult(
                algorithm_name=self.algorithm_name,
                makespan=float('inf'),
                execution_time=execution_time,
                additional_metrics={"error": str(e)}
            )
    
    def _position_to_solution(self, position: np.ndarray, problem: ProblemInstance) -> Tuple[Solution, float]:
        """Convert PSO position to scheduling solution."""
        operations = list(problem.all_operations)
        
        # Schedule operations respecting precedence constraints
        available_ops = set(operations)
        scheduled_ops = []
        scheduled_set = set()
        machine_assignment = []
        
        # Sort by priority but respect precedence
        remaining_ops = operations.copy()
        
        while remaining_ops:
            # Find schedulable operations
            schedulable = []
            for op in remaining_ops:
                predecessors = problem.predecessors_map.get(op, set())
                if predecessors.issubset(scheduled_set):
                    schedulable.append(op)
            
            if not schedulable:
                break
            
            # Select highest priority schedulable operation
            op_idx_map = {op: i for i, op in enumerate(operations)}
            best_op = max(schedulable, key=lambda op: position[op_idx_map[op]])
            
            # Select best machine
            best_machine = min(problem.get_valid_machines(best_op),
                             key=lambda m: problem.get_processing_time(best_op, m))
            
            scheduled_ops.append(best_op)
            machine_assignment.append(best_machine)
            scheduled_set.add(best_op)
            remaining_ops.remove(best_op)
        
        solution = Solution(scheduled_ops, machine_assignment)
        makespan, _, _ = decode_solution(solution, problem)
        return solution, makespan


# Create instances of specific algorithms for easy access  
SPTDispatchingRule = lambda: DispatchingRulesAlgorithm("SPT")
LPTDispatchingRule = lambda: DispatchingRulesAlgorithm("LPT")
ESTDispatchingRule = lambda: DispatchingRulesAlgorithm("EST")
LSTDispatchingRule = lambda: DispatchingRulesAlgorithm("LST")
FIFODispatchingRule = lambda: DispatchingRulesAlgorithm("FIFO")
