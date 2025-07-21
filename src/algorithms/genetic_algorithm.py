"""
Genetic Algorithm for POFJSP (Partially Ordered Flexible Job Shop Scheduling)

This module implements a Genetic Algorithm specifically designed for POFJSP
with precedence constraints and flexible machine assignments.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import copy
from dataclasses import dataclass

from src.problems.problem_instance import ProblemInstance
from src.algorithms.decoder import decode_solution


@dataclass
class GAParams:
    """Parameters for Genetic Algorithm."""
    population_size: int = 100
    max_generations: int = 1000
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1
    tournament_size: int = 3
    diversity_threshold: float = 0.01
    no_improvement_limit: int = 50


class Individual:
    """Represents an individual in the GA population."""
    
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.chromosome = None  # List of (job_id, operation_id, machine_id)
        self.fitness = float('inf')
        self.makespan = float('inf')
        
    def initialize_random(self) -> None:
        """Initialize chromosome with random valid solution."""
        chromosome = []
        
        # Create all operations (job_id, operation_id)
        operations = []
        for job_id in range(self.problem.num_jobs):
            for op_id in range(self.problem.num_operations_per_job[job_id]):
                operations.append((job_id, op_id))
        
        # Random permutation respecting precedence constraints
        valid_sequence = self._generate_valid_sequence(operations)
        
        # Assign machines to operations
        for job_id, op_id in valid_sequence:
            # Get valid machines for this operation
            valid_machines = []
            for machine_id in range(self.problem.num_machines):
                if self.problem.processing_times[job_id][op_id][machine_id] != float('inf'):
                    valid_machines.append(machine_id)
            
            if valid_machines:
                machine_id = random.choice(valid_machines)
                chromosome.append((job_id, op_id, machine_id))
        
        self.chromosome = chromosome
        self.evaluate()
    
    def _generate_valid_sequence(self, operations: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Generate a valid sequence respecting precedence constraints."""
        sequence = []
        remaining = set(operations)
        
        while remaining:
            # Find operations whose predecessors are all scheduled
            available = []
            for job_id, op_id in remaining:
                op_key = f"({job_id}, {op_id})"
                predecessors = self.problem.predecessors_map.get(op_key, [])
                
                # Check if all predecessors are in sequence
                all_scheduled = True
                for pred in predecessors:
                    pred_job, pred_op = map(int, pred.strip('()').split(','))
                    if (pred_job, pred_op) not in sequence:
                        all_scheduled = False
                        break
                
                if all_scheduled:
                    available.append((job_id, op_id))
            
            if not available:
                # Should not happen with valid precedence constraints
                break
                
            # Randomly select from available
            selected = random.choice(available)
            sequence.append(selected)
            remaining.remove(selected)
        
        return sequence
    
    def evaluate(self) -> None:
        """Evaluate the individual's fitness (makespan)."""
        try:
            from src.problems.problem_instance import Solution
            
            # Convert chromosome format to Solution format
            operation_sequence = []
            machine_assignment = []
            
            for job_id, op_id, machine_id in self.chromosome:
                operation_sequence.append((job_id, op_id))
                machine_assignment.append(machine_id)
            
            solution = Solution(operation_sequence, machine_assignment)
            makespan, _, _ = decode_solution(solution, self.problem)
            self.makespan = makespan
            self.fitness = self.makespan  # Minimize makespan
        except Exception as e:
            self.makespan = float('inf')
            self.fitness = float('inf')
    
    def crossover(self, other: 'Individual') -> 'Individual':
        """Perform crossover with another individual."""
        child = Individual(self.problem)
        
        # Order-based crossover (respecting precedence constraints)
        operations1 = [(chrom[0], chrom[1]) for chrom in self.chromosome]
        operations2 = [(chrom[0], chrom[1]) for chrom in other.chromosome]
        
        # Create child sequence
        child_sequence = self._order_crossover(operations1, operations2)
        
        # Assign machines (machine assignment crossover)
        child_chromosome = []
        for job_id, op_id in child_sequence:
            if random.random() < 0.5:
                # Take machine from parent 1
                for chrom in self.chromosome:
                    if chrom[0] == job_id and chrom[1] == op_id:
                        machine_id = chrom[2]
                        break
                else:
                    machine_id = self._get_random_machine(job_id, op_id)
            else:
                # Take machine from parent 2
                for chrom in other.chromosome:
                    if chrom[0] == job_id and chrom[1] == op_id:
                        machine_id = chrom[2]
                        break
                else:
                    machine_id = self._get_random_machine(job_id, op_id)
            
            child_chromosome.append((job_id, op_id, machine_id))
        
        child.chromosome = child_chromosome
        child.evaluate()
        return child
    
    def _order_crossover(self, seq1: List[Tuple[int, int]], seq2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Perform order-based crossover respecting precedence constraints."""
        n = len(seq1)
        start, end = sorted(random.sample(range(n), 2))
        
        # Copy segment from parent 1
        child_segment = seq1[start:end]
        
        # Fill remaining positions from parent 2
        child_sequence = []
        segment_set = set(child_segment)
        
        for op in seq2:
            if op not in segment_set and self._is_valid_at_position(child_sequence, op):
                child_sequence.append(op)
        
        # Insert segment at correct position
        child_sequence[start:start] = child_segment
        
        return child_sequence
    
    def _is_valid_at_position(self, sequence: List[Tuple[int, int]], operation: Tuple[int, int]) -> bool:
        """Check if operation is valid at current position."""
        job_id, op_id = operation
        op_key = f"({job_id}, {op_id})"
        predecessors = self.problem.predecessors_map.get(op_key, [])
        
        for pred in predecessors:
            pred_job, pred_op = map(int, pred.strip('()').split(','))
            if (pred_job, pred_op) not in sequence:
                return False
        
        return True
    
    def mutate(self) -> None:
        """Perform mutation on the individual."""
        if random.random() < 0.5:
            # Sequence mutation (swap two valid operations)
            self._sequence_mutation()
        else:
            # Machine mutation
            self._machine_mutation()
    
    def _sequence_mutation(self) -> None:
        """Mutate the operation sequence."""
        if len(self.chromosome) < 2:
            return
        
        # Find valid swaps respecting precedence
        valid_swaps = []
        for i in range(len(self.chromosome) - 1):
            for j in range(i + 1, len(self.chromosome)):
                # Check if swap is valid
                temp_chrom = self.chromosome[:]
                temp_chrom[i], temp_chrom[j] = temp_chrom[j], temp_chrom[i]
                
                # Check precedence constraints
                if self._is_valid_sequence(temp_chrom):
                    valid_swaps.append((i, j))
        
        if valid_swaps:
            i, j = random.choice(valid_swaps)
            self.chromosome[i], self.chromosome[j] = self.chromosome[j], self.chromosome[i]
            self.evaluate()
    
    def _machine_mutation(self) -> None:
        """Mutate machine assignments."""
        for i in range(len(self.chromosome)):
            if random.random() < self.problem.num_machines / len(self.chromosome):
                job_id, op_id, old_machine = self.chromosome[i]
                
                # Get valid machines
                valid_machines = []
                for machine_id in range(self.problem.num_machines):
                    if self.problem.processing_times[job_id][op_id][machine_id] != float('inf'):
                        valid_machines.append(machine_id)
                
                if len(valid_machines) > 1:
                    valid_machines.remove(old_machine)
                    new_machine = random.choice(valid_machines)
                    self.chromosome[i] = (job_id, op_id, new_machine)
        
        self.evaluate()
    
    def _is_valid_sequence(self, chromosome: List[Tuple[int, int, int]]) -> bool:
        """Check if chromosome represents a valid sequence."""
        sequence = [(chrom[0], chrom[1]) for chrom in chromosome]
        
        for idx, (job_id, op_id) in enumerate(sequence):
            op_key = f"({job_id}, {op_id})"
            predecessors = self.problem.predecessors_map.get(op_key, [])
            
            for pred in predecessors:
                pred_job, pred_op = map(int, pred.strip('()').split(','))
                if (pred_job, pred_op) not in sequence[:idx]:
                    return False
        
        return True
    
    def _get_random_machine(self, job_id: int, op_id: int) -> int:
        """Get a random valid machine for an operation."""
        valid_machines = []
        for machine_id in range(self.problem.num_machines):
            if self.problem.processing_times[job_id][op_id][machine_id] != float('inf'):
                valid_machines.append(machine_id)
        
        return random.choice(valid_machines) if valid_machines else 0


class GeneticAlgorithm:
    """Genetic Algorithm for POFJSP."""
    
    def __init__(self, problem: ProblemInstance, params: GAParams):
        self.problem = problem
        self.params = params
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness_history = []
        self.diversity_history = []
        self.no_improvement_count = 0
    
    def initialize_population(self) -> None:
        """Initialize the population with random individuals."""
        self.population = []
        for _ in range(self.params.population_size):
            individual = Individual(self.problem)
            individual.initialize_random()
            self.population.append(individual)
        
        self.population.sort(key=lambda x: x.fitness)
        self.best_individual = copy.deepcopy(self.population[0])
    
    def tournament_selection(self) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, self.params.tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    def evolve_generation(self) -> None:
        """Evolve one generation."""
        new_population = []
        
        # Elitism: keep best individuals
        num_elites = max(1, int(self.params.elitism_rate * self.params.population_size))
        new_population.extend(copy.deepcopy(self.population[:num_elites]))
        
        # Generate offspring
        while len(new_population) < self.params.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < self.params.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # Mutation
            if random.random() < self.params.mutation_rate:
                child.mutate()
            
            new_population.append(child)
        
        self.population = new_population
        self.population.sort(key=lambda x: x.fitness)
        self.generation += 1
    
    def check_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        fitnesses = [ind.fitness for ind in self.population]
        return np.std(fitnesses)
    
    def run(self, verbose: bool = False) -> Individual:
        """Run the genetic algorithm."""
        self.initialize_population()
        
        if verbose:
            print(f"Initial best fitness: {self.best_individual.fitness:.2f}")
        
        for generation in range(self.params.max_generations):
            # Evolve generation
            self.evolve_generation()
            
            # Update best individual
            if self.population[0].fitness < self.best_individual.fitness:
                self.best_individual = copy.deepcopy(self.population[0])
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Track history
            self.best_fitness_history.append(self.best_individual.fitness)
            self.diversity_history.append(self.check_diversity())
            
            if verbose and generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_individual.fitness:.2f}")
            
            # Check termination criteria
            if self.no_improvement_count >= self.params.no_improvement_limit:
                if verbose:
                    print(f"No improvement for {self.params.no_improvement_limit} generations. Stopping.")
                break
            
            if self.check_diversity() < self.params.diversity_threshold:
                if verbose:
                    print("Population diversity too low. Stopping.")
                break
        
        return self.best_individual
    
    def get_statistics(self) -> Dict:
        """Get algorithm statistics."""
        return {
            'generations': self.generation,
            'best_fitness': self.best_individual.fitness,
            'final_diversity': self.check_diversity(),
            'convergence_rate': len(self.best_fitness_history),
            'fitness_history': self.best_fitness_history
        }


def solve_with_ga(problem: ProblemInstance, params: GAParams = None, verbose: bool = False) -> Dict:
    """Solve POFJSP with Genetic Algorithm."""
    if params is None:
        params = GAParams()
    
    ga = GeneticAlgorithm(problem, params)
    best_individual = ga.run(verbose=verbose)
    
    # Convert to Solution format
    from src.problems.problem_instance import Solution
    
    operation_sequence = []
    machine_assignment = []
    for job_id, op_id, machine_id in best_individual.chromosome:
        operation_sequence.append((job_id, op_id))
        machine_assignment.append(machine_id)
    
    solution = Solution(operation_sequence, machine_assignment)
    makespan, _, _ = decode_solution(solution, problem)
    
    return {
        'solution': solution,
        'makespan': solution.makespan,
        'statistics': ga.get_statistics()
    }


def main():
    """Main function for testing Genetic Algorithm."""
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from problems.problem_instance import ProblemInstance, Operation
    import numpy as np
    
    # Create a simple test problem
    print("Creating sample POFJSP problem...")
    num_operations_per_job = [2, 2]  # 2 jobs, 2 operations each
    processing_times = [
        np.array([[3, 5], [6, np.inf]]),  # Job 0: Op0 on M0=3,M1=5; Op1 on M0=6,M1=inf
        np.array([[4, 2], [np.inf, 7]])   # Job 1: Op0 on M0=4,M1=2; Op1 on M0=inf,M1=7
    ]
    
    # Simple precedence: op0 -> op1 for each job
    predecessors_map = {
        "(0, 1)": ["(0, 0)"],
        "(1, 1)": ["(1, 0)"]
    }
    successors_map = {
        "(0, 0)": ["(0, 1)"],
        "(1, 0)": ["(1, 1)"]
    }
    
    problem = ProblemInstance(
        num_jobs=2,
        num_machines=2,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )
    
    # Run Genetic Algorithm
    print("\nRunning Genetic Algorithm...")
    params = GAParams(
        population_size=20,  # Small params for testing
        max_generations=10
    )
    
    result = solve_with_ga(problem, params, verbose=True)
    
    if result['solution']:
        print(f"\n[SUCCESS] Algorithm completed successfully!")
        print(f"Final makespan: {result['makespan']}")
        print(f"Statistics: {result['statistics']}")
    else:
        print("[FAILED] Algorithm failed to find a solution")


if __name__ == "__main__":
    main()