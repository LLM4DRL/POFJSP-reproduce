"""
Simulated Annealing for POFJSP (Partially Ordered Flexible Job Shop Scheduling)

This module implements a Simulated Annealing algorithm specifically designed for POFJSP
with precedence constraints and flexible machine assignments.
"""

import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional
import copy
from dataclasses import dataclass

from src.problems.problem_instance import ProblemInstance
from src.algorithms.decoder import decode_solution


@dataclass
class SAParams:
    """Parameters for Simulated Annealing."""
    initial_temp: float = 100.0
    final_temp: float = 0.01
    cooling_rate: float = 0.95
    max_iterations: int = 10000
    max_stagnation: int = 1000
    neighborhood_size: int = 50
    reheat_factor: float = 1.5
    acceptance_criterion: str = "metropolis"  # "metropolis" or "threshold"


class POFJSPSolution:
    """Represents a solution for POFJSP."""
    
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.chromosome = []  # List of (job_id, operation_id, machine_id)
        self.makespan = float('inf')
        self.machine_schedules = {}
        self.job_schedules = {}
    
    def initialize_random(self) -> None:
        """Initialize with random valid solution."""
        operations = []
        for job_id in range(self.problem.num_jobs):
            for op_id in range(self.problem.num_operations_per_job[job_id]):
                operations.append((job_id, op_id))
        
        # Generate valid sequence respecting precedence constraints
        valid_sequence = self._generate_valid_sequence(operations)
        
        # Assign random valid machines
        chromosome = []
        for job_id, op_id in valid_sequence:
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
        """Generate valid sequence respecting precedence constraints."""
        sequence = []
        remaining = set(operations)
        
        while remaining:
            available = []
            for job_id, op_id in remaining:
                op_key = f"({job_id}, {op_id})"
                predecessors = self.problem.predecessors_map.get(op_key, [])
                
                all_scheduled = True
                for pred in predecessors:
                    pred_job, pred_op = map(int, pred.strip('()').split(','))
                    if (pred_job, pred_op) not in sequence:
                        all_scheduled = False
                        break
                
                if all_scheduled:
                    available.append((job_id, op_id))
            
            if not available:
                break
                
            selected = random.choice(available)
            sequence.append(selected)
            remaining.remove(selected)
        
        return sequence
    
    def evaluate(self) -> None:
        """Evaluate the solution's makespan."""
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
        except Exception as e:
            self.makespan = float('inf')
    
    def copy(self) -> 'POFJSPSolution':
        """Create a deep copy of the solution."""
        new_solution = POFJSPSolution(self.problem)
        new_solution.chromosome = copy.deepcopy(self.chromosome)
        new_solution.makespan = self.makespan
        return new_solution


class SimulatedAnnealing:
    """Simulated Annealing for POFJSP."""
    
    def __init__(self, problem: ProblemInstance, params: SAParams):
        self.problem = problem
        self.params = params
        self.current_solution = None
        self.best_solution = None
        self.current_temp = params.initial_temp
        self.iteration = 0
        self.stagnation_count = 0
        self.history = []
    
    def initialize_solution(self) -> None:
        """Initialize the starting solution."""
        self.current_solution = POFJSPSolution(self.problem)
        self.current_solution.initialize_random()
        self.best_solution = self.current_solution.copy()
    
    def generate_neighbor(self, solution: POFJSPSolution) -> POFJSPSolution:
        """Generate a neighboring solution."""
        neighbor = solution.copy()
        
        # Choose neighborhood operator
        operator = random.choice(['sequence_swap', 'machine_change', 'sequence_shift'])
        
        if operator == 'sequence_swap':
            self._sequence_swap(neighbor)
        elif operator == 'machine_change':
            self._machine_change(neighbor)
        elif operator == 'sequence_shift':
            self._sequence_shift(neighbor)
        
        neighbor.evaluate()
        return neighbor
    
    def _sequence_swap(self, solution: POFJSPSolution) -> None:
        """Swap two operations in sequence respecting precedence."""
        chromosome = solution.chromosome
        if len(chromosome) < 2:
            return
        
        # Find valid swaps
        valid_swaps = []
        for i in range(len(chromosome) - 1):
            for j in range(i + 1, len(chromosome)):
                temp_chrom = chromosome[:]
                temp_chrom[i], temp_chrom[j] = temp_chrom[j], temp_chrom[i]
                
                if self._is_valid_sequence(temp_chrom):
                    valid_swaps.append((i, j))
        
        if valid_swaps:
            i, j = random.choice(valid_swaps)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    
    def _machine_change(self, solution: POFJSPSolution) -> None:
        """Change machine assignment for an operation."""
        chromosome = solution.chromosome
        if not chromosome:
            return
        
        # Select random operation
        idx = random.randint(0, len(chromosome) - 1)
        job_id, op_id, old_machine = chromosome[idx]
        
        # Get valid machines
        valid_machines = []
        for machine_id in range(self.problem.num_machines):
            if self.problem.processing_times[job_id][op_id][machine_id] != float('inf'):
                valid_machines.append(machine_id)
        
        if len(valid_machines) > 1:
            valid_machines.remove(old_machine)
            new_machine = random.choice(valid_machines)
            chromosome[idx] = (job_id, op_id, new_machine)
    
    def _sequence_shift(self, solution: POFJSPSolution) -> None:
        """Shift an operation to a new position respecting precedence."""
        chromosome = solution.chromosome
        if len(chromosome) < 2:
            return
        
        # Select operation to shift
        from_idx = random.randint(0, len(chromosome) - 1)
        to_idx = random.randint(0, len(chromosome) - 1)
        
        if from_idx == to_idx:
            return
        
        # Create new sequence
        new_chromosome = chromosome[:]
        operation = new_chromosome.pop(from_idx)
        new_chromosome.insert(to_idx, operation)
        
        # Check validity
        if self._is_valid_sequence(new_chromosome):
            solution.chromosome = new_chromosome
    
    def _is_valid_sequence(self, chromosome: List[Tuple[int, int, int]]) -> bool:
        """Check if chromosome represents valid sequence."""
        sequence = [(chrom[0], chrom[1]) for chrom in chromosome]
        
        for idx, (job_id, op_id) in enumerate(sequence):
            op_key = f"({job_id}, {op_id})"
            predecessors = self.problem.predecessors_map.get(op_key, [])
            
            for pred in predecessors:
                pred_job, pred_op = map(int, pred.strip('()').split(','))
                if (pred_job, pred_op) not in sequence[:idx]:
                    return False
        
        return True
    
    def acceptance_probability(self, old_cost: float, new_cost: float) -> float:
        """Calculate acceptance probability."""
        if new_cost < old_cost:
            return 1.0
        
        if self.params.acceptance_criterion == "metropolis":
            return math.exp((old_cost - new_cost) / self.current_temp)
        else:  # threshold
            return 1.0 if new_cost - old_cost < self.current_temp else 0.0
    
    def cool_temperature(self) -> None:
        """Cool the temperature."""
        self.current_temp *= self.params.cooling_rate
    
    def reheat(self) -> None:
        """Reheat the temperature."""
        self.current_temp *= self.params.reheat_factor
    
    def run(self, verbose: bool = False) -> POFJSPSolution:
        """Run the simulated annealing algorithm."""
        self.initialize_solution()
        
        if verbose:
            print(f"Initial makespan: {self.current_solution.makespan:.2f}")
            print(f"Initial temperature: {self.current_temp:.2f}")
        
        while self.iteration < self.params.max_iterations and self.current_temp > self.params.final_temp:
            # Generate neighbor
            neighbor = self.generate_neighbor(self.current_solution)
            
            # Calculate acceptance
            acceptance_prob = self.acceptance_probability(
                self.current_solution.makespan,
                neighbor.makespan
            )
            
            # Accept or reject
            if random.random() < acceptance_prob:
                self.current_solution = neighbor
                
                # Update best solution
                if neighbor.makespan < self.best_solution.makespan:
                    self.best_solution = neighbor.copy()
                    self.stagnation_count = 0
                else:
                    self.stagnation_count += 1
            else:
                self.stagnation_count += 1
            
            # Track history
            self.history.append({
                'iteration': self.iteration,
                'temperature': self.current_temp,
                'current_makespan': self.current_solution.makespan,
                'best_makespan': self.best_solution.makespan
            })
            
            # Reheat if stagnated
            if self.stagnation_count >= self.params.max_stagnation:
                self.reheat()
                self.stagnation_count = 0
                if verbose:
                    print(f"Reheating at iteration {self.iteration}")
            
            # Cool temperature
            self.cool_temperature()
            self.iteration += 1
            
            if verbose and self.iteration % 1000 == 0:
                print(f"Iteration {self.iteration}: T={self.current_temp:.2f}, "
                      f"Best={self.best_solution.makespan:.2f}")
        
        if verbose:
            print(f"Final makespan: {self.best_solution.makespan:.2f}")
            print(f"Total iterations: {self.iteration}")
        
        return self.best_solution
    
    def get_statistics(self) -> Dict:
        """Get algorithm statistics."""
        return {
            'iterations': self.iteration,
            'final_temperature': self.current_temp,
            'best_makespan': self.best_solution.makespan,
            'temperature_history': [h['temperature'] for h in self.history],
            'makespan_history': [h['best_makespan'] for h in self.history]
        }


def solve_with_sa(problem: ProblemInstance, params: SAParams = None, verbose: bool = False) -> Dict:
    """Solve POFJSP with Simulated Annealing."""
    if params is None:
        params = SAParams()
    
    sa = SimulatedAnnealing(problem, params)
    best_solution = sa.run(verbose=verbose)
    
    # Convert to Solution format
    from src.problems.problem_instance import Solution
    
    operation_sequence = []
    machine_assignment = []
    for job_id, op_id, machine_id in best_solution.chromosome:
        operation_sequence.append((job_id, op_id))
        machine_assignment.append(machine_id)
    
    solution = Solution(operation_sequence, machine_assignment)
    makespan, _, _ = decode_solution(solution, problem)
    
    return {
        'solution': solution,
        'makespan': makespan,
        'statistics': sa.get_statistics()
    }


def main():
    """Main function for testing Simulated Annealing."""
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
    
    # Run Simulated Annealing
    print("\nRunning Simulated Annealing...")
    params = SAParams(
        initial_temp=50.0,  # Small params for testing
        max_iterations=100,
        cooling_rate=0.95
    )
    
    result = solve_with_sa(problem, params, verbose=True)
    
    if result['solution']:
        print(f"\n[SUCCESS] Algorithm completed successfully!")
        print(f"Final makespan: {result['makespan']}")
        print(f"Statistics: {result['statistics']}")
    else:
        print("[FAILED] Algorithm failed to find a solution")


if __name__ == "__main__":
    main()