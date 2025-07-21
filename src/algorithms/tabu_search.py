"""
Tabu Search for POFJSP (Partially Ordered Flexible Job Shop Scheduling)

This module implements a Tabu Search algorithm specifically designed for POFJSP
with precedence constraints and flexible machine assignments.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Set, Optional
import copy
from dataclasses import dataclass
from collections import deque

from src.problems.problem_instance import ProblemInstance
from src.algorithms.decoder import decode_solution


@dataclass
class TabuParams:
    """Parameters for Tabu Search."""
    max_iterations: int = 1000
    tabu_list_size: int = 50
    max_stagnation: int = 100
    neighborhood_size: int = 30
    aspiration_criterion: bool = True
    intensification_threshold: int = 20
    diversification_threshold: int = 50


class TabuMove:
    """Represents a tabu move (operation swap or machine change)."""
    
    def __init__(self, move_type: str, operation1: Tuple[int, int], operation2: Tuple[int, int] = None, 
                 old_machine: int = None, new_machine: int = None):
        self.move_type = move_type  # 'swap', 'machine', 'shift'
        self.operation1 = operation1
        self.operation2 = operation2
        self.old_machine = old_machine
        self.new_machine = new_machine
    
    def __eq__(self, other):
        if not isinstance(other, TabuMove):
            return False
        return (self.move_type == other.move_type and
                self.operation1 == other.operation1 and
                self.operation2 == other.operation2 and
                self.old_machine == other.old_machine and
                self.new_machine == other.new_machine)
    
    def __hash__(self):
        return hash((self.move_type, self.operation1, self.operation2, 
                    self.old_machine, self.new_machine))


class POFJSPTabuSolution:
    """Represents a solution for POFJSP Tabu Search."""
    
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
    
    def copy(self) -> 'POFJSPTabuSolution':
        """Create a deep copy of the solution."""
        new_solution = POFJSPTabuSolution(self.problem)
        new_solution.chromosome = copy.deepcopy(self.chromosome)
        new_solution.makespan = self.makespan
        return new_solution
    
    def get_operations_sequence(self) -> List[Tuple[int, int]]:
        """Get the sequence of operations (without machines)."""
        return [(chrom[0], chrom[1]) for chrom in self.chromosome]
    
    def set_operations_sequence(self, sequence: List[Tuple[int, int]]) -> None:
        """Set the sequence of operations while keeping machine assignments."""
        new_chromosome = []
        
        # Map old operations to machine assignments
        machine_map = {(chrom[0], chrom[1]): chrom[2] for chrom in self.chromosome}
        
        for job_id, op_id in sequence:
            machine_id = machine_map.get((job_id, op_id), 0)
            new_chromosome.append((job_id, op_id, machine_id))
        
        self.chromosome = new_chromosome
        self.evaluate()


class TabuSearch:
    """Tabu Search for POFJSP."""
    
    def __init__(self, problem: ProblemInstance, params: TabuParams):
        self.problem = problem
        self.params = params
        self.current_solution = None
        self.best_solution = None
        self.tabu_list = deque(maxlen=params.tabu_list_size)
        self.iteration = 0
        self.stagnation_count = 0
        self.history = []
        self.frequency_matrix = {}  # For diversification
    
    def initialize_solution(self) -> None:
        """Initialize the starting solution."""
        self.current_solution = POFJSPTabuSolution(self.problem)
        self.current_solution.initialize_random()
        self.best_solution = self.current_solution.copy()
    
    def generate_neighbors(self, solution: POFJSPTabuSolution) -> List[Tuple[POFJSPTabuSolution, TabuMove]]:
        """Generate neighboring solutions."""
        neighbors = []
        
        # Generate sequence neighbors
        sequence_neighbors = self._generate_sequence_neighbors(solution)
        neighbors.extend(sequence_neighbors)
        
        # Generate machine neighbors
        machine_neighbors = self._generate_machine_neighbors(solution)
        neighbors.extend(machine_neighbors)
        
        # Limit neighborhood size
        if len(neighbors) > self.params.neighborhood_size:
            neighbors = random.sample(neighbors, self.params.neighborhood_size)
        
        return neighbors
    
    def _generate_sequence_neighbors(self, solution: POFJSPTabuSolution) -> List[Tuple[POFJSPTabuSolution, TabuMove]]:
        """Generate neighbors by swapping operations."""
        neighbors = []
        chromosome = solution.chromosome
        
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                # Create new sequence
                new_sequence = solution.get_operations_sequence()[:]
                new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
                
                # Check if valid
                if self._is_valid_sequence(new_sequence):
                    # Create move
                    move = TabuMove(
                        'swap',
                        (chromosome[i][0], chromosome[i][1]),
                        (chromosome[j][0], chromosome[j][1])
                    )
                    
                    # Create new solution
                    new_solution = solution.copy()
                    new_solution.set_operations_sequence(new_sequence)
                    neighbors.append((new_solution, move))
        
        return neighbors
    
    def _generate_machine_neighbors(self, solution: POFJSPTabuSolution) -> List[Tuple[POFJSPTabuSolution, TabuMove]]:
        """Generate neighbors by changing machine assignments."""
        neighbors = []
        chromosome = solution.chromosome
        
        for i, (job_id, op_id, old_machine) in enumerate(chromosome):
            # Get valid machines
            valid_machines = []
            for machine_id in range(self.problem.num_machines):
                if self.problem.processing_times[job_id][op_id][machine_id] != float('inf'):
                    valid_machines.append(machine_id)
            
            # Try different machines
            for new_machine in valid_machines:
                if new_machine != old_machine:
                    new_chromosome = chromosome[:]
                    new_chromosome[i] = (job_id, op_id, new_machine)
                    
                    # Create move
                    move = TabuMove(
                        'machine',
                        (job_id, op_id),
                        old_machine=old_machine,
                        new_machine=new_machine
                    )
                    
                    # Create new solution
                    new_solution = solution.copy()
                    new_solution.chromosome = new_chromosome
                    new_solution.evaluate()
                    neighbors.append((new_solution, move))
        
        return neighbors
    
    def _is_valid_sequence(self, sequence: List[Tuple[int, int]]) -> bool:
        """Check if sequence respects precedence constraints."""
        for idx, (job_id, op_id) in enumerate(sequence):
            op_key = f"({job_id}, {op_id})"
            predecessors = self.problem.predecessors_map.get(op_key, [])
            
            for pred in predecessors:
                pred_job, pred_op = map(int, pred.strip('()').split(','))
                if (pred_job, pred_op) not in sequence[:idx]:
                    return False
        
        return True
    
    def is_tabu(self, move: TabuMove) -> bool:
        """Check if move is tabu."""
        return move in self.tabu_list
    
    def add_to_tabu(self, move: TabuMove) -> None:
        """Add move to tabu list."""
        self.tabu_list.append(move)
    
    def aspiration_criterion(self, move: TabuMove, new_solution: POFJSPTabuSolution) -> bool:
        """Check if tabu move should be accepted via aspiration criterion."""
        if not self.params.aspiration_criterion:
            return False
        
        return new_solution.makespan < self.best_solution.makespan
    
    def update_frequency_matrix(self, solution: POFJSPTabuSolution) -> None:
        """Update frequency matrix for diversification."""
        for i, (job_id, op_id, machine_id) in enumerate(solution.chromosome):
            key = ((job_id, op_id), machine_id)
            self.frequency_matrix[key] = self.frequency_matrix.get(key, 0) + 1
    
    def diversification_penalty(self, solution: POFJSPTabuSolution) -> float:
        """Calculate diversification penalty based on frequency."""
        penalty = 0.0
        for i, (job_id, op_id, machine_id) in enumerate(solution.chromosome):
            key = ((job_id, op_id), machine_id)
            penalty += self.frequency_matrix.get(key, 0)
        return penalty * 0.1  # Small penalty factor
    
    def intensification_phase(self) -> None:
        """Perform intensification - focus search around best solution."""
        # Restart from best solution with reduced neighborhood
        self.current_solution = self.best_solution.copy()
    
    def diversification_phase(self) -> None:
        """Perform diversification - explore new regions."""
        # Create new random solution with penalty consideration
        new_solution = POFJSPTabuSolution(self.problem)
        new_solution.initialize_random()
        self.current_solution = new_solution
    
    def run(self, verbose: bool = False) -> POFJSPTabuSolution:
        """Run the tabu search algorithm."""
        self.initialize_solution()
        
        if verbose:
            print(f"Initial makespan: {self.current_solution.makespan:.2f}")
        
        while self.iteration < self.params.max_iterations:
            # Generate neighbors
            neighbors = self.generate_neighbors(self.current_solution)
            
            # Find best non-tabu neighbor
            best_neighbor = None
            best_move = None
            best_makespan = float('inf')
            
            for neighbor, move in neighbors:
                # Check if move is tabu
                is_tabu_move = self.is_tabu(move)
                
                # Apply aspiration criterion
                if is_tabu_move and not self.aspiration_criterion(move, neighbor):
                    continue
                
                # Consider diversification penalty
                total_cost = neighbor.makespan + self.diversification_penalty(neighbor)
                
                if total_cost < best_makespan:
                    best_makespan = total_cost
                    best_neighbor = neighbor
                    best_move = move
            
            if best_neighbor is None:
                # No valid moves found
                break
            
            # Update current solution
            self.current_solution = best_neighbor
            
            # Add move to tabu list
            self.add_to_tabu(best_move)
            
            # Update best solution
            if best_neighbor.makespan < self.best_solution.makespan:
                self.best_solution = best_neighbor.copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            
            # Update frequency matrix
            self.update_frequency_matrix(best_neighbor)
            
            # Track history
            self.history.append({
                'iteration': self.iteration,
                'current_makespan': self.current_solution.makespan,
                'best_makespan': self.best_solution.makespan,
                'tabu_list_size': len(self.tabu_list)
            })
            
            # Check for intensification/diversification
            if self.stagnation_count >= self.params.intensification_threshold:
                if verbose:
                    print(f"Intensification at iteration {self.iteration}")
                self.intensification_phase()
                self.stagnation_count = 0
            
            if self.stagnation_count >= self.params.diversification_threshold:
                if verbose:
                    print(f"Diversification at iteration {self.iteration}")
                self.diversification_phase()
                self.stagnation_count = 0
            
            self.iteration += 1
            
            if verbose and self.iteration % 100 == 0:
                print(f"Iteration {self.iteration}: Best={self.best_solution.makespan:.2f}")
        
        if verbose:
            print(f"Final makespan: {self.best_solution.makespan:.2f}")
            print(f"Total iterations: {self.iteration}")
        
        return self.best_solution
    
    def get_statistics(self) -> Dict:
        """Get algorithm statistics."""
        return {
            'iterations': self.iteration,
            'best_makespan': self.best_solution.makespan,
            'tabu_list_size': len(self.tabu_list),
            'history': self.history,
            'frequency_matrix_size': len(self.frequency_matrix)
        }


def solve_with_tabu(problem: ProblemInstance, params: TabuParams = None, verbose: bool = False) -> Dict:
    """Solve POFJSP with Tabu Search."""
    if params is None:
        params = TabuParams()
    
    ts = TabuSearch(problem, params)
    best_solution = ts.run(verbose=verbose)
    
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
        'statistics': ts.get_statistics()
    }


def main():
    """Main function for testing Tabu Search."""
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
    
    # Run Tabu Search
    print("\nRunning Tabu Search...")
    params = TabuParams(
        max_iterations=100,  # Small params for testing
        tabu_list_size=10,
        neighborhood_size=15
    )
    
    result = solve_with_tabu(problem, params, verbose=True)
    
    if result['solution']:
        print(f"\n[SUCCESS] Algorithm completed successfully!")
        print(f"Final makespan: {result['makespan']}")
        print(f"Statistics: {result['statistics']}")
    else:
        print("[FAILED] Algorithm failed to find a solution")


if __name__ == "__main__":
    main()