"""
Unit tests for POFJSP algorithms.

Tests cover:
- Problem instance creation and validation
- Solution encoding/decoding
- Algorithm components (crossover, mutation, GNS)
- Full algorithm execution
"""

import unittest
import numpy as np
from collections import namedtuple
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms import (
    ProblemInstance, Solution, Operation,
    decode_solution, iaoa_gns_algorithm,
    get_topological_sort_operations,
    initialize_population,
    two_d_clustering_crossover,
    effective_parallel_mutation,
    grade_neighborhood_search
)


class TestProblemInstance(unittest.TestCase):
    """Test problem instance creation and validation."""
    
    def setUp(self):
        """Create a simple test problem instance."""
        self.num_jobs = 2
        self.num_machines = 2
        self.num_operations_per_job = [2, 2]
        
        # Processing times: Job 0: [[3,5], [6,inf]], Job 1: [[4,2], [inf,7]]
        self.processing_times = [
            np.array([[3, 5], [6, np.inf]]),
            np.array([[4, 2], [np.inf, 7]])
        ]
        
        # Simple precedence: each job's second operation follows the first
        self.predecessors_map = {
            Operation(0, 0): set(),
            Operation(0, 1): {Operation(0, 0)},
            Operation(1, 0): set(),
            Operation(1, 1): {Operation(1, 0)}
        }
        
        self.successors_map = {
            Operation(0, 0): {Operation(0, 1)},
            Operation(0, 1): set(),
            Operation(1, 0): {Operation(1, 1)},
            Operation(1, 1): set()
        }
        
        self.problem = ProblemInstance(
            self.num_jobs, self.num_machines, self.num_operations_per_job,
            self.processing_times, self.predecessors_map, self.successors_map
        )
    
    def test_problem_creation(self):
        """Test basic problem instance creation."""
        self.assertEqual(self.problem.num_jobs, 2)
        self.assertEqual(self.problem.num_machines, 2)
        self.assertEqual(self.problem.total_operations, 4)
        self.assertEqual(len(self.problem.all_operations), 4)
    
    def test_processing_times_format(self):
        """Test processing times are correctly formatted."""
        # Check dimensions
        self.assertEqual(len(self.problem.processing_times), self.num_jobs)
        for j in range(self.num_jobs):
            expected_shape = (self.num_operations_per_job[j], self.num_machines)
            self.assertEqual(self.problem.processing_times[j].shape, expected_shape)
    
    def test_precedence_constraints(self):
        """Test precedence constraints are valid."""
        # All operations should be in both maps
        all_ops = set(self.problem.all_operations)
        self.assertEqual(set(self.problem.predecessors_map.keys()), all_ops)
        self.assertEqual(set(self.problem.successors_map.keys()), all_ops)
        
        # Check consistency between predecessor and successor maps
        for op, predecessors in self.problem.predecessors_map.items():
            for pred_op in predecessors:
                self.assertIn(op, self.problem.successors_map[pred_op])


class TestSolutionDecoding(unittest.TestCase):
    """Test solution representation and decoding."""
    
    def setUp(self):
        """Set up test problem and solution."""
        # Reuse problem from TestProblemInstance
        self.problem = TestProblemInstance().setUp.__wrapped__(TestProblemInstance())
        if hasattr(TestProblemInstance(), 'problem'):
            self.problem = TestProblemInstance().problem
        else:
            # Recreate problem manually
            num_jobs = 2
            num_machines = 2
            num_operations_per_job = [2, 2]
            processing_times = [
                np.array([[3, 5], [6, np.inf]]),
                np.array([[4, 2], [np.inf, 7]])
            ]
            predecessors_map = {
                Operation(0, 0): set(),
                Operation(0, 1): {Operation(0, 0)},
                Operation(1, 0): set(),
                Operation(1, 1): {Operation(1, 0)}
            }
            successors_map = {
                Operation(0, 0): {Operation(0, 1)},
                Operation(0, 1): set(),
                Operation(1, 0): {Operation(1, 1)},
                Operation(1, 1): set()
            }
            self.problem = ProblemInstance(
                num_jobs, num_machines, num_operations_per_job,
                processing_times, predecessors_map, successors_map
            )
        
        # Create a valid solution
        self.operation_sequence = [
            Operation(0, 0), Operation(1, 0), 
            Operation(0, 1), Operation(1, 1)
        ]
        self.machine_assignment = [0, 1, 0, 1]  # Valid machines for each op
        
        self.solution = Solution(self.operation_sequence, self.machine_assignment)
    
    def test_solution_creation(self):
        """Test solution object creation."""
        self.assertEqual(len(self.solution.operation_sequence), 4)
        self.assertEqual(len(self.solution.machine_assignment), 4)
        self.assertEqual(self.solution.makespan, float('inf'))  # Before decoding
    
    def test_solution_decoding(self):
        """Test solution decoding produces valid schedule."""
        makespan, schedule_details, machine_schedules = decode_solution(self.solution, self.problem)
        
        # Check that makespan is finite and positive
        self.assertTrue(np.isfinite(makespan))
        self.assertGreater(makespan, 0)
        
        # Check that all operations are scheduled
        self.assertEqual(len(schedule_details), len(self.operation_sequence))
        
        # Check that solution object is updated
        self.assertEqual(self.solution.makespan, makespan)
        self.assertEqual(len(self.solution.schedule_details), 4)
    
    def test_precedence_constraints_respected(self):
        """Test that decoding respects precedence constraints."""
        decode_solution(self.solution, self.problem)
        
        # Check precedence constraints
        for op, details in self.solution.schedule_details.items():
            if op in self.problem.predecessors_map:
                for pred_op in self.problem.predecessors_map[op]:
                    if pred_op in self.solution.schedule_details:
                        pred_end = self.solution.schedule_details[pred_op]['end_time']
                        op_start = details['start_time']
                        self.assertGreaterEqual(op_start, pred_end,
                            f"Operation {op} starts before predecessor {pred_op} ends")


class TestAlgorithmComponents(unittest.TestCase):
    """Test individual algorithm components."""
    
    def setUp(self):
        """Set up test problem for algorithm testing."""
        # Create a slightly larger problem for algorithm testing
        self.num_jobs = 3
        self.num_machines = 3
        self.num_operations_per_job = [3, 2, 2]
        
        # Simple processing times
        self.processing_times = [
            np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1]]),  # Job 0
            np.array([[2, 3, 1], [1, 2, 3]]),              # Job 1
            np.array([[3, 1, 2], [2, 3, 1]])               # Job 2
        ]
        
        # Simple sequential precedence within each job
        self.predecessors_map = {}
        self.successors_map = {}
        
        for j in range(self.num_jobs):
            for o in range(self.num_operations_per_job[j]):
                op = Operation(j, o)
                if o == 0:
                    self.predecessors_map[op] = set()
                else:
                    self.predecessors_map[op] = {Operation(j, o-1)}
                
                if o == self.num_operations_per_job[j] - 1:
                    self.successors_map[op] = set()
                else:
                    self.successors_map[op] = {Operation(j, o+1)}
        
        self.problem = ProblemInstance(
            self.num_jobs, self.num_machines, self.num_operations_per_job,
            self.processing_times, self.predecessors_map, self.successors_map
        )
    
    def test_topological_sort(self):
        """Test topological sort operation ordering."""
        topo_ops = get_topological_sort_operations(self.problem)
        
        # Should return all operations
        self.assertEqual(len(topo_ops), self.problem.total_operations)
        
        # Should respect precedence (simplified check)
        for i, op in enumerate(topo_ops):
            if op in self.problem.predecessors_map:
                for pred_op in self.problem.predecessors_map[op]:
                    pred_index = topo_ops.index(pred_op)
                    self.assertLess(pred_index, i, 
                        f"Predecessor {pred_op} appears after {op} in topological sort")
    
    def test_population_initialization(self):
        """Test population initialization."""
        pop_size = 10
        population = initialize_population(pop_size, self.problem)
        
        # Check population size
        self.assertEqual(len(population), pop_size)
        
        # Check each solution is valid
        for sol in population:
            self.assertIsInstance(sol, Solution)
            self.assertEqual(len(sol.operation_sequence), self.problem.total_operations)
            self.assertEqual(len(sol.machine_assignment), self.problem.total_operations)
            self.assertTrue(np.isfinite(sol.makespan))  # Should be decoded during initialization
    
    def test_mutation_operator(self):
        """Test effective parallel mutation."""
        # Create a solution
        population = initialize_population(1, self.problem)
        original_solution = population[0]
        
        # Apply mutation
        mutated_solution = effective_parallel_mutation(original_solution, self.problem)
        
        # Check that mutation produces a valid solution
        self.assertIsInstance(mutated_solution, Solution)
        self.assertEqual(len(mutated_solution.operation_sequence), self.problem.total_operations)
        self.assertTrue(np.isfinite(mutated_solution.makespan))
        
        # Solution should be different (most of the time)
        # Note: Due to randomness, this test might occasionally fail
        operations_changed = any(
            orig_op != mut_op for orig_op, mut_op in 
            zip(original_solution.operation_sequence, mutated_solution.operation_sequence)
        )
        machines_changed = any(
            orig_m != mut_m for orig_m, mut_m in 
            zip(original_solution.machine_assignment, mutated_solution.machine_assignment)
        )
        # At least one of the encodings should change
        self.assertTrue(operations_changed or machines_changed)


class TestFullAlgorithm(unittest.TestCase):
    """Test complete IAOA+GNS algorithm execution."""
    
    def setUp(self):
        """Set up small test problem for algorithm execution."""
        # Use a very small problem for quick testing
        self.num_jobs = 2
        self.num_machines = 2
        self.num_operations_per_job = [2, 2]
        
        self.processing_times = [
            np.array([[3, 5], [6, 4]]),
            np.array([[4, 2], [3, 7]])
        ]
        
        self.predecessors_map = {
            Operation(0, 0): set(),
            Operation(0, 1): {Operation(0, 0)},
            Operation(1, 0): set(),
            Operation(1, 1): {Operation(1, 0)}
        }
        
        self.successors_map = {
            Operation(0, 0): {Operation(0, 1)},
            Operation(0, 1): set(),
            Operation(1, 0): {Operation(1, 1)},
            Operation(1, 1): set()
        }
        
        self.problem = ProblemInstance(
            self.num_jobs, self.num_machines, self.num_operations_per_job,
            self.processing_times, self.predecessors_map, self.successors_map
        )
    
    def test_algorithm_execution(self):
        """Test full algorithm execution."""
        # Run algorithm with small parameters for quick test
        pop_size = 10
        max_iterations = 5
        
        best_solution = iaoa_gns_algorithm(self.problem, pop_size, max_iterations)
        
        # Check that algorithm returns a valid solution
        self.assertIsInstance(best_solution, Solution)
        self.assertTrue(np.isfinite(best_solution.makespan))
        self.assertGreater(best_solution.makespan, 0)
        
        # Check that solution has correct structure
        self.assertEqual(len(best_solution.operation_sequence), self.problem.total_operations)
        self.assertEqual(len(best_solution.machine_assignment), self.problem.total_operations)
        self.assertEqual(len(best_solution.schedule_details), self.problem.total_operations)
    
    def test_algorithm_convergence(self):
        """Test that algorithm shows improvement over iterations."""
        # Run algorithm multiple times to check for improvement potential
        pop_size = 20
        max_iterations = 10
        
        results = []
        for _ in range(3):  # Run 3 times
            solution = iaoa_gns_algorithm(self.problem, pop_size, max_iterations)
            results.append(solution.makespan)
        
        # All results should be finite
        for makespan in results:
            self.assertTrue(np.isfinite(makespan))
            self.assertGreater(makespan, 0)
        
        # At least one result should be reasonably good (less than naive upper bound)
        # Naive upper bound: sum of all minimum processing times
        min_total_time = 0
        for j in range(self.problem.num_jobs):
            for o in range(self.problem.num_operations_per_job[j]):
                min_time = np.min(self.problem.processing_times[j][o, :])
                if np.isfinite(min_time):
                    min_total_time += min_time
        
        best_result = min(results)
        # The best result should be better than just sequentially processing everything
        # This is a very loose bound, but helps catch major errors
        self.assertLess(best_result, min_total_time * 2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_single_job_single_machine(self):
        """Test trivial problem with one job and one machine."""
        problem = ProblemInstance(
            num_jobs=1,
            num_machines=1,
            num_operations_per_job=[1],
            processing_times=[np.array([[5]])],
            predecessors_map={Operation(0, 0): set()},
            successors_map={Operation(0, 0): set()}
        )
        
        # Algorithm should handle this correctly
        solution = iaoa_gns_algorithm(problem, pop_size=5, max_iterations=3)
        self.assertEqual(solution.makespan, 5)  # Should be exactly the processing time
    
    def test_infeasible_machine_assignment(self):
        """Test handling of infeasible machine assignments."""
        # Create problem where some operations can't be processed on some machines
        problem = ProblemInstance(
            num_jobs=1,
            num_machines=2,
            num_operations_per_job=[2],
            processing_times=[np.array([[3, np.inf], [np.inf, 4]])],
            predecessors_map={
                Operation(0, 0): set(),
                Operation(0, 1): {Operation(0, 0)}
            },
            successors_map={
                Operation(0, 0): {Operation(0, 1)},
                Operation(0, 1): set()
            }
        )
        
        # Algorithm should still find a valid solution
        solution = iaoa_gns_algorithm(problem, pop_size=5, max_iterations=3)
        self.assertTrue(np.isfinite(solution.makespan))
        self.assertEqual(solution.makespan, 7)  # 3 + 4


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 