#!/usr/bin/env python3
"""
Comprehensive Algorithm Execution Tests

These tests ensure all algorithms actually execute successfully and produce valid results,
catching runtime bugs that the basic interface tests might miss.
"""

import unittest
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.algorithms.factory import AlgorithmFactory
from src.problems.problem_instance import ProblemInstance


class TestAlgorithmExecution(unittest.TestCase):
    """Test that all algorithms execute successfully on real problems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = AlgorithmFactory()
        
        # Create a simple test problem
        self.test_problem_data = {
            "num_jobs": 3,
            "num_machines": 3,
            "jobs": [
                {
                    "id": 0,
                    "operations": [
                        {"id": 0, "processing_times": [3, 2, 4], "precedence": []},
                        {"id": 1, "processing_times": [2, 4, 1], "precedence": [0]}
                    ]
                },
                {
                    "id": 1,
                    "operations": [
                        {"id": 2, "processing_times": [1, 3, 2], "precedence": []},
                        {"id": 3, "processing_times": [4, 1, 3], "precedence": [2]}
                    ]
                },
                {
                    "id": 2,
                    "operations": [
                        {"id": 4, "processing_times": [2, 1, 3], "precedence": []},
                        {"id": 5, "processing_times": [3, 2, 1], "precedence": [4]}
                    ]
                }
            ]
        }
        
        self.problem = ProblemInstance.from_dict(self.test_problem_data)
        
        # Algorithm parameters for testing (fast execution)
        self.test_params = {
            'genetic': {'pop_size': 5, 'generations': 3},
            'simulated_annealing': {'max_iterations': 10, 'initial_temp': 100.0},
            'aco': {'max_iterations': 5, 'n_ants': 3},
            'pso': {'max_iterations': 5, 'n_particles': 3},
            'differential_evolution': {'max_iterations': 5, 'population_size': 3},
            'vns': {'max_iterations': 5},
            'tabu_search': {'max_iterations': 5},
            'hybrid_ga_ls': {'population_size': 3, 'max_generations': 2},
            'memory_sa': {'max_iterations': 10, 'memory_size': 2},
            'iaoa_gns': {},
            'random': {},
            'greedy': {}
        }
    
    def test_all_algorithms_execute_successfully(self):
        """Test that all algorithms execute without crashing."""
        available_algorithms = self.factory.get_available_algorithms()
        failed_algorithms = []
        
        for algo_name in available_algorithms.keys():
            with self.subTest(algorithm=algo_name):
                try:
                    # Get appropriate parameters
                    params = self.test_params.get(algo_name, {})
                    
                    # Create algorithm
                    algorithm = self.factory.create_algorithm(algo_name, params)
                    
                    # Run algorithm with short timeout
                    result = algorithm.solve(self.problem, timeout=30.0)
                    
                    # Verify result is valid
                    self.assertIsNotNone(result, f"Algorithm {algo_name} returned None")
                    self.assertIsNotNone(result.makespan, f"Algorithm {algo_name} returned None makespan")
                    self.assertIsInstance(result.execution_time, float, 
                                        f"Algorithm {algo_name} returned invalid execution time")
                    
                    # Verify makespan is finite (not infinity)
                    if result.makespan == float('inf'):
                        failed_algorithms.append(f"{algo_name}: infinite makespan")
                    
                    # Verify makespan is positive
                    elif result.makespan <= 0:
                        failed_algorithms.append(f"{algo_name}: non-positive makespan {result.makespan}")
                    
                    # Verify execution time is reasonable
                    elif result.execution_time < 0:
                        failed_algorithms.append(f"{algo_name}: negative execution time {result.execution_time}")
                    
                    print(f"✅ {algo_name}: makespan={result.makespan:.2f}, time={result.execution_time:.3f}s")
                    
                except Exception as e:
                    failed_algorithms.append(f"{algo_name}: {str(e)}")
                    print(f"❌ {algo_name}: {str(e)}")
        
        # Report any failures
        if failed_algorithms:
            self.fail(f"The following algorithms failed execution:\n" + "\n".join(failed_algorithms))
    
    def test_algorithm_result_consistency(self):
        """Test that algorithms produce consistent results with same parameters."""
        test_algorithms = ['genetic', 'simulated_annealing', 'aco']
        
        for algo_name in test_algorithms:
            with self.subTest(algorithm=algo_name):
                params = self.test_params.get(algo_name, {})
                
                # Run algorithm twice
                algorithm1 = self.factory.create_algorithm(algo_name, params)
                algorithm2 = self.factory.create_algorithm(algo_name, params)
                
                result1 = algorithm1.solve(self.problem, timeout=30.0)
                result2 = algorithm2.solve(self.problem, timeout=30.0)
                
                # Both should produce finite results
                self.assertNotEqual(result1.makespan, float('inf'), 
                                  f"Algorithm {algo_name} run 1 produced infinite makespan")
                self.assertNotEqual(result2.makespan, float('inf'), 
                                  f"Algorithm {algo_name} run 2 produced infinite makespan")
                
                # Both should be positive
                self.assertGreater(result1.makespan, 0, 
                                 f"Algorithm {algo_name} run 1 produced non-positive makespan")
                self.assertGreater(result2.makespan, 0, 
                                 f"Algorithm {algo_name} run 2 produced non-positive makespan")
    
    def test_algorithm_scalability(self):
        """Test algorithms on slightly larger problem to check scalability."""
        # Create a larger test problem
        larger_problem_data = {
            "num_jobs": 4,
            "num_machines": 4,
            "jobs": []
        }
        
        for job_id in range(4):
            job = {
                "id": job_id,
                "operations": []
            }
            for op_id in range(2):
                operation = {
                    "id": job_id * 2 + op_id,
                    "processing_times": [1 + (job_id + op_id) % 3, 2 + (job_id + op_id) % 2, 
                                       3 + op_id % 2, 1 + job_id % 3],
                    "precedence": [job_id * 2 + op_id - 1] if op_id > 0 else []
                }
                job["operations"].append(operation)
            larger_problem_data["jobs"].append(job)
        
        larger_problem = ProblemInstance.from_dict(larger_problem_data)
        
        # Test a few key algorithms on larger problem
        key_algorithms = ['genetic', 'simulated_annealing', 'aco', 'vns']
        
        for algo_name in key_algorithms:
            with self.subTest(algorithm=algo_name, problem_size="4x4"):
                params = self.test_params.get(algo_name, {})
                algorithm = self.factory.create_algorithm(algo_name, params)
                
                result = algorithm.solve(larger_problem, timeout=60.0)
                
                self.assertIsNotNone(result)
                self.assertNotEqual(result.makespan, float('inf'), 
                                  f"Algorithm {algo_name} failed on 4x4 problem")
                self.assertGreater(result.makespan, 0, 
                                 f"Algorithm {algo_name} produced invalid makespan on 4x4 problem")
                
                print(f"✅ {algo_name} (4x4): makespan={result.makespan:.2f}, time={result.execution_time:.3f}s")
    
    def test_algorithm_timeout_handling(self):
        """Test that algorithms respect timeout parameters."""
        # Test with very short timeout
        algorithm = self.factory.create_algorithm('vns', {'max_iterations': 1000})  # Potentially long-running
        
        start_time = time.time()
        result = algorithm.solve(self.problem, timeout=0.1)  # Very short timeout
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Should not take much longer than timeout (allowing some overhead)
        self.assertLess(elapsed, 1.0, "Algorithm did not respect timeout")
        self.assertIsNotNone(result, "Algorithm should return a result even when timing out")
        
    def test_algorithm_parameter_validation(self):
        """Test that algorithms handle invalid parameters gracefully."""
        # Test with invalid parameters
        invalid_params_tests = [
            ('genetic', {'pop_size': -1}),  # Negative population size
            ('simulated_annealing', {'initial_temp': -100.0}),  # Negative temperature
            ('aco', {'n_ants': 0}),  # Zero ants
            ('pso', {'n_particles': -5}),  # Negative particles
        ]
        
        for algo_name, invalid_params in invalid_params_tests:
            with self.subTest(algorithm=algo_name, params=invalid_params):
                with self.assertRaises((ValueError, TypeError, AssertionError), 
                                     msg=f"Algorithm {algo_name} should reject invalid parameters {invalid_params}"):
                    algorithm = self.factory.create_algorithm(algo_name, invalid_params)
                    algorithm.solve(self.problem, timeout=30.0)
    
    def test_dispatching_rules_execution(self):
        """Test all dispatching rules execute successfully."""
        dispatching_rules = ['SPT', 'LPT', 'EST', 'LST', 'FIFO']
        
        for rule in dispatching_rules:
            with self.subTest(rule=rule):
                algorithm = self.factory.create_algorithm('dispatching', {'rule': rule})
                result = algorithm.solve(self.problem, timeout=30.0)
                
                self.assertIsNotNone(result)
                self.assertNotEqual(result.makespan, float('inf'), 
                                  f"Dispatching rule {rule} produced infinite makespan")
                self.assertGreater(result.makespan, 0, 
                                 f"Dispatching rule {rule} produced invalid makespan")
                
                print(f"✅ Dispatching-{rule}: makespan={result.makespan:.2f}, time={result.execution_time:.3f}s")


if __name__ == '__main__':
    # Configure test output
    unittest.main(verbosity=2, buffer=True)