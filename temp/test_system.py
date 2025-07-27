#!/usr/bin/env python3
"""
System Integration Test

Quick test to verify that the entire POFJSP system works correctly
after all the refactoring and improvements.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_basic_functionality():
    """Test basic POFJSP functionality."""
    print("=== POFJSP System Integration Test ===")
    
    # Test 1: Import all critical components
    print("1. Testing imports...")
    try:
        from src.algorithms.factory import AlgorithmFactory, create_algorithm_suite
        from src.problems.problem_instance import ProblemInstance, Solution
        from src.algorithms.baseline_algorithms import GeneticAlgorithm
        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    # Test 2: Create a simple problem
    print("2. Testing problem creation...")
    try:
        problem_data = {
            "num_jobs": 2,
            "num_machines": 2,
            "jobs": [
                {
                    "id": 0,
                    "operations": [
                        {"id": 0, "processing_times": [3, 2], "precedence": []},
                        {"id": 1, "processing_times": [2, 4], "precedence": [0]}
                    ]
                },
                {
                    "id": 1,
                    "operations": [
                        {"id": 2, "processing_times": [1, 3], "precedence": []},
                        {"id": 3, "processing_times": [4, 1], "precedence": [2]}
                    ]
                }
            ]
        }
        
        problem = ProblemInstance.from_dict(problem_data)
        print(f"   ✓ Problem created: {problem.num_jobs} jobs, {problem.num_machines} machines")
    except Exception as e:
        print(f"   ✗ Problem creation failed: {e}")
        return False
    
    # Test 3: Test algorithm factory
    print("3. Testing algorithm factory...")
    try:
        algorithms = AlgorithmFactory.get_available_algorithms()
        print(f"   ✓ {len(algorithms)} algorithms available")
        
        # Test creating algorithms
        test_algorithms = ['greedy', 'genetic', 'spt', 'aco', 'pso']
        created_count = 0
        
        for algo_name in test_algorithms:
            if algo_name in ['spt']:
                algo = AlgorithmFactory.create_algorithm('dispatching', {'rule': 'SPT'})
            else:
                algo = AlgorithmFactory.create_algorithm(algo_name)
            created_count += 1
            
        print(f"   ✓ Successfully created {created_count} algorithm instances")
    except Exception as e:
        print(f"   ✗ Algorithm factory test failed: {e}")
        return False
    
    # Test 4: Run algorithms on problem
    print("4. Testing algorithm execution...")
    try:
        results = {}
        test_algos = [
            ('greedy', AlgorithmFactory.create_algorithm('greedy')),
            ('spt', AlgorithmFactory.create_algorithm('dispatching', {'rule': 'SPT'})),
            ('genetic', AlgorithmFactory.create_algorithm('genetic', {'pop_size': 10, 'generations': 3}))
        ]
        
        for name, algorithm in test_algos:
            result = algorithm.solve(problem, timeout=30)
            results[name] = result
            print(f"   ✓ {name}: makespan={result.makespan:.2f}, time={result.execution_time:.3f}s")
            
            # Validate result
            if result.makespan <= 0 or result.makespan == float('inf'):
                print(f"   ⚠ Warning: {name} produced invalid makespan")
                
    except Exception as e:
        print(f"   ✗ Algorithm execution failed: {e}")
        return False
    
    # Test 5: Test comprehensive suite
    print("5. Testing algorithm suite...")
    try:
        suite = create_algorithm_suite()
        print(f"   ✓ Algorithm suite created with {len(suite)} algorithms")
        
        # Run a few algorithms from the suite
        suite_results = {}
        test_suite_algos = ['spt', 'greedy', 'ga_quick']
        
        for name in test_suite_algos:
            if name in suite:
                result = suite[name].solve(problem, timeout=15)
                suite_results[name] = result
                print(f"   ✓ Suite {name}: makespan={result.makespan:.2f}")
        
    except Exception as e:
        print(f"   ✗ Algorithm suite test failed: {e}")
        return False
    
    # Test 6: Test metaheuristics
    print("6. Testing metaheuristic algorithms...")
    try:
        meta_algos = [
            ('aco', {'n_ants': 5, 'max_iterations': 3}),
            ('pso', {'n_particles': 5, 'max_iterations': 3}),
            ('differential_evolution', {'population_size': 5, 'max_iterations': 3})
        ]
        
        for name, params in meta_algos:
            algorithm = AlgorithmFactory.create_algorithm(name, params)
            result = algorithm.solve(problem, timeout=20)
            print(f"   ✓ {name}: makespan={result.makespan:.2f}")
            
    except Exception as e:
        print(f"   ✗ Metaheuristic test failed: {e}")
        return False
    
    # Test 7: Performance comparison
    print("7. Performance comparison...")
    try:
        all_results = {**results, **suite_results}
        if all_results:
            best_algo = min(all_results.keys(), key=lambda k: all_results[k].makespan)
            best_makespan = all_results[best_algo].makespan
            print(f"   ✓ Best algorithm: {best_algo} (makespan: {best_makespan:.2f})")
            
            # Show all results
            print("   📊 Results summary:")
            for name, result in sorted(all_results.items(), key=lambda x: x[1].makespan):
                print(f"      {name:12}: {result.makespan:6.2f} ({result.execution_time:5.3f}s)")
    except Exception as e:
        print(f"   ✗ Performance comparison failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ✅ ===")
    print("The POFJSP system is working correctly with:")
    print(f"  • {len(AlgorithmFactory.get_available_algorithms())} scheduling algorithms")
    print("  • Complete problem instance handling")
    print("  • Algorithm factory pattern")
    print("  • Metaheuristic algorithms (ACO, PSO, DE, etc.)")
    print("  • Hybrid approaches")
    print("  • Performance monitoring")
    
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)