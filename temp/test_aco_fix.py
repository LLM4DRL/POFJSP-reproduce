#!/usr/bin/env python3
"""
Test script to verify ACO algorithm fix
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_aco_algorithm():
    """Test the fixed ACO algorithm."""
    print("=== Testing ACO Algorithm Fix ===")
    
    try:
        from src.algorithms.baseline_algorithms import AntColonyOptimization
        from src.problems.problem_instance import ProblemInstance
        print("✓ Imports successful")
        
        # Create a simple test problem
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
        print("✓ Problem instance created")
        
        # Test ACO algorithm
        aco = AntColonyOptimization(n_ants=5, max_iterations=3)  # Small values for quick test
        print("✓ ACO algorithm initialized")
        
        result = aco.solve(problem, timeout=10.0)
        print(f"✓ ACO solve completed")
        print(f"  Algorithm: {result.algorithm_name}")
        print(f"  Makespan: {result.makespan}")
        print(f"  Execution time: {result.execution_time:.4f}s")
        
        if result.additional_metrics and "error" in result.additional_metrics:
            print(f"  ✗ Error occurred: {result.additional_metrics['error']}")
            return False
        elif result.makespan == float('inf'):
            print("  ✗ Algorithm returned infinite makespan")
            return False
        else:
            print("  ✓ Algorithm completed successfully with finite makespan")
            return True
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_aco_algorithm()
    sys.exit(0 if success else 1)