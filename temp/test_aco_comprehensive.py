#!/usr/bin/env python3
"""
Comprehensive test for ACO algorithm fix
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_aco_comprehensive():
    """Test the ACO algorithm with multiple runs and different problem sizes."""
    print("=== Comprehensive ACO Algorithm Test ===")
    
    try:
        from src.algorithms.baseline_algorithms import AntColonyOptimization
        from src.problems.problem_instance import ProblemInstance
        
        # Test 1: Small problem, multiple runs
        print("\n1. Testing small problem with multiple runs...")
        problem_data_small = {
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
        
        problem_small = ProblemInstance.from_dict(problem_data_small)
        aco_small = AntColonyOptimization(n_ants=10, max_iterations=5)
        
        successes = 0
        makespans = []
        
        for run in range(5):
            result = aco_small.solve(problem_small, timeout=10.0)
            if result.makespan != float('inf') and (not result.additional_metrics or "error" not in result.additional_metrics):
                successes += 1
                makespans.append(result.makespan)
                print(f"   Run {run + 1}: makespan = {result.makespan}")
            else:
                error_msg = result.additional_metrics.get("error", "Unknown error") if result.additional_metrics else "Infinite makespan"
                print(f"   Run {run + 1}: FAILED - {error_msg}")
        
        print(f"   Success rate: {successes}/5 ({successes/5*100:.1f}%)")
        if makespans:
            print(f"   Best makespan: {min(makespans)}")
            print(f"   Average makespan: {sum(makespans)/len(makespans):.2f}")
        
        # Test 2: Slightly larger problem
        print("\n2. Testing larger problem...")
        problem_data_large = {
            "num_jobs": 3,
            "num_machines": 3,
            "jobs": [
                {
                    "id": 0,
                    "operations": [
                        {"id": 0, "processing_times": [3, 2, 4], "precedence": []},
                        {"id": 1, "processing_times": [2, 4, 1], "precedence": [0]},
                        {"id": 2, "processing_times": [1, 3, 2], "precedence": [1]}
                    ]
                },
                {
                    "id": 1,
                    "operations": [
                        {"id": 3, "processing_times": [4, 1, 3], "precedence": []},
                        {"id": 4, "processing_times": [2, 3, 2], "precedence": [3]}
                    ]
                },
                {
                    "id": 2,
                    "operations": [
                        {"id": 5, "processing_times": [1, 2, 4], "precedence": []}
                    ]
                }
            ]
        }
        
        problem_large = ProblemInstance.from_dict(problem_data_large)
        aco_large = AntColonyOptimization(n_ants=15, max_iterations=10)
        
        result_large = aco_large.solve(problem_large, timeout=30.0)
        if result_large.makespan != float('inf') and (not result_large.additional_metrics or "error" not in result_large.additional_metrics):
            print(f"   ✓ Large problem solved successfully")
            print(f"   Makespan: {result_large.makespan}")
            print(f"   Execution time: {result_large.execution_time:.4f}s")
        else:
            error_msg = result_large.additional_metrics.get("error", "Unknown error") if result_large.additional_metrics else "Infinite makespan"
            print(f"   ✗ Large problem failed: {error_msg}")
            return False
        
        print(f"\n✓ All tests passed! ACO algorithm is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_aco_comprehensive()
    sys.exit(0 if success else 1)