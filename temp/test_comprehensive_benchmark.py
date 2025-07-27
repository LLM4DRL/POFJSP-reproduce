#!/usr/bin/env python3
"""
Test script for the comprehensive benchmark system
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_benchmark_system():
    """Test the comprehensive benchmark system with a small subset."""
    print("=== Testing Comprehensive Benchmark System ===")
    
    try:
        # Import the benchmark system
        from comprehensive_benchmark import ComprehensiveBenchmark, ProblemGenerator
        
        print("✓ Imports successful")
        
        # Test problem generation
        problem = ProblemGenerator.generate_problem(5, 5, complexity='medium', seed=42)
        print(f"✓ Problem generated: {len(problem.all_operations)} operations")
        
        # Test benchmark initialization
        benchmark = ComprehensiveBenchmark(output_dir="test_benchmark_results")
        
        # Override problem sizes for testing
        benchmark.problem_sizes = [(5, 5), (10, 10)]  # Just test small sizes
        benchmark.algorithms = ['genetic', 'aco', 'greedy']  # Limited algorithms
        benchmark.dispatching_rules = ['SPT']  # Single dispatching rule
        
        print("✓ Benchmark system initialized")
        
        # Test algorithm creation
        algorithms = benchmark.create_algorithm_instances((5, 5))
        print(f"✓ Created {len(algorithms)} algorithm instances")
        
        # Test single benchmark run
        genetic_alg = algorithms['genetic']
        result = benchmark.run_single_benchmark('genetic', genetic_alg, problem, (5, 5), 0)
        
        print(f"✓ Single benchmark completed:")
        print(f"  Algorithm: {result.algorithm}")
        print(f"  Makespan: {result.makespan}")
        print(f"  Time: {result.execution_time:.4f}s")
        print(f"  Error: {result.error}")
        
        if result.error:
            print(f"  ✗ Algorithm had error: {result.error}")
            return False
        elif result.makespan == float('inf'):
            print(f"  ✗ Algorithm returned infinite makespan")
            return False
        else:
            print(f"  ✓ Algorithm completed successfully")
        
        # Test mini benchmark run
        print("\n--- Running mini benchmark ---")
        benchmark.run_comprehensive_benchmark(runs_per_config=1)
        
        print(f"✓ Mini benchmark completed with {len(benchmark.results)} results")
        
        # Test analysis generation
        df = benchmark.save_results()
        print(f"✓ Results saved: {len(df)} records")
        
        benchmark.generate_analysis_report(df)
        print("✓ Analysis report generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_benchmark_system()
    sys.exit(0 if success else 1)