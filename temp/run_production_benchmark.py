#!/usr/bin/env python3
"""
Production-ready comprehensive POFJSP benchmark

This script runs the comprehensive benchmark with realistic parameters
for a CCF-A conference paper submission.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from comprehensive_benchmark import ComprehensiveBenchmark

def main():
    """Run the production benchmark."""
    print("Starting Production POFJSP Comprehensive Benchmark")
    print("=" * 60)
    print("This benchmark will run all algorithms from 5x5 to 150x150")
    print("with multiple runs for statistical significance.")
    print("Expected runtime: 2-6 hours depending on system performance")
    print("=" * 60)
    
    # Initialize comprehensive benchmark
    benchmark = ComprehensiveBenchmark(output_dir="production_benchmark_results")
    
    # For production, we want comprehensive coverage but manageable runtime
    # Adjust problem sizes for comprehensive coverage
    benchmark.problem_sizes = [
        (5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30),
        (40, 40), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90),
        (100, 100), (120, 120), (150, 150)
    ]
    
    print(f"Configured for {len(benchmark.problem_sizes)} problem sizes")
    print(f"Testing {len(benchmark.algorithms)} main algorithms")
    print(f"Plus {len(benchmark.dispatching_rules)} dispatching rules")
    print()
    
    # Run comprehensive benchmark with multiple runs for statistical significance
    runs_per_config = 5  # Increased for better statistical significance
    
    benchmark.run_comprehensive_benchmark(runs_per_config=runs_per_config)
    
    # Generate comprehensive analysis
    df = benchmark.save_results()
    benchmark.generate_analysis_report(df)
    
    print("\n" + "=" * 60)
    print("PRODUCTION BENCHMARK COMPLETED!")
    print(f"Results available in: {benchmark.output_dir}")
    print("Key outputs:")
    print("  - Raw results: benchmark_results_*.csv")
    print("  - Analysis plots: analysis/comprehensive_analysis.pdf")
    print("  - LaTeX tables: analysis/*.tex")
    print("  - Performance heatmap: analysis/performance_heatmap.png")
    print("=" * 60)

if __name__ == "__main__":
    main()