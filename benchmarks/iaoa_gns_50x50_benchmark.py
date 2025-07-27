#!/usr/bin/env python3
"""
Focused 50x50 IAOA+GNS Benchmark

Optimized benchmark focusing on the key comparison requested:
IAOA+GNS vs representative algorithms on a 50x50 POFJSP instance.
"""

import sys
from pathlib import Path
import time
import numpy as np
import random
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from problems.problem_instance import ProblemInstance, Operation
from algorithms.iaoa_gns import IAOAGNSAlgorithm, IAOAConfig
from algorithms.baseline_algorithms import (
    DispatchingRulesAlgorithm, GeneticAlgorithm, SimulatedAnnealing, GreedyAlgorithm
)

def generate_efficient_50x50_problem(seed: int = 42):
    """Generate an efficient 50x50 POFJSP instance for testing."""
    np.random.seed(seed)
    random.seed(seed)
    
    print("Generating optimized 50x50 POFJSP instance...")
    num_jobs = 50
    num_machines = 50
    
    # Moderate complexity: 2-5 operations per job
    num_operations_per_job = [random.randint(2, 5) for _ in range(num_jobs)]
    total_operations = sum(num_operations_per_job)
    
    print(f"Total operations: {total_operations}")
    
    # Generate processing times with higher machine flexibility
    processing_times = []
    for job_idx in range(num_jobs):
        job_processing_times = []
        for op_idx in range(num_operations_per_job[job_idx]):
            # Higher flexibility: 60-80% of machines are valid
            num_valid_machines = random.randint(
                int(0.6 * num_machines), int(0.8 * num_machines)
            )
            valid_machines = sorted(random.sample(range(num_machines), num_valid_machines))
            
            op_times = np.full(num_machines, np.inf)
            for machine in valid_machines:
                op_times[machine] = random.randint(20, 80)  # Tighter processing time range
            
            job_processing_times.append(op_times)
        
        processing_times.append(np.array(job_processing_times))
    
    # Simple chain precedence (no inter-job constraints for efficiency)
    predecessors_map = {}
    successors_map = {}
    
    for job_idx in range(num_jobs):
        for op_idx in range(num_operations_per_job[job_idx]):
            operation = Operation(job_idx, op_idx)
            
            if op_idx > 0:
                predecessor = Operation(job_idx, op_idx - 1)
                predecessors_map[operation] = {predecessor}
                
                if predecessor not in successors_map:
                    successors_map[predecessor] = set()
                successors_map[predecessor].add(operation)
            else:
                predecessors_map[operation] = set()
    
    problem = ProblemInstance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )
    
    avg_flexibility = np.mean([
        np.sum(pt < np.inf) 
        for job_times in processing_times 
        for pt in job_times
    ])
    
    print(f"✓ Generated efficient 50x50 instance:")
    print(f"  • {num_jobs} jobs, {num_machines} machines")
    print(f"  • {total_operations} total operations") 
    print(f"  • Average flexibility: {avg_flexibility:.1f} machines per operation")
    print(f"  • Chain precedence structure (efficient)")
    
    return problem

def run_focused_benchmark(problem: ProblemInstance):
    """Run focused benchmark with key algorithms."""
    print(f"\n{'='*70}")
    print("FOCUSED 50x50 IAOA+GNS BENCHMARK")
    print(f"{'='*70}")
    
    # Define representative algorithms for comparison
    algorithms = [
        ("IAOA+GNS", lambda: IAOAGNSAlgorithm(IAOAConfig(pop_size=30, max_iterations=50)), 600),
        ("SPT_Dispatching", lambda: DispatchingRulesAlgorithm("SPT"), 60),
        ("LPT_Dispatching", lambda: DispatchingRulesAlgorithm("LPT"), 60),
        ("GreedyScheduling", lambda: GreedyAlgorithm(), 60),
        ("GeneticAlgorithm", lambda: GeneticAlgorithm(pop_size=20, generations=30), 300),
        ("SimulatedAnnealing", lambda: SimulatedAnnealing(max_iterations=3000), 300),
    ]
    
    results = {}
    
    for algo_name, algo_factory, timeout in algorithms:
        print(f"\n{'-'*50}")
        print(f"Running {algo_name} (timeout: {timeout}s)")
        print(f"{'-'*50}")
        
        try:
            start_time = time.time()
            
            if algo_name == "IAOA+GNS":
                # Special handling for IAOA+GNS with verbose output
                algorithm = algo_factory()
                print("Starting IAOA+GNS optimization...")
                solution = algorithm.solve(problem, verbose=True)
                makespan = solution.makespan
                additional_metrics = algorithm.get_algorithm_info()
            else:
                # Run baseline algorithm
                algorithm = algo_factory()
                result = algorithm.solve(problem, timeout)
                makespan = result.makespan
                additional_metrics = result.additional_metrics or {}
            
            execution_time = time.time() - start_time
            
            print(f"✓ {algo_name} completed:")
            print(f"  Makespan: {makespan:.2f}")
            print(f"  Execution time: {execution_time:.2f}s")
            
            results[algo_name] = {
                "makespan": float(makespan) if makespan != float('inf') else None,
                "execution_time": execution_time,
                "additional_metrics": additional_metrics
            }
            
        except Exception as e:
            print(f"✗ {algo_name} failed: {e}")
            results[algo_name] = {
                "makespan": None,
                "execution_time": timeout,
                "error": str(e)
            }
    
    return results

def analyze_focused_results(results: dict):
    """Analyze and display focused benchmark results."""
    print(f"\n{'='*70}")
    print("FOCUSED BENCHMARK RESULTS ANALYSIS")
    print(f"{'='*70}")
    
    # Extract valid results
    valid_results = {
        name: data for name, data in results.items() 
        if data.get("makespan") is not None
    }
    
    if not valid_results:
        print("❌ No valid results obtained!")
        return
    
    # Sort by makespan
    sorted_results = sorted(
        valid_results.items(), 
        key=lambda x: x[1]["makespan"]
    )
    
    best_makespan = sorted_results[0][1]["makespan"]
    
    print(f"\n🏆 ALGORITHM RANKING (by Solution Quality):")
    print("="*70)
    print(f"{'Rank':<6} {'Algorithm':<20} {'Makespan':<12} {'Gap %':<10} {'Time (s)':<12}")
    print("-" * 70)
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        makespan = data["makespan"]
        exec_time = data["execution_time"]
        gap_percent = ((makespan - best_makespan) / best_makespan) * 100
        
        print(f"{rank:<6} {name:<20} {makespan:<12.2f} {gap_percent:<10.1f} {exec_time:<12.2f}")
    
    # IAOA+GNS specific analysis
    if "IAOA+GNS" in valid_results:
        iaoa_data = valid_results["IAOA+GNS"]
        iaoa_rank = next(i for i, (name, _) in enumerate(sorted_results, 1) if name == "IAOA+GNS")
        
        print(f"\n🔬 IAOA+GNS DETAILED ANALYSIS:")
        print("="*50)
        print(f"  • Final makespan: {iaoa_data['makespan']:.2f}")
        print(f"  • Quality ranking: {iaoa_rank}/{len(valid_results)}")
        print(f"  • Execution time: {iaoa_data['execution_time']:.2f} seconds")
        
        if iaoa_rank == 1:
            print(f"  • 🎯 ACHIEVED BEST SOLUTION!")
        else:
            gap = ((iaoa_data['makespan'] - best_makespan) / best_makespan) * 100
            print(f"  • Gap from best: {gap:.1f}%")
        
        # Compare against categories
        dispatching_algos = [name for name in valid_results.keys() if "Dispatching" in name]
        metaheuristic_algos = [name for name in valid_results.keys() if name in ["GeneticAlgorithm", "SimulatedAnnealing"]]
        
        if dispatching_algos:
            best_dispatching_makespan = min(valid_results[name]["makespan"] for name in dispatching_algos)
            improvement = ((best_dispatching_makespan - iaoa_data['makespan']) / best_dispatching_makespan) * 100
            print(f"  • Improvement vs best dispatching rule: {improvement:.1f}%")
        
        if metaheuristic_algos:
            best_metaheuristic_makespan = min(valid_results[name]["makespan"] for name in metaheuristic_algos)
            improvement = ((best_metaheuristic_makespan - iaoa_data['makespan']) / best_metaheuristic_makespan) * 100
            print(f"  • Improvement vs best metaheuristic: {improvement:.1f}%")
    
    # Summary statistics
    makespans = [data["makespan"] for data in valid_results.values()]
    exec_times = [data["execution_time"] for data in valid_results.values()]
    
    print(f"\n📊 BENCHMARK SUMMARY:")
    print("="*50)
    print(f"  • Algorithms tested: {len(results)}")
    print(f"  • Successful runs: {len(valid_results)}")
    print(f"  • Best makespan: {min(makespans):.2f}")
    print(f"  • Worst makespan: {max(makespans):.2f}")
    print(f"  • Makespan range: {max(makespans) - min(makespans):.2f}")
    print(f"  • Average makespan: {np.mean(makespans):.2f} ± {np.std(makespans):.2f}")
    print(f"  • Fastest algorithm: {min(exec_times):.2f}s")
    print(f"  • Slowest algorithm: {max(exec_times):.2f}s")
    
    return {
        "ranking": [(name, data["makespan"], data["execution_time"]) for name, data in sorted_results],
        "best_algorithm": sorted_results[0][0],
        "best_makespan": best_makespan,
        "summary_stats": {
            "algorithms_tested": len(results),
            "successful_runs": len(valid_results),
            "makespan_range": [min(makespans), max(makespans)],
            "execution_time_range": [min(exec_times), max(exec_times)]
        }
    }

def main():
    """Main focused benchmark execution."""
    print("🚀 Focused 50x50 IAOA+GNS Benchmark")
    print("="*80)
    print("Testing IAOA+GNS against representative scheduling algorithms")
    print("on an optimized 50x50 POFJSP instance")
    print("="*80)
    
    try:
        # Generate optimized 50x50 problem
        problem = generate_efficient_50x50_problem(seed=42)
        
        # Run focused benchmark
        results = run_focused_benchmark(problem)
        
        # Analyze results
        analysis = analyze_focused_results(results)
        
        # Save results
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problem_info": {
                "num_jobs": problem.num_jobs,
                "num_machines": problem.num_machines,
                "total_operations": problem.total_operations,
                "precedence_constraints": len(problem.predecessors_map)
            },
            "results": results,
            "analysis": analysis
        }
        
        with open("focused_50x50_results.json", "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: focused_50x50_results.json")
        
        print(f"\n{'='*80}")
        print("🎉 FOCUSED 50x50 BENCHMARK COMPLETED!")
        print(f"{'='*80}")
        
        if analysis and "best_algorithm" in analysis:
            print(f"\n🏆 Winner: {analysis['best_algorithm']} with makespan {analysis['best_makespan']:.2f}")
            
            if analysis['best_algorithm'] == "IAOA+GNS":
                print("🎯 IAOA+GNS achieved the best solution on the 50x50 instance!")
            else:
                if "IAOA+GNS" in results and results["IAOA+GNS"].get("makespan"):
                    iaoa_makespan = results["IAOA+GNS"]["makespan"]
                    gap = ((iaoa_makespan - analysis['best_makespan']) / analysis['best_makespan']) * 100
                    print(f"📈 IAOA+GNS gap from best: {gap:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Focused benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)