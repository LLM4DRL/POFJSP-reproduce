"""
Benchmark Suite for POFJSP Algorithms

Provides standardized benchmarks to evaluate and compare
algorithm performance across different problem instances.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

from performance.monitor import AlgorithmBenchmark, performance_tracker
from problems.problem_instance import ProblemInstance

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSuite:
    """Configuration for a benchmark suite."""
    name: str
    description: str
    problem_sizes: List[Tuple[int, int]]  # (jobs, machines)
    algorithms: Dict[str, Dict[str, Any]]  # algorithm_name -> config
    num_runs: int = 5
    timeout_seconds: int = 300


class StandardBenchmarkSuite:
    """Standard benchmark problems for POFJSP algorithms."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.benchmark_engine = AlgorithmBenchmark(str(self.output_dir))
        
    def create_test_problem(self, num_jobs: int, num_machines: int, 
                          complexity: str = "medium") -> ProblemInstance:
        """Create a standardized test problem."""
        np.random.seed(42)  # For reproducible benchmarks
        
        # Generate number of operations per job
        if complexity == "simple":
            ops_per_job = [np.random.randint(2, 4) for _ in range(num_jobs)]
        elif complexity == "medium":
            ops_per_job = [np.random.randint(3, 6) for _ in range(num_jobs)]
        else:  # complex
            ops_per_job = [np.random.randint(4, 8) for _ in range(num_jobs)]
        
        # Generate processing times
        processing_times = []
        for job_idx in range(num_jobs):
            job_ops = ops_per_job[job_idx]
            job_times = np.random.randint(10, 100, size=(job_ops, num_machines)).astype(float)
            
            # Add some infeasible assignments (machine cannot process operation)
            if complexity != "simple":
                infeasible_ratio = 0.3 if complexity == "medium" else 0.5
                for op_idx in range(job_ops):
                    infeasible_machines = np.random.choice(
                        num_machines, 
                        size=int(num_machines * infeasible_ratio), 
                        replace=False
                    )
                    job_times[op_idx, infeasible_machines] = np.inf
            
            processing_times.append(job_times)
        
        # Generate simple precedence constraints (within jobs)
        from src.problems.problem_instance import Operation
        predecessors_map = {}
        successors_map = {}
        
        # Initialize maps
        all_operations = []
        for j in range(num_jobs):
            for o in range(ops_per_job[j]):
                op = Operation(j, o)
                all_operations.append(op)
                predecessors_map[op] = set()
                successors_map[op] = set()
        
        # Add intra-job precedence constraints
        for job_idx in range(num_jobs):
            for op_idx in range(1, ops_per_job[job_idx]):
                current_op = Operation(job_idx, op_idx)
                prev_op = Operation(job_idx, op_idx - 1)
                
                predecessors_map[current_op].add(prev_op)
                successors_map[prev_op].add(current_op)
        
        # Add some inter-job constraints for complex problems
        if complexity == "complex" and num_jobs > 1:
            num_inter_constraints = min(5, num_jobs // 2)
            for _ in range(num_inter_constraints):
                job1, job2 = np.random.choice(num_jobs, 2, replace=False)
                op1 = Operation(job1, np.random.randint(ops_per_job[job1]))
                op2 = Operation(job2, np.random.randint(ops_per_job[job2]))
                
                if op2 not in predecessors_map[op1]:  # Avoid cycles
                    predecessors_map[op2].add(op1)
                    successors_map[op1].add(op2)
        
        return ProblemInstance(
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_operations_per_job=ops_per_job,
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
    
    def run_small_scale_benchmark(self) -> Dict[str, Any]:
        """Run benchmark on small-scale problems (quick validation)."""
        logger.info("Running small-scale benchmark suite")
        
        problem_configs = [
            (6, 4, "simple"),
            (8, 5, "medium"),
            (10, 6, "medium")
        ]
        
        results = {}
        
        for num_jobs, num_machines, complexity in problem_configs:
            problem_key = f"{num_jobs}x{num_machines}_{complexity}"
            logger.info(f"Testing problem size: {problem_key}")
            
            problem = self.create_test_problem(num_jobs, num_machines, complexity)
            problem_results = {}
            
            # Test IAOA+GNS
            try:
                from src.algorithms.iaoa_gns_refactored import IAOAGNSAlgorithm
                from src.config import IAOAGNSConfig
                
                config = IAOAGNSConfig(pop_size=30, max_iterations=50)
                
                def iaoa_factory():
                    return IAOAGNSAlgorithm(config)
                
                result = self.benchmark_engine.benchmark_algorithm(
                    iaoa_factory, problem, "IAOA_GNS", num_runs=3
                )
                problem_results["IAOA_GNS"] = {
                    "best_makespan": result.best_makespan,
                    "avg_makespan": result.avg_makespan,
                    "execution_time": result.metrics.execution_time,
                    "success_rate": result.success_rate
                }
                
            except Exception as e:
                logger.error(f"IAOA+GNS benchmark failed: {e}")
                problem_results["IAOA_GNS"] = {"status": "failed", "error": str(e)}
            
            # Test PPO (if available)
            try:
                from src.rl.models.ppo_agent import PPOAgent
                from src.rl.environments.pofjsp_env import POFJSPEnv
                
                def ppo_factory():
                    class PPOWrapper:
                        def __init__(self):
                            self.env = POFJSPEnv(problem)
                            self.agent = PPOAgent(
                                input_dim=8,
                                hidden_dim=64,
                                num_jobs=problem.num_jobs,
                                num_machines=problem.num_machines
                            )
                        
                        def solve(self, problem_instance):
                            # Simple greedy solution for benchmark
                            from src.problems.problem_instance import Solution
                            operations = problem_instance.all_operations
                            machines = [0] * len(operations)  # Assign all to machine 0
                            return Solution(operations, machines)
                    
                    return PPOWrapper()
                
                result = self.benchmark_engine.benchmark_algorithm(
                    ppo_factory, problem, "PPO_Greedy", num_runs=3
                )
                problem_results["PPO_Greedy"] = {
                    "best_makespan": result.best_makespan,
                    "avg_makespan": result.avg_makespan,
                    "execution_time": result.metrics.execution_time,
                    "success_rate": result.success_rate
                }
                
            except Exception as e:
                logger.error(f"PPO benchmark failed: {e}")
                problem_results["PPO_Greedy"] = {"status": "failed", "error": str(e)}
            
            results[problem_key] = problem_results
        
        # Save results
        self._save_benchmark_results("small_scale", results)
        return results
    
    def run_medium_scale_benchmark(self) -> Dict[str, Any]:
        """Run benchmark on medium-scale problems."""
        logger.info("Running medium-scale benchmark suite")
        
        problem_configs = [
            (15, 8, "medium"),
            (20, 10, "medium"),
            (25, 12, "complex")
        ]
        
        results = {}
        
        for num_jobs, num_machines, complexity in problem_configs:
            problem_key = f"{num_jobs}x{num_machines}_{complexity}"
            logger.info(f"Testing problem size: {problem_key}")
            
            problem = self.create_test_problem(num_jobs, num_machines, complexity)
            problem_results = {}
            
            # Test IAOA+GNS with larger parameters
            try:
                from src.algorithms.iaoa_gns_refactored import IAOAGNSAlgorithm
                from src.config import IAOAGNSConfig
                
                config = IAOAGNSConfig(pop_size=60, max_iterations=100)
                
                def iaoa_factory():
                    return IAOAGNSAlgorithm(config)
                
                result = self.benchmark_engine.benchmark_algorithm(
                    iaoa_factory, problem, "IAOA_GNS_Medium", num_runs=5
                )
                problem_results["IAOA_GNS"] = {
                    "best_makespan": result.best_makespan,
                    "avg_makespan": result.avg_makespan,
                    "std_makespan": result.std_makespan,
                    "execution_time": result.metrics.execution_time,
                    "memory_peak_mb": result.metrics.memory_peak_mb,
                    "success_rate": result.success_rate,
                    "convergence_iteration": result.metrics.convergence_iteration
                }
                
            except Exception as e:
                logger.error(f"IAOA+GNS medium benchmark failed: {e}")
                problem_results["IAOA_GNS"] = {"status": "failed", "error": str(e)}
            
            results[problem_key] = problem_results
        
        self._save_benchmark_results("medium_scale", results)
        return results
    
    def run_large_scale_benchmark(self) -> Dict[str, Any]:
        """Run benchmark on large-scale problems (stress test)."""
        logger.info("Running large-scale benchmark suite")
        
        problem_configs = [
            (50, 20, "complex"),
            (75, 25, "complex"),
            (100, 30, "complex")
        ]
        
        results = {}
        
        for num_jobs, num_machines, complexity in problem_configs:
            problem_key = f"{num_jobs}x{num_machines}_{complexity}"
            logger.info(f"Testing problem size: {problem_key}")
            
            problem = self.create_test_problem(num_jobs, num_machines, complexity)
            problem_results = {}
            
            # Test IAOA+GNS with optimized parameters for large problems
            try:
                from src.algorithms.iaoa_gns_refactored import IAOAGNSAlgorithm
                from src.config import IAOAGNSConfig
                
                config = IAOAGNSConfig(
                    pop_size=80, 
                    max_iterations=200,
                    crossover_rate=0.8,
                    mutation_rate=0.1
                )
                
                def iaoa_factory():
                    return IAOAGNSAlgorithm(config)
                
                # Use performance tracker for detailed monitoring
                with performance_tracker("IAOA_GNS_Large") as tracker:
                    result = self.benchmark_engine.benchmark_algorithm(
                        iaoa_factory, problem, "IAOA_GNS_Large", num_runs=3
                    )
                    
                    # Record additional metrics
                    tracker.record_algorithm_metric("problem_size", f"{num_jobs}x{num_machines}")
                    tracker.record_algorithm_metric("total_operations", problem.total_operations)
                
                problem_results["IAOA_GNS"] = {
                    "best_makespan": result.best_makespan,
                    "avg_makespan": result.avg_makespan,
                    "std_makespan": result.std_makespan,
                    "execution_time": result.metrics.execution_time,
                    "memory_peak_mb": result.metrics.memory_peak_mb,
                    "memory_avg_mb": result.metrics.memory_avg_mb,
                    "cpu_peak_percent": result.metrics.cpu_peak_percent,
                    "success_rate": result.success_rate,
                    "convergence_iteration": result.metrics.convergence_iteration,
                    "scalability_score": self._calculate_scalability_score(result, problem)
                }
                
            except Exception as e:
                logger.error(f"IAOA+GNS large benchmark failed: {e}")
                problem_results["IAOA_GNS"] = {"status": "failed", "error": str(e)}
            
            results[problem_key] = problem_results
        
        self._save_benchmark_results("large_scale", results)
        return results
    
    def _calculate_scalability_score(self, result, problem) -> float:
        """Calculate a scalability score based on performance vs problem size."""
        # Simple heuristic: lower time per operation is better
        if result.metrics.execution_time > 0 and problem.total_operations > 0:
            time_per_op = result.metrics.execution_time / problem.total_operations
            # Normalize to 0-100 scale (lower is better, so invert)
            return max(0, 100 - (time_per_op * 100))
        return 0.0
    
    def _save_benchmark_results(self, suite_name: str, results: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{suite_name}_benchmark_{timestamp}.json"
        filepath = self.output_dir / filename
        
        benchmark_data = {
            "suite_name": suite_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "summary": self._generate_summary(results)
        }
        
        with open(filepath, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            "total_problems": len(results),
            "successful_problems": 0,
            "failed_problems": 0,
            "algorithms_tested": set(),
            "avg_execution_time": 0.0,
            "avg_memory_usage": 0.0
        }
        
        execution_times = []
        memory_usages = []
        
        for problem_key, problem_results in results.items():
            problem_success = False
            
            for alg_name, alg_results in problem_results.items():
                summary["algorithms_tested"].add(alg_name)
                
                if isinstance(alg_results, dict) and "status" not in alg_results:
                    problem_success = True
                    
                    if "execution_time" in alg_results:
                        execution_times.append(alg_results["execution_time"])
                    if "memory_peak_mb" in alg_results:
                        memory_usages.append(alg_results["memory_peak_mb"])
            
            if problem_success:
                summary["successful_problems"] += 1
            else:
                summary["failed_problems"] += 1
        
        summary["algorithms_tested"] = list(summary["algorithms_tested"])
        
        if execution_times:
            summary["avg_execution_time"] = np.mean(execution_times)
            summary["max_execution_time"] = np.max(execution_times)
            summary["min_execution_time"] = np.min(execution_times)
        
        if memory_usages:
            summary["avg_memory_usage"] = np.mean(memory_usages)
            summary["max_memory_usage"] = np.max(memory_usages)
        
        return summary
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting full benchmark suite")
        
        full_results = {
            "benchmark_start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "small_scale": {},
            "medium_scale": {},
            "large_scale": {},
            "benchmark_end": None,
            "total_duration": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Run small scale first
            full_results["small_scale"] = self.run_small_scale_benchmark()
            
            # Run medium scale
            full_results["medium_scale"] = self.run_medium_scale_benchmark()
            
            # Run large scale
            full_results["large_scale"] = self.run_large_scale_benchmark()
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
            full_results["error"] = str(e)
        
        finally:
            end_time = time.time()
            full_results["benchmark_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
            full_results["total_duration"] = end_time - start_time
            
            # Save complete results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"full_benchmark_suite_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(full_results, f, indent=2)
            
            logger.info(f"Full benchmark suite completed in {full_results['total_duration']:.1f} seconds")
            logger.info(f"Complete results saved to {filepath}")
        
        return full_results


# Command-line interface for running benchmarks
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run POFJSP algorithm benchmarks")
    parser.add_argument("--suite", choices=["small", "medium", "large", "full"], 
                       default="small", help="Benchmark suite to run")
    parser.add_argument("--output-dir", default="benchmark_results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run benchmark
    benchmark_suite = StandardBenchmarkSuite(args.output_dir)
    
    if args.suite == "small":
        results = benchmark_suite.run_small_scale_benchmark()
    elif args.suite == "medium":
        results = benchmark_suite.run_medium_scale_benchmark()
    elif args.suite == "large":
        results = benchmark_suite.run_large_scale_benchmark()
    else:  # full
        results = benchmark_suite.run_full_benchmark_suite()
    
    print(f"\nBenchmark completed! Results saved to {args.output_dir}")