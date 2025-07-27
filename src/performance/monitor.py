"""
Performance Monitoring System

Provides comprehensive performance tracking for POFJSP algorithms
including memory usage, execution time, and algorithm convergence metrics.
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0
    cpu_peak_percent: float = 0.0
    cpu_avg_percent: float = 0.0
    makespan_progression: List[float] = field(default_factory=list)
    convergence_iteration: Optional[int] = None
    algorithm_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    algorithm_name: str
    problem_size: str
    best_makespan: float
    avg_makespan: float
    std_makespan: float
    success_rate: float
    metrics: PerformanceMetrics
    timestamp: str


class ResourceMonitor:
    """Real-time system resource monitoring."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        
        # Metrics storage
        self.cpu_samples = deque(maxlen=1000)
        self.memory_samples = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
    
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.debug("Resource monitoring started")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics."""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.cpu_samples:
            return {"memory_peak_mb": 0.0, "memory_avg_mb": 0.0, "cpu_peak_percent": 0.0, "cpu_avg_percent": 0.0}
        
        cpu_samples = list(self.cpu_samples)
        memory_samples = list(self.memory_samples)
        
        metrics = {
            "memory_peak_mb": max(memory_samples),
            "memory_avg_mb": np.mean(memory_samples),
            "cpu_peak_percent": max(cpu_samples),
            "cpu_avg_percent": np.mean(cpu_samples)
        }
        
        logger.debug(f"Resource monitoring stopped. Peak memory: {metrics['memory_peak_mb']:.1f}MB")
        return metrics
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Sample CPU and memory
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_mb)
                self.timestamps.append(time.time())
                
                time.sleep(self.sampling_interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.sampling_interval)


class PerformanceTracker:
    """Main performance tracking interface."""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.current_metrics = PerformanceMetrics()
        self.start_time = None
        self.makespan_history = []
        
    def start_tracking(self, algorithm_name: str):
        """Start performance tracking for an algorithm run."""
        self.algorithm_name = algorithm_name
        self.start_time = time.time()
        self.current_metrics = PerformanceMetrics()
        self.makespan_history = []
        
        self.resource_monitor.start_monitoring()
        logger.info(f"Performance tracking started for {algorithm_name}")
    
    def record_makespan(self, makespan: float, iteration: int):
        """Record makespan at specific iteration."""
        self.makespan_history.append((iteration, makespan))
        self.current_metrics.makespan_progression.append(makespan)
        
        # Check for convergence (no improvement in last 10 iterations)
        if len(self.makespan_history) >= 10:
            recent_makespans = [ms for _, ms in self.makespan_history[-10:]]
            if all(ms >= recent_makespans[0] for ms in recent_makespans[1:]):
                if self.current_metrics.convergence_iteration is None:
                    self.current_metrics.convergence_iteration = iteration - 9
    
    def record_algorithm_metric(self, key: str, value: Any):
        """Record algorithm-specific metric."""
        self.current_metrics.algorithm_specific[key] = value
    
    def stop_tracking(self) -> PerformanceMetrics:
        """Stop tracking and return final metrics."""
        if self.start_time is None:
            return self.current_metrics
        
        # Calculate execution time
        self.current_metrics.execution_time = time.time() - self.start_time
        
        # Get resource metrics
        resource_metrics = self.resource_monitor.stop_monitoring()
        self.current_metrics.memory_peak_mb = resource_metrics.get("memory_peak_mb", 0.0)
        self.current_metrics.memory_avg_mb = resource_metrics.get("memory_avg_mb", 0.0)
        self.current_metrics.cpu_peak_percent = resource_metrics.get("cpu_peak_percent", 0.0)
        self.current_metrics.cpu_avg_percent = resource_metrics.get("cpu_avg_percent", 0.0)
        
        logger.info(f"Performance tracking completed for {self.algorithm_name}. "
                   f"Time: {self.current_metrics.execution_time:.2f}s, "
                   f"Peak Memory: {self.current_metrics.memory_peak_mb:.1f}MB")
        
        return self.current_metrics


class AlgorithmBenchmark:
    """Comprehensive algorithm benchmarking."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def benchmark_algorithm(self, 
                          algorithm_factory: Callable,
                          problem_instance,
                          algorithm_name: str,
                          num_runs: int = 5,
                          **algorithm_kwargs) -> BenchmarkResult:
        """Benchmark an algorithm on a problem instance."""
        logger.info(f"Benchmarking {algorithm_name} on problem {problem_instance} ({num_runs} runs)")
        
        run_results = []
        all_metrics = []
        
        for run in range(num_runs):
            logger.debug(f"Run {run + 1}/{num_runs}")
            
            # Create fresh algorithm instance
            algorithm = algorithm_factory(**algorithm_kwargs)
            
            # Track performance
            tracker = PerformanceTracker()
            tracker.start_tracking(f"{algorithm_name}_run_{run}")
            
            try:
                # Run algorithm
                solution = algorithm.solve(problem_instance)
                makespan = solution.makespan if hasattr(solution, 'makespan') else float('inf')
                
                # Record final makespan
                tracker.record_makespan(makespan, getattr(algorithm, 'current_iteration', 0))
                
                # Record algorithm-specific metrics
                if hasattr(algorithm, 'population_diversity'):
                    tracker.record_algorithm_metric('final_diversity', algorithm.population_diversity)
                if hasattr(algorithm, 'best_fitness_history'):
                    tracker.record_algorithm_metric('fitness_variance', np.var(algorithm.best_fitness_history))
                
                run_results.append(makespan)
                
            except Exception as e:
                logger.error(f"Algorithm run {run} failed: {e}")
                run_results.append(float('inf'))
            
            finally:
                metrics = tracker.stop_tracking()
                all_metrics.append(metrics)
        
        # Aggregate results
        valid_results = [r for r in run_results if r != float('inf')]
        success_rate = len(valid_results) / num_runs
        
        if valid_results:
            best_makespan = min(valid_results)
            avg_makespan = np.mean(valid_results)
            std_makespan = np.std(valid_results)
        else:
            best_makespan = avg_makespan = std_makespan = float('inf')
        
        # Aggregate metrics
        avg_metrics = PerformanceMetrics()
        if all_metrics:
            avg_metrics.execution_time = np.mean([m.execution_time for m in all_metrics])
            avg_metrics.memory_peak_mb = np.max([m.memory_peak_mb for m in all_metrics])
            avg_metrics.memory_avg_mb = np.mean([m.memory_avg_mb for m in all_metrics])
            avg_metrics.cpu_peak_percent = np.max([m.cpu_peak_percent for m in all_metrics])
            avg_metrics.cpu_avg_percent = np.mean([m.cpu_avg_percent for m in all_metrics])
            
            # Aggregate convergence data
            convergence_iters = [m.convergence_iteration for m in all_metrics if m.convergence_iteration is not None]
            if convergence_iters:
                avg_metrics.convergence_iteration = int(np.mean(convergence_iters))
        
        problem_size = f"{problem_instance.num_jobs}x{problem_instance.num_machines}"
        
        result = BenchmarkResult(
            algorithm_name=algorithm_name,
            problem_size=problem_size,
            best_makespan=best_makespan,
            avg_makespan=avg_makespan,
            std_makespan=std_makespan,
            success_rate=success_rate,
            metrics=avg_metrics,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        self._save_result(result)
        
        logger.info(f"Benchmark completed. Best: {best_makespan:.2f}, "
                   f"Avg: {avg_makespan:.2f} ± {std_makespan:.2f}, "
                   f"Success: {success_rate:.1%}")
        
        return result
    
    def _save_result(self, result: BenchmarkResult):
        """Save individual benchmark result."""
        filename = f"{result.algorithm_name}_{result.problem_size}_{result.timestamp.replace(':', '-').replace(' ', '_')}.json"
        filepath = self.output_dir / filename
        
        # Convert to JSON-serializable format
        result_dict = {
            "algorithm_name": result.algorithm_name,
            "problem_size": result.problem_size,
            "best_makespan": result.best_makespan,
            "avg_makespan": result.avg_makespan,
            "std_makespan": result.std_makespan,
            "success_rate": result.success_rate,
            "timestamp": result.timestamp,
            "metrics": {
                "execution_time": result.metrics.execution_time,
                "memory_peak_mb": result.metrics.memory_peak_mb,
                "memory_avg_mb": result.metrics.memory_avg_mb,
                "cpu_peak_percent": result.metrics.cpu_peak_percent,
                "cpu_avg_percent": result.metrics.cpu_avg_percent,
                "makespan_progression": result.metrics.makespan_progression,
                "convergence_iteration": result.metrics.convergence_iteration,
                "algorithm_specific": result.metrics.algorithm_specific
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def save_summary_report(self, filename: str = None):
        """Save comprehensive benchmark summary."""
        if filename is None:
            filename = f"benchmark_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        summary = {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmarks": len(self.results),
            "results": []
        }
        
        for result in self.results:
            summary["results"].append({
                "algorithm_name": result.algorithm_name,
                "problem_size": result.problem_size,
                "best_makespan": result.best_makespan,
                "avg_makespan": result.avg_makespan,
                "std_makespan": result.std_makespan,
                "success_rate": result.success_rate,
                "execution_time": result.metrics.execution_time,
                "memory_peak_mb": result.metrics.memory_peak_mb,
                "convergence_iteration": result.metrics.convergence_iteration,
                "timestamp": result.timestamp
            })
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Benchmark summary saved to {filepath}")


# Context manager for easy performance tracking
class performance_tracker:
    """Context manager for performance tracking."""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.tracker = PerformanceTracker()
    
    def __enter__(self):
        self.tracker.start_tracking(self.algorithm_name)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        metrics = self.tracker.stop_tracking()
        if exc_type is None:
            logger.info(f"Performance tracking completed successfully for {self.algorithm_name}")
        else:
            logger.warning(f"Performance tracking ended with exception for {self.algorithm_name}: {exc_type.__name__}")
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test resource monitoring
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    # Simulate some work
    time.sleep(1.0)
    
    metrics = monitor.stop_monitoring()
    print(f"Test metrics: {metrics}")