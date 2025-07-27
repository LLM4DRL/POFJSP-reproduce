"""
Code Profiling and Analysis Tools

Provides detailed profiling capabilities for POFJSP algorithms
to identify performance bottlenecks and optimization opportunities.
"""

import cProfile
import pstats
import io
import time
import functools
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import threading
import line_profiler
import memory_profiler
from collections import defaultdict

logger = logging.getLogger(__name__)


class FunctionProfiler:
    """Profile individual functions for performance analysis."""
    
    def __init__(self):
        self.profiles = {}
        self.call_counts = defaultdict(int)
        self.total_times = defaultdict(float)
        
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        func_name = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                self.call_counts[func_name] += 1
                self.total_times[func_name] += execution_time
                
                if execution_time > 0.1:  # Log slow calls
                    logger.debug(f"Function {func_name} took {execution_time:.3f}s")
        
        return wrapper
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get profiling statistics."""
        stats = {}
        for func_name in self.call_counts:
            call_count = self.call_counts[func_name]
            total_time = self.total_times[func_name]
            avg_time = total_time / call_count if call_count > 0 else 0
            
            stats[func_name] = {
                'call_count': call_count,
                'total_time': total_time,
                'avg_time': avg_time,
                'percent_total': 0.0  # Will be calculated later
            }
        
        # Calculate percentages
        total_execution_time = sum(self.total_times.values())
        if total_execution_time > 0:
            for func_stats in stats.values():
                func_stats['percent_total'] = (func_stats['total_time'] / total_execution_time) * 100
        
        return stats
    
    def print_statistics(self, top_n: int = 10):
        """Print top N functions by total time."""
        stats = self.get_statistics()
        
        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        print(f"\nTop {top_n} Functions by Total Time:")
        print("-" * 80)
        print(f"{'Function':<40} {'Calls':<8} {'Total (s)':<12} {'Avg (s)':<12} {'%':<8}")
        print("-" * 80)
        
        for func_name, func_stats in sorted_stats[:top_n]:
            print(f"{func_name:<40} {func_stats['call_count']:<8} "
                  f"{func_stats['total_time']:<12.3f} {func_stats['avg_time']:<12.6f} "
                  f"{func_stats['percent_total']:<8.1f}")


class AlgorithmProfiler:
    """Comprehensive profiling for algorithm execution."""
    
    def __init__(self, output_dir: str = "profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.function_profiler = FunctionProfiler()
        
    def profile_execution(self, func: Callable, *args, **kwargs) -> Any:
        """Profile function execution with cProfile."""
        pr = cProfile.Profile()
        
        logger.info(f"Starting detailed profiling of {func.__name__}")
        
        # Run with profiling
        pr.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            pr.disable()
        
        # Save profile data
        profile_file = self.output_dir / f"{func.__name__}_profile.prof"
        pr.dump_stats(str(profile_file))
        
        # Generate text report
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative').print_stats(20)
        
        report_file = self.output_dir / f"{func.__name__}_profile.txt"
        with open(report_file, 'w') as f:
            f.write(s.getvalue())
        
        logger.info(f"Profiling completed. Reports saved to {self.output_dir}")
        
        return result
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Any:
        """Profile memory usage during execution."""
        try:
            from memory_profiler import profile as memory_profile
            
            # Create a wrapper function for memory profiling
            @memory_profile
            def profiled_func():
                return func(*args, **kwargs)
            
            # Redirect output to file
            output_file = self.output_dir / f"{func.__name__}_memory.txt"
            
            import sys
            original_stdout = sys.stdout
            try:
                with open(output_file, 'w') as f:
                    sys.stdout = f
                    result = profiled_func()
            finally:
                sys.stdout = original_stdout
            
            logger.info(f"Memory profiling completed. Report saved to {output_file}")
            return result
            
        except ImportError:
            logger.warning("memory_profiler not available. Skipping memory profiling.")
            return func(*args, **kwargs)
    
    def profile_line_by_line(self, func: Callable, *args, **kwargs) -> Any:
        """Profile function line by line."""
        try:
            profiler = line_profiler.LineProfiler()
            profiler.add_function(func)
            
            # Run with line profiling
            profiler.enable_by_count()
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable_by_count()
            
            # Save line-by-line report
            output_file = self.output_dir / f"{func.__name__}_lines.txt"
            with open(output_file, 'w') as f:
                profiler.print_stats(stream=f)
            
            logger.info(f"Line profiling completed. Report saved to {output_file}")
            return result
            
        except ImportError:
            logger.warning("line_profiler not available. Skipping line profiling.")
            return func(*args, **kwargs)


class PerformanceComparison:
    """Compare performance between different algorithms or implementations."""
    
    def __init__(self):
        self.results = {}
        
    def compare_algorithms(self, 
                          algorithms: Dict[str, Callable],
                          problem_instance,
                          num_runs: int = 3) -> Dict[str, Dict[str, float]]:
        """Compare multiple algorithms on the same problem."""
        logger.info(f"Comparing {len(algorithms)} algorithms ({num_runs} runs each)")
        
        comparison_results = {}
        
        for alg_name, alg_factory in algorithms.items():
            logger.info(f"Testing {alg_name}...")
            
            run_times = []
            makespans = []
            
            for run in range(num_runs):
                algorithm = alg_factory()
                
                start_time = time.perf_counter()
                try:
                    solution = algorithm.solve(problem_instance)
                    end_time = time.perf_counter()
                    
                    execution_time = end_time - start_time
                    makespan = getattr(solution, 'makespan', float('inf'))
                    
                    run_times.append(execution_time)
                    makespans.append(makespan)
                    
                except Exception as e:
                    logger.error(f"Algorithm {alg_name} run {run} failed: {e}")
                    run_times.append(float('inf'))
                    makespans.append(float('inf'))
            
            # Calculate statistics
            valid_times = [t for t in run_times if t != float('inf')]
            valid_makespans = [m for m in makespans if m != float('inf')]
            
            comparison_results[alg_name] = {
                'avg_time': sum(valid_times) / len(valid_times) if valid_times else float('inf'),
                'min_time': min(valid_times) if valid_times else float('inf'),
                'max_time': max(valid_times) if valid_times else float('inf'),
                'avg_makespan': sum(valid_makespans) / len(valid_makespans) if valid_makespans else float('inf'),
                'best_makespan': min(valid_makespans) if valid_makespans else float('inf'),
                'worst_makespan': max(valid_makespans) if valid_makespans else float('inf'),
                'success_rate': len(valid_times) / num_runs
            }
        
        self.results[f"comparison_{int(time.time())}"] = comparison_results
        return comparison_results
    
    def print_comparison(self, results: Dict[str, Dict[str, float]]):
        """Print formatted comparison results."""
        print("\nAlgorithm Performance Comparison")
        print("=" * 80)
        
        print(f"{'Algorithm':<20} {'Avg Time (s)':<12} {'Best Makespan':<15} {'Success Rate':<12}")
        print("-" * 80)
        
        # Sort by average makespan
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_makespan'])
        
        for alg_name, stats in sorted_results:
            print(f"{alg_name:<20} {stats['avg_time']:<12.3f} "
                  f"{stats['best_makespan']:<15.2f} {stats['success_rate']:<12.1%}")


# Decorators for easy profiling
def profile_time(func: Callable) -> Callable:
    """Simple timing decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            logger.info(f"{func.__name__} executed in {end_time - start_time:.3f} seconds")
    
    return wrapper


def profile_memory(func: Callable) -> Callable:
    """Memory usage profiling decorator."""
    @functools.wraps(func) 
    def wrapper(*args, **kwargs):
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            logger.info(f"{func.__name__} memory usage: {mem_diff:+.1f} MB "
                       f"(before: {mem_before:.1f} MB, after: {mem_after:.1f} MB)")
            
            return result
        except ImportError:
            logger.warning("psutil not available for memory profiling")
            return func(*args, **kwargs)
    
    return wrapper


class ProfilerContext:
    """Context manager for profiling code blocks."""
    
    def __init__(self, name: str, profiler_type: str = "time"):
        self.name = name
        self.profiler_type = profiler_type
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        if self.profiler_type in ["time", "both"]:
            self.start_time = time.perf_counter()
            
        if self.profiler_type in ["memory", "both"]:
            try:
                import psutil
                process = psutil.Process()
                self.start_memory = process.memory_info().rss / 1024 / 1024
            except ImportError:
                self.start_memory = None
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.perf_counter()
            duration = end_time - self.start_time
            logger.info(f"{self.name} completed in {duration:.3f} seconds")
            
        if self.start_memory is not None:
            try:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024
                memory_diff = end_memory - self.start_memory
                logger.info(f"{self.name} memory usage: {memory_diff:+.1f} MB")
            except ImportError:
                pass


# Example usage
if __name__ == "__main__":
    # Test function profiler
    profiler = FunctionProfiler()
    
    @profiler.profile_function
    def test_function(n):
        return sum(range(n))
    
    # Test profiling
    for _ in range(5):
        test_function(1000)
    
    profiler.print_statistics()