#!/usr/bin/env python3
"""
Comprehensive POFJSP Algorithm Benchmark for CCF-A Conference Paper

This script benchmarks all available POFJSP algorithms across multiple problem sizes
from 5x5 to 150x150, performs statistical analysis, and generates publication-quality
plots and results suitable for a CCF-A conference paper submission.
"""

import sys
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.algorithms.factory import AlgorithmFactory
from src.problems.problem_instance import ProblemInstance
from src.exceptions import AlgorithmError

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result for a single algorithm-problem combination."""
    algorithm: str
    problem_size: str
    n_jobs: int
    n_machines: int
    makespan: float
    execution_time: float
    iterations: int
    convergence: bool
    error: Optional[str] = None
    run_id: int = 0
    additional_metrics: Dict[str, Any] = None

@dataclass
class ProblemGenerator:
    """Generates POFJSP problem instances of various sizes and characteristics."""
    
    @staticmethod
    def generate_problem(n_jobs: int, n_machines: int, 
                        complexity: str = 'medium', 
                        precedence_density: float = 0.3,
                        processing_time_range: Tuple[int, int] = (1, 10),
                        seed: int = None) -> ProblemInstance:
        """
        Generate a POFJSP problem instance.
        
        Args:
            n_jobs: Number of jobs
            n_machines: Number of machines
            complexity: Problem complexity ('simple', 'medium', 'complex')
            precedence_density: Density of precedence constraints
            processing_time_range: Range for processing times
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Define operations per job based on complexity
        ops_per_job = {
            'simple': max(2, n_machines // 2),
            'medium': max(3, n_machines // 2 + 1),
            'complex': max(4, n_machines // 2 + 2)
        }
        
        operations_count = ops_per_job[complexity]
        
        jobs = []
        op_id = 0
        
        for job_id in range(n_jobs):
            operations = []
            job_ops = []
            
            for op_idx in range(operations_count):
                # Generate processing times (0 means cannot process on that machine)
                processing_times = []
                for m in range(n_machines):
                    if random.random() < 0.7:  # 70% chance machine can process operation
                        processing_times.append(random.randint(*processing_time_range))
                    else:
                        processing_times.append(0)  # Machine cannot process this operation
                
                # Ensure at least one machine can process each operation
                if all(t == 0 for t in processing_times):
                    processing_times[random.randint(0, n_machines - 1)] = random.randint(*processing_time_range)
                
                # Generate precedence constraints
                precedence = []
                if op_idx > 0 and random.random() < precedence_density:
                    # Add precedence to previous operations in same job
                    for prev_op_idx in range(op_idx):
                        if random.random() < 0.5:  # 50% chance to depend on each previous op
                            precedence.append(job_ops[prev_op_idx])
                
                op = {
                    "id": op_id,
                    "processing_times": processing_times,
                    "precedence": precedence
                }
                
                operations.append(op)
                job_ops.append(op_id)
                op_id += 1
            
            jobs.append({
                "id": job_id,
                "operations": operations
            })
        
        problem_data = {
            "num_jobs": n_jobs,
            "num_machines": n_machines,
            "jobs": jobs
        }
        
        return ProblemInstance.from_dict(problem_data)

class ComprehensiveBenchmark:
    """Comprehensive benchmarking system for POFJSP algorithms."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define problem sizes to benchmark
        self.problem_sizes = [
            (5, 5), (10, 10), (15, 15), (20, 20), (30, 30),
            (40, 40), (50, 50), (60, 60), (70, 70), (80, 80),
            (90, 90), (100, 100), (120, 120), (150, 150)
        ]
        
        # Define algorithms to benchmark (excluding RL PPO as requested)
        self.algorithms = [
            'genetic', 'simulated_annealing', 'aco', 'pso',
            'differential_evolution', 'vns', 'tabu_search',
            'hybrid_ga_ls', 'memory_sa', 'iaoa_gns',
            'greedy', 'random'
        ]
        
        # Dispatching rules
        self.dispatching_rules = ['SPT', 'LPT', 'EST', 'LST', 'FIFO']
        
        # Results storage
        self.results = []
        
        # Configure timeouts based on problem size
        self.timeout_config = {
            (5, 5): 30, (10, 10): 60, (15, 15): 120, (20, 20): 180,
            (30, 30): 300, (40, 40): 450, (50, 50): 600, (60, 60): 750,
            (70, 70): 900, (80, 80): 1200, (90, 90): 1500, (100, 100): 1800,
            (120, 120): 2400, (150, 150): 3600
        }
    
    def get_timeout(self, problem_size: Tuple[int, int]) -> float:
        """Get appropriate timeout for problem size."""
        return self.timeout_config.get(problem_size, 3600.0)
    
    def create_algorithm_instances(self, problem_size: Tuple[int, int]) -> Dict[str, Any]:
        """Create algorithm instances with size-appropriate parameters."""
        n_jobs, n_machines = problem_size
        problem_complexity = n_jobs * n_machines
        
        algorithms = {}
        
        # Scale parameters based on problem complexity
        if problem_complexity <= 100:  # Small problems
            scale_factor = 1.0
        elif problem_complexity <= 1000:  # Medium problems
            scale_factor = 1.5
        else:  # Large problems
            scale_factor = 2.0
        
        try:
            # Genetic Algorithm
            algorithms['genetic'] = AlgorithmFactory.create_algorithm('genetic', {
                'pop_size': min(100, max(30, int(30 * scale_factor))),
                'generations': min(200, max(50, int(50 * scale_factor))),
                'crossover_rate': 0.8,
                'mutation_rate': 0.1
            })
            
            # Simulated Annealing
            algorithms['simulated_annealing'] = AlgorithmFactory.create_algorithm('simulated_annealing', {
                'initial_temp': 1000.0 * scale_factor,
                'cooling_rate': 0.95,
                'min_temp': 0.1,
                'max_iterations': min(20000, max(5000, int(5000 * scale_factor)))
            })
            
            # Ant Colony Optimization
            algorithms['aco'] = AlgorithmFactory.create_algorithm('aco', {
                'n_ants': min(50, max(15, int(15 * scale_factor))),
                'max_iterations': min(100, max(30, int(30 * scale_factor))),
                'alpha': 1.0,
                'beta': 2.0,
                'rho': 0.1
            })
            
            # Particle Swarm Optimization
            algorithms['pso'] = AlgorithmFactory.create_algorithm('pso', {
                'n_particles': min(50, max(20, int(20 * scale_factor))),
                'max_iterations': min(100, max(30, int(30 * scale_factor))),
                'w': 0.7,
                'c1': 2.0,
                'c2': 2.0
            })
            
            # Differential Evolution
            algorithms['differential_evolution'] = AlgorithmFactory.create_algorithm('differential_evolution', {
                'population_size': min(60, max(20, int(20 * scale_factor))),
                'max_iterations': min(100, max(30, int(30 * scale_factor))),
                'F': 0.8,
                'CR': 0.9
            })
            
            # Variable Neighborhood Search
            algorithms['vns'] = AlgorithmFactory.create_algorithm('vns', {
                'max_iterations': min(200, max(50, int(50 * scale_factor))),
                'k_max': min(5, max(3, int(3 * scale_factor)))
            })
            
            # Tabu Search
            algorithms['tabu_search'] = AlgorithmFactory.create_algorithm('tabu_search', {
                'max_iterations': min(200, max(50, int(50 * scale_factor))),
                'tabu_tenure': min(15, max(5, int(5 * scale_factor)))
            })
            
            # Hybrid GA with Local Search
            algorithms['hybrid_ga_ls'] = AlgorithmFactory.create_algorithm('hybrid_ga_ls', {
                'population_size': min(60, max(30, int(30 * scale_factor))),
                'max_generations': min(150, max(50, int(50 * scale_factor))),
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'local_search_rate': 0.3
            })
            
            # Memory-based Simulated Annealing
            algorithms['memory_sa'] = AlgorithmFactory.create_algorithm('memory_sa', {
                'initial_temp': 800.0 * scale_factor,
                'cooling_rate': 0.95,
                'min_temp': 0.1,
                'max_iterations': min(15000, max(5000, int(5000 * scale_factor))),
                'memory_size': min(30, max(10, int(10 * scale_factor)))
            })
            
            # IAOA+GNS (Advanced algorithm)
            algorithms['iaoa_gns'] = AlgorithmFactory.create_algorithm('iaoa_gns', {
                'pop_size': min(100, max(50, int(50 * scale_factor))),
                'max_iterations': min(120, max(40, int(40 * scale_factor)))
            })
            
            # Baseline algorithms
            algorithms['greedy'] = AlgorithmFactory.create_algorithm('greedy')
            algorithms['random'] = AlgorithmFactory.create_algorithm('random', {
                'num_trials': min(5000, max(1000, int(1000 * scale_factor)))
            })
            
            # Dispatching rules
            for rule in self.dispatching_rules:
                algorithms[f'dispatching_{rule.lower()}'] = AlgorithmFactory.create_algorithm('dispatching', {'rule': rule})
                
        except Exception as e:
            print(f"Warning: Could not create some algorithms for size {problem_size}: {e}")
        
        return algorithms
    
    def run_single_benchmark(self, algorithm_name: str, algorithm, problem: ProblemInstance, 
                           problem_size: Tuple[int, int], run_id: int = 0) -> BenchmarkResult:
        """Run a single algorithm on a single problem instance."""
        timeout = self.get_timeout(problem_size)
        
        try:
            start_time = time.time()
            result = algorithm.solve(problem, timeout=timeout)
            end_time = time.time()
            
            # Extract metrics
            makespan = result.makespan if result.makespan != float('inf') else np.inf
            execution_time = result.execution_time
            error = result.additional_metrics.get('error') if result.additional_metrics else None
            
            # Determine iterations and convergence
            iterations = 0
            convergence = False
            
            if result.additional_metrics:
                iterations = result.additional_metrics.get('generations', 
                           result.additional_metrics.get('iterations', 0))
                convergence = makespan != float('inf') and error is None
            
            return BenchmarkResult(
                algorithm=algorithm_name,
                problem_size=f"{problem_size[0]}x{problem_size[1]}",
                n_jobs=problem_size[0],
                n_machines=problem_size[1],
                makespan=makespan,
                execution_time=execution_time,
                iterations=iterations,
                convergence=convergence,
                error=error,
                run_id=run_id,
                additional_metrics=result.additional_metrics
            )
            
        except Exception as e:
            return BenchmarkResult(
                algorithm=algorithm_name,
                problem_size=f"{problem_size[0]}x{problem_size[1]}",
                n_jobs=problem_size[0],
                n_machines=problem_size[1],
                makespan=np.inf,
                execution_time=0.0,
                iterations=0,
                convergence=False,
                error=str(e),
                run_id=run_id
            )
    
    def run_comprehensive_benchmark(self, runs_per_config: int = 5, max_workers: int = None):
        """Run comprehensive benchmark across all algorithms and problem sizes."""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Limit for memory considerations
        
        print("=== COMPREHENSIVE POFJSP ALGORITHM BENCHMARK ===")
        print(f"Problem sizes: {len(self.problem_sizes)} sizes from 5x5 to 150x150")
        print(f"Algorithms: {len(self.algorithms)} + {len(self.dispatching_rules)} dispatching rules")
        print(f"Runs per configuration: {runs_per_config}")
        print(f"Max workers: {max_workers}")
        print(f"Results will be saved to: {self.output_dir}")
        print()
        
        total_experiments = len(self.problem_sizes) * (len(self.algorithms) + len(self.dispatching_rules)) * runs_per_config
        completed_experiments = 0
        
        for problem_size in self.problem_sizes:
            print(f"\n--- Benchmarking {problem_size[0]}x{problem_size[1]} problems ---")
            
            # Create algorithm instances for this problem size
            algorithms = self.create_algorithm_instances(problem_size)
            
            for run in range(runs_per_config):
                print(f"  Run {run + 1}/{runs_per_config}")
                
                # Generate problem instance
                problem = ProblemGenerator.generate_problem(
                    problem_size[0], problem_size[1], 
                    complexity='medium',
                    seed=42 + run  # Ensure reproducible but different problems per run
                )
                
                # Run all algorithms on this problem
                for algo_name, algorithm in algorithms.items():
                    try:
                        result = self.run_single_benchmark(
                            algo_name, algorithm, problem, problem_size, run
                        )
                        self.results.append(result)
                        
                        # Progress indicator
                        completed_experiments += 1
                        if completed_experiments % 10 == 0:
                            progress = (completed_experiments / total_experiments) * 100
                            print(f"    Progress: {completed_experiments}/{total_experiments} ({progress:.1f}%)")
                        
                    except Exception as e:
                        print(f"    Error with {algo_name}: {e}")
                        # Add failed result
                        failed_result = BenchmarkResult(
                            algorithm=algo_name,
                            problem_size=f"{problem_size[0]}x{problem_size[1]}",
                            n_jobs=problem_size[0],
                            n_machines=problem_size[1],
                            makespan=np.inf,
                            execution_time=0.0,
                            iterations=0,
                            convergence=False,
                            error=str(e),
                            run_id=run
                        )
                        self.results.append(failed_result)
                        completed_experiments += 1
        
        print(f"\nBenchmark completed! Total experiments: {completed_experiments}")
        
        # Save raw results
        self.save_results()
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Save as CSV
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to: {csv_file}")
        
        # Save as JSON for detailed analysis
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2, default=str)
        print(f"Detailed results saved to: {json_file}")
        
        return df
    
    def generate_analysis_report(self, df: pd.DataFrame = None):
        """Generate comprehensive analysis report suitable for CCF-A paper."""
        if df is None:
            df = pd.DataFrame([asdict(result) for result in self.results])
        
        print("\n=== GENERATING ANALYSIS REPORT ===")
        
        # Create analysis directory
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # 1. Performance Analysis
        self.analyze_performance(df, analysis_dir)
        
        # 2. Scalability Analysis
        self.analyze_scalability(df, analysis_dir)
        
        # 3. Algorithm Comparison
        self.analyze_algorithm_comparison(df, analysis_dir)
        
        # 4. Statistical Analysis
        self.analyze_statistics(df, analysis_dir)
        
        # 5. Generate Publication-Quality Plots
        self.generate_publication_plots(df, analysis_dir)
        
        # 6. Generate LaTeX Tables
        self.generate_latex_tables(df, analysis_dir)
        
        print(f"Analysis report generated in: {analysis_dir}")
    
    def analyze_performance(self, df: pd.DataFrame, output_dir: Path):
        """Analyze algorithm performance metrics."""
        print("  Analyzing performance metrics...")
        
        # Filter out infinite makespans for meaningful analysis
        valid_df = df[df['makespan'] != np.inf].copy()
        
        # Performance summary by algorithm
        perf_summary = valid_df.groupby('algorithm').agg({
            'makespan': ['mean', 'std', 'min', 'max', 'count'],
            'execution_time': ['mean', 'std', 'min', 'max'],
            'convergence': 'mean'
        }).round(4)
        
        perf_summary.to_csv(output_dir / "performance_summary.csv")
        
        # Success rate analysis
        success_rate = df.groupby('algorithm').agg({
            'convergence': 'mean',
            'makespan': lambda x: (x != np.inf).mean()
        }).round(4)
        success_rate.columns = ['convergence_rate', 'success_rate']
        success_rate.to_csv(output_dir / "success_rates.csv")
        
        return perf_summary, success_rate
    
    def analyze_scalability(self, df: pd.DataFrame, output_dir: Path):
        """Analyze algorithm scalability with problem size."""
        print("  Analyzing scalability...")
        
        # Create problem complexity metric
        df['complexity'] = df['n_jobs'] * df['n_machines']
        
        # Scalability analysis
        scalability = df.groupby(['algorithm', 'complexity']).agg({
            'makespan': 'mean',
            'execution_time': 'mean',
            'convergence': 'mean'
        }).reset_index()
        
        scalability.to_csv(output_dir / "scalability_analysis.csv", index=False)
        
        return scalability
    
    def analyze_algorithm_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Compare algorithms across different metrics."""
        print("  Comparing algorithms...")
        
        # Rank algorithms by performance
        valid_df = df[df['makespan'] != np.inf].copy()
        
        # Average rank across all problems
        ranking_data = []
        
        for (size, run), group in valid_df.groupby(['problem_size', 'run_id']):
            # Rank by makespan (lower is better)
            group_sorted = group.sort_values('makespan')
            for rank, (_, row) in enumerate(group_sorted.iterrows(), 1):
                ranking_data.append({
                    'algorithm': row['algorithm'],
                    'problem_size': size,
                    'run_id': run,
                    'rank': rank,
                    'makespan': row['makespan']
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        avg_ranking = ranking_df.groupby('algorithm')['rank'].mean().sort_values()
        avg_ranking.to_csv(output_dir / "algorithm_rankings.csv")
        
        return avg_ranking
    
    def analyze_statistics(self, df: pd.DataFrame, output_dir: Path):
        """Perform statistical analysis."""
        print("  Performing statistical analysis...")
        
        # Statistical tests would go here (ANOVA, post-hoc tests, etc.)
        # For now, provide descriptive statistics
        
        valid_df = df[df['makespan'] != np.inf].copy()
        
        stats_summary = valid_df.groupby('algorithm').agg({
            'makespan': ['count', 'mean', 'std', 'min', 'median', 'max'],
            'execution_time': ['mean', 'std', 'min', 'median', 'max']
        })
        
        stats_summary.to_csv(output_dir / "statistical_summary.csv")
        
        # Algorithm comparison matrix (mean makespan)
        comparison_matrix = valid_df.pivot_table(
            values='makespan', 
            index='algorithm', 
            columns='problem_size', 
            aggfunc='mean'
        )
        comparison_matrix.to_csv(output_dir / "algorithm_comparison_matrix.csv")
        
        return stats_summary, comparison_matrix
    
    def generate_publication_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate publication-quality plots."""
        print("  Generating publication-quality plots...")
        
        # Set up the plotting style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        valid_df = df[df['makespan'] != np.inf].copy()
        valid_df['complexity'] = valid_df['n_jobs'] * valid_df['n_machines']
        
        # 1. Algorithm Performance Comparison
        plt.figure(figsize=(14, 8))
        
        # Box plot of makespan by algorithm
        plt.subplot(2, 2, 1)
        top_algorithms = valid_df.groupby('algorithm')['makespan'].mean().nsmallest(10).index
        sns.boxplot(data=valid_df[valid_df['algorithm'].isin(top_algorithms)], 
                   x='algorithm', y='makespan')
        plt.xticks(rotation=45, ha='right')
        plt.title('Algorithm Performance Comparison (Top 10)')
        plt.ylabel('Makespan')
        
        # 2. Scalability Analysis
        plt.subplot(2, 2, 2)
        scalability_data = valid_df.groupby(['algorithm', 'complexity'])['makespan'].mean().reset_index()
        top_algos = ['genetic', 'simulated_annealing', 'aco', 'iaoa_gns', 'greedy']
        
        for algo in top_algos:
            if algo in scalability_data['algorithm'].values:
                algo_data = scalability_data[scalability_data['algorithm'] == algo]
                plt.plot(algo_data['complexity'], algo_data['makespan'], marker='o', label=algo)
        
        plt.xlabel('Problem Complexity (Jobs × Machines)')
        plt.ylabel('Average Makespan')
        plt.title('Scalability Analysis')
        plt.legend()
        plt.yscale('log')
        
        # 3. Execution Time Analysis
        plt.subplot(2, 2, 3)
        time_data = valid_df.groupby(['algorithm', 'complexity'])['execution_time'].mean().reset_index()
        
        for algo in top_algos:
            if algo in time_data['algorithm'].values:
                algo_data = time_data[time_data['algorithm'] == algo]
                plt.plot(algo_data['complexity'], algo_data['execution_time'], marker='s', label=algo)
        
        plt.xlabel('Problem Complexity (Jobs × Machines)')
        plt.ylabel('Average Execution Time (s)')
        plt.title('Execution Time vs Problem Complexity')
        plt.legend()
        plt.yscale('log')
        
        # 4. Success Rate Analysis
        plt.subplot(2, 2, 4)
        success_rates = df.groupby('algorithm').agg({
            'convergence': 'mean'
        }).sort_values('convergence', ascending=True)
        
        plt.barh(range(len(success_rates)), success_rates['convergence'])
        plt.yticks(range(len(success_rates)), success_rates.index)
        plt.xlabel('Success Rate')
        plt.title('Algorithm Success Rates')
        plt.xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "comprehensive_analysis.pdf", bbox_inches='tight')
        plt.close()
        
        # Additional detailed plots
        self.generate_detailed_plots(valid_df, output_dir)
    
    def generate_detailed_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate additional detailed plots."""
        
        # Performance vs Problem Size Heatmap
        plt.figure(figsize=(16, 10))
        
        # Create pivot table for heatmap
        heatmap_data = df.groupby(['algorithm', 'problem_size'])['makespan'].mean().unstack()
        
        # Select top algorithms for clarity
        top_algorithms = df.groupby('algorithm')['makespan'].mean().nsmallest(12).index
        heatmap_data_filtered = heatmap_data.loc[top_algorithms]
        
        sns.heatmap(heatmap_data_filtered, annot=True, fmt='.1f', cmap='viridis_r')
        plt.title('Algorithm Performance Across Problem Sizes (Average Makespan)')
        plt.xlabel('Problem Size')
        plt.ylabel('Algorithm')
        plt.tight_layout()
        plt.savefig(output_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Convergence Analysis
        plt.figure(figsize=(12, 8))
        convergence_data = df.groupby(['algorithm', 'complexity']).agg({
            'convergence': 'mean',
            'makespan': lambda x: (x != np.inf).mean()
        }).reset_index()
        
        plt.scatter(convergence_data['complexity'], convergence_data['convergence'], 
                   alpha=0.6, s=50)
        
        # Add algorithm labels for key points
        for _, row in convergence_data.iterrows():
            if row['convergence'] > 0.8:  # High success algorithms
                plt.annotate(row['algorithm'], 
                           (row['complexity'], row['convergence']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.xlabel('Problem Complexity')
        plt.ylabel('Convergence Rate')
        plt.title('Algorithm Convergence vs Problem Complexity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_latex_tables(self, df: pd.DataFrame, output_dir: Path):
        """Generate LaTeX tables for publication."""
        print("  Generating LaTeX tables...")
        
        valid_df = df[df['makespan'] != np.inf].copy()
        
        # Algorithm comparison table
        comparison = valid_df.groupby('algorithm').agg({
            'makespan': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'convergence': 'mean'
        }).round(4)
        
        # Flatten column names
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        
        # Generate LaTeX table
        latex_table = comparison.to_latex(
            caption="Algorithm Performance Comparison",
            label="tab:algorithm_comparison",
            column_format="l|cc|cc|c",
            escape=False
        )
        
        with open(output_dir / "algorithm_comparison_table.tex", 'w') as f:
            f.write(latex_table)
        
        # Problem size scaling table
        scaling_data = valid_df.groupby(['problem_size', 'algorithm'])['makespan'].mean().unstack()
        top_algos = valid_df.groupby('algorithm')['makespan'].mean().nsmallest(8).index
        
        scaling_latex = scaling_data[top_algos].round(2).to_latex(
            caption="Algorithm Performance Across Problem Sizes",
            label="tab:scaling_analysis",
            escape=False
        )
        
        with open(output_dir / "scaling_analysis_table.tex", 'w') as f:
            f.write(scaling_latex)

def main():
    """Main benchmark execution."""
    print("Starting Comprehensive POFJSP Algorithm Benchmark")
    print("=" * 60)
    
    # Initialize benchmark system
    benchmark = ComprehensiveBenchmark()
    
    # Run benchmark with multiple runs for statistical significance
    benchmark.run_comprehensive_benchmark(runs_per_config=3)  # Reduced for testing
    
    # Generate comprehensive analysis
    df = benchmark.save_results()
    benchmark.generate_analysis_report(df)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Results available in: {benchmark.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()