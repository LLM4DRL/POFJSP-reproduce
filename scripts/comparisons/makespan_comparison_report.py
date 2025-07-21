#!/usr/bin/env python3
"""
Makespan Comparison Report - Same Instances, Different Algorithms
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.problems.problem_instance import ProblemInstance
from src.algorithms.iaoa_gns import IAOAGNSAlgorithm

def run_makespan_comparison():
    """Run detailed makespan comparison on same instances."""
    
    print("="*80)
    print("MAKESPAN COMPARISON REPORT - SAME INSTANCES")
    print("="*80)
    
    # Load instances
    instances = []
    dataset_path = Path("../../data/flexible")
    
    if not dataset_path.exists():
        print("Dataset not found. Please run: python generate_flexible_dataset.py --custom-config")
        return
    
    # Load first 6 instances for detailed comparison
    json_files = sorted([f for f in dataset_path.glob("*.json") 
                        if not f.name.startswith("dataset_summary")])[:6]
    
    print(f"Loading {len(json_files)} instances for comparison...")
    for json_file in json_files:
        try:
            instance = ProblemInstance.from_json(str(json_file))
            instances.append((json_file.stem, instance))
            print(f"✓ {json_file.name}: {instance.num_jobs}J×{instance.num_machines}M ({instance.total_operations} ops)")
        except Exception as e:
            print(f"✗ Error loading {json_file}: {e}")
    
    if not instances:
        print("No valid instances found!")
        return
    
    # Algorithm configurations for comparison
    algorithms = [
        {
            'name': 'IAOA+GNS_Fast',
            'params': {'pop_size': 30, 'max_iterations': 40},
            'description': 'Fast configuration'
        },
        {
            'name': 'IAOA+GNS_Standard', 
            'params': {'pop_size': 50, 'max_iterations': 60},
            'description': 'Standard configuration'
        },
        {
            'name': 'IAOA+GNS_Intensive',
            'params': {'pop_size': 80, 'max_iterations': 100}, 
            'description': 'Intensive configuration'
        },
        {
            'name': 'IAOA+GNS_Premium',
            'params': {'pop_size': 100, 'max_iterations': 120},
            'description': 'Premium configuration'
        }
    ]
    
    print(f"\\nRunning comparison with {len(algorithms)} algorithm configurations...")
    print(f"Each instance will be solved {len(algorithms)} times\\n")
    
    # Results storage
    results = []
    comparison_data = {}
    
    # Run comparisons
    for i, (instance_name, instance) in enumerate(instances):
        print(f"[{i+1}/{len(instances)}] Instance: {instance_name}")
        print(f"  Size: {instance.num_jobs}J × {instance.num_machines}M ({instance.total_operations} operations)")
        
        instance_results = {}
        
        for algo in algorithms:
            print(f"    {algo['name']}: ", end="")
            
            start_time = time.time()
            try:
                algorithm = IAOAGNSAlgorithm(**algo['params'])
                solution = algorithm.solve(instance)
                solve_time = time.time() - start_time
                makespan = solution.makespan
                
                print(f"Makespan={makespan:.1f}, Time={solve_time:.2f}s")
                
                instance_results[algo['name']] = {
                    'makespan': makespan,
                    'solve_time': solve_time,
                    'success': True
                }
                
                results.append({
                    'instance': instance_name,
                    'algorithm': algo['name'],
                    'num_jobs': instance.num_jobs,
                    'num_machines': instance.num_machines,
                    'total_operations': instance.total_operations,
                    'makespan': makespan,
                    'solve_time': solve_time,
                    'pop_size': algo['params']['pop_size'],
                    'max_iterations': algo['params']['max_iterations'],
                    'success': True
                })
                
            except Exception as e:
                solve_time = time.time() - start_time
                print(f"FAILED - {e}")
                
                instance_results[algo['name']] = {
                    'makespan': float('inf'),
                    'solve_time': solve_time,
                    'success': False
                }
                
                results.append({
                    'instance': instance_name,
                    'algorithm': algo['name'],
                    'num_jobs': instance.num_jobs,
                    'num_machines': instance.num_machines,
                    'total_operations': instance.total_operations,
                    'makespan': float('inf'),
                    'solve_time': solve_time,
                    'pop_size': algo['params']['pop_size'],
                    'max_iterations': algo['params']['max_iterations'],
                    'success': False
                })
        
        comparison_data[instance_name] = instance_results
        print()
    
    # Generate detailed comparison report
    print("="*80)
    print("DETAILED MAKESPAN COMPARISON RESULTS")
    print("="*80)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    successful_df = df[df['success'] == True]
    
    # Instance-by-instance comparison table
    print("\\nMakespan Comparison Table:")
    print("-" * 120)
    header = f"{'Instance':<20} | {'Size':<10} | "
    for algo in algorithms:
        header += f"{algo['name']:<15} | "
    header += "Best | Improvement"
    print(header)
    print("-" * 120)
    
    for instance_name in comparison_data.keys():
        instance_info = next(item for item in instances if item[0] == instance_name)
        size_str = f"{instance_info[1].num_jobs}×{instance_info[1].num_machines}"
        
        row = f"{instance_name:<20} | {size_str:<10} | "
        
        makespans = []
        for algo in algorithms:
            if instance_name in comparison_data:
                makespan = comparison_data[instance_name][algo['name']]['makespan']
                if makespan != float('inf'):
                    row += f"{makespan:<15.1f} | "
                    makespans.append(makespan)
                else:
                    row += f"{'FAILED':<15} | "
        
        if makespans:
            best_makespan = min(makespans)
            worst_makespan = max(makespans)
            improvement = ((worst_makespan - best_makespan) / worst_makespan * 100) if worst_makespan > 0 else 0
            row += f"{best_makespan:<4.1f} | {improvement:<.1f}%"
        else:
            row += "N/A  | N/A"
        
        print(row)
    
    # Algorithm performance summary
    print(f"\\n{'='*80}")
    print("ALGORITHM PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for algo in algorithms:
        algo_df = successful_df[successful_df['algorithm'] == algo['name']]
        if len(algo_df) > 0:
            print(f"\\n{algo['name']} ({algo['description']}):")
            print(f"  Parameters: pop_size={algo['params']['pop_size']}, max_iter={algo['params']['max_iterations']}")
            print(f"  Success rate: {len(algo_df)}/{len(instances)} ({len(algo_df)/len(instances)*100:.1f}%)")
            print(f"  Average makespan: {algo_df['makespan'].mean():.2f} ± {algo_df['makespan'].std():.2f}")
            print(f"  Best makespan: {algo_df['makespan'].min():.1f}")
            print(f"  Worst makespan: {algo_df['makespan'].max():.1f}")
            print(f"  Average time: {algo_df['solve_time'].mean():.2f}s")
        else:
            print(f"\\n{algo['name']}: No successful runs")
    
    # Best performer per instance
    print(f"\\n{'='*80}")
    print("BEST PERFORMER PER INSTANCE")
    print(f"{'='*80}")
    
    for instance_name in comparison_data.keys():
        instance_data = comparison_data[instance_name]
        successful_algos = {k: v for k, v in instance_data.items() if v['success']}
        
        if successful_algos:
            best_algo = min(successful_algos.items(), key=lambda x: x[1]['makespan'])
            worst_algo = max(successful_algos.items(), key=lambda x: x[1]['makespan'])
            
            improvement = ((worst_algo[1]['makespan'] - best_algo[1]['makespan']) / worst_algo[1]['makespan'] * 100)
            
            print(f"{instance_name}:")
            print(f"  🏆 Best: {best_algo[0]} (Makespan: {best_algo[1]['makespan']:.1f})")
            print(f"  📊 Worst: {worst_algo[0]} (Makespan: {worst_algo[1]['makespan']:.1f})")
            print(f"  📈 Improvement: {improvement:.1f}%")
        else:
            print(f"{instance_name}: No successful algorithms")
    
    # Save results
    output_dir = "./outputs/makespan_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "makespan_comparison.csv")
    df.to_csv(results_file, index=False)
    
    # Create comparison matrix
    comparison_matrix = successful_df.pivot(index='instance', columns='algorithm', values='makespan')
    matrix_file = os.path.join(output_dir, "makespan_matrix.csv")
    comparison_matrix.to_csv(matrix_file)
    
    # Generate visualization
    if len(successful_df) > 0:
        create_comparison_plot(successful_df, output_dir)
    
    print(f"\\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Total instances tested: {len(instances)}")
    print(f"Total algorithm runs: {len(df)}")
    print(f"Successful runs: {len(successful_df)} ({len(successful_df)/len(df)*100:.1f}%)")
    
    if len(successful_df) > 0:
        print(f"Overall best makespan: {successful_df['makespan'].min():.1f}")
        print(f"Overall worst makespan: {successful_df['makespan'].max():.1f}")
        print(f"Average makespan: {successful_df['makespan'].mean():.2f}")
    
    print(f"\\nResults saved in: {output_dir}")
    print(f"  📊 Detailed results: makespan_comparison.csv")
    print(f"  📋 Comparison matrix: makespan_matrix.csv")
    print(f"  📈 Visualization: comparison_plot.png")

def create_comparison_plot(df, output_dir):
    """Create visualization of makespan comparison."""
    
    plt.figure(figsize=(14, 8))
    
    # Group by instance and algorithm
    instances = df['instance'].unique()
    algorithms = df['algorithm'].unique()
    
    x = np.arange(len(instances))
    width = 0.8 / len(algorithms)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        makespans = []
        
        for instance in instances:
            instance_data = algo_data[algo_data['instance'] == instance]
            if len(instance_data) > 0:
                makespans.append(instance_data['makespan'].iloc[0])
            else:
                makespans.append(0)
        
        plt.bar(x + i * width, makespans, width, label=algo, color=colors[i], alpha=0.8)
    
    plt.xlabel('Problem Instances', fontsize=12)
    plt.ylabel('Makespan', fontsize=12)
    plt.title('Makespan Comparison Across Algorithms', fontsize=14, fontweight='bold')
    plt.xticks(x + width * (len(algorithms) - 1) / 2, instances, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_makespan_comparison()