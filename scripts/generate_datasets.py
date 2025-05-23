#!/usr/bin/env python3
"""
Dataset Generation Script for POFJSP Instances

This script generates comprehensive datasets for evaluating the IAOA+GNS algorithm
performance on Partially Ordered Flexible Job Shop Scheduling Problems.

Usage:
    python scripts/generate_datasets.py --dataset small
    python scripts/generate_datasets.py --dataset medium  
    python scripts/generate_datasets.py --dataset large
    python scripts/generate_datasets.py --dataset all
    python scripts/generate_datasets.py --custom
"""

import argparse
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import POFJSPDataGenerator, create_standard_configurations


def create_development_dataset():
    """Create a small dataset for development and testing."""
    configurations = [
        # Small instances for quick testing
        {
            'name': 'dev_tiny_seq',
            'size_config': 'tiny',
            'precedence_pattern': 'sequential',
            'time_distribution': 'uniform',
            'machine_capability_prob': 0.8
        },
        {
            'name': 'dev_small_mixed',
            'size_config': 'small',
            'precedence_pattern': 'mixed',
            'time_distribution': 'uniform',
            'machine_capability_prob': 0.8
        },
        {
            'name': 'dev_small_parallel',
            'size_config': 'small',
            'precedence_pattern': 'parallel',
            'time_distribution': 'normal',
            'machine_capability_prob': 0.7
        }
    ]
    return configurations


def create_benchmark_dataset():
    """Create a medium-sized dataset for benchmarking."""
    configurations = []
    
    # Systematic combinations for benchmarking
    sizes = ['small', 'medium']
    patterns = ['sequential', 'parallel', 'assembly', 'mixed']
    distributions = ['uniform', 'normal']
    capabilities = [0.7, 0.8, 0.9]
    
    config_id = 0
    for size in sizes:
        for pattern in patterns:
            for dist in distributions:
                for cap in capabilities:
                    config = {
                        'name': f'bench_{config_id:03d}_{size}_{pattern}_{dist}_{int(cap*100)}',
                        'size_config': size,
                        'precedence_pattern': pattern,
                        'time_distribution': dist,
                        'machine_capability_prob': cap
                    }
                    configurations.append(config)
                    config_id += 1
    
    return configurations


def create_performance_dataset():
    """Create a large dataset for comprehensive performance evaluation."""
    configurations = []
    
    # Extended combinations for performance testing
    sizes = ['medium', 'large', 'xlarge']
    patterns = ['sequential', 'parallel', 'assembly', 'mixed', 'tree']
    distributions = ['uniform', 'normal', 'exponential', 'bimodal']
    capabilities = [0.6, 0.7, 0.8, 0.9, 1.0]
    
    config_id = 0
    for size in sizes:
        for pattern in patterns:
            for dist in distributions:
                for cap in capabilities:
                    config = {
                        'name': f'perf_{config_id:03d}_{size}_{pattern}_{dist}_{int(cap*100)}',
                        'size_config': size,
                        'precedence_pattern': pattern,
                        'time_distribution': dist,
                        'machine_capability_prob': cap
                    }
                    configurations.append(config)
                    config_id += 1
    
    return configurations


def create_stress_test_dataset():
    """Create very large instances for stress testing."""
    configurations = [
        # Extra large instances
        {
            'name': 'stress_xlarge_mixed_uniform',
            'size_config': 'xlarge',
            'precedence_pattern': 'mixed',
            'time_distribution': 'uniform',
            'machine_capability_prob': 0.8
        },
        {
            'name': 'stress_xlarge_assembly_normal',
            'size_config': 'xlarge',
            'precedence_pattern': 'assembly',
            'time_distribution': 'normal',
            'machine_capability_prob': 0.7
        },
        {
            'name': 'stress_xxlarge_mixed_uniform',
            'size_config': 'xxlarge',
            'precedence_pattern': 'mixed',
            'time_distribution': 'uniform',
            'machine_capability_prob': 0.8
        },
        # Complex patterns
        {
            'name': 'stress_xlarge_tree_bimodal',
            'size_config': 'xlarge',
            'precedence_pattern': 'tree',
            'time_distribution': 'bimodal',
            'machine_capability_prob': 0.6
        }
    ]
    return configurations


def create_algorithmic_study_dataset():
    """Create specialized dataset for algorithmic studies."""
    configurations = []
    
    # Focus on specific algorithmic challenges
    
    # 1. High precedence density (many dependencies)
    for size in ['medium', 'large']:
        config = {
            'name': f'algo_high_precedence_{size}',
            'size_config': size,
            'precedence_pattern': 'tree',  # Creates many dependencies
            'time_distribution': 'uniform',
            'machine_capability_prob': 0.8
        }
        configurations.append(config)
    
    # 2. Low machine flexibility (bottlenecks)
    for size in ['medium', 'large']:
        config = {
            'name': f'algo_low_flexibility_{size}',
            'size_config': size,
            'precedence_pattern': 'mixed',
            'time_distribution': 'uniform',
            'machine_capability_prob': 0.5  # Low flexibility
        }
        configurations.append(config)
    
    # 3. High processing time variance
    for size in ['medium', 'large']:
        config = {
            'name': f'algo_high_variance_{size}',
            'size_config': size,
            'precedence_pattern': 'mixed',
            'time_distribution': 'bimodal',  # High variance
            'machine_capability_prob': 0.8
        }
        configurations.append(config)
    
    # 4. Assembly-focused (convergence challenges)
    for size in ['medium', 'large']:
        config = {
            'name': f'algo_assembly_focus_{size}',
            'size_config': size,
            'precedence_pattern': 'assembly',
            'time_distribution': 'normal',
            'machine_capability_prob': 0.7
        }
        configurations.append(config)
    
    return configurations


def generate_dataset(dataset_type: str, output_base_dir: str = './data'):
    """Generate specified dataset type."""
    
    # Create base output directory
    base_path = Path(output_base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset configurations
    dataset_configs = {
        'development': {
            'configs': create_development_dataset(),
            'instances_per_config': 5,
            'output_subdir': 'development'
        },
        'benchmark': {
            'configs': create_benchmark_dataset(),
            'instances_per_config': 10,
            'output_subdir': 'benchmark'
        },
        'performance': {
            'configs': create_performance_dataset(),
            'instances_per_config': 15,
            'output_subdir': 'performance'
        },
        'stress': {
            'configs': create_stress_test_dataset(),
            'instances_per_config': 5,
            'output_subdir': 'stress_test'
        },
        'algorithmic': {
            'configs': create_algorithmic_study_dataset(),
            'instances_per_config': 20,
            'output_subdir': 'algorithmic_study'
        }
    }
    
    if dataset_type not in dataset_configs:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    config_info = dataset_configs[dataset_type]
    output_dir = base_path / config_info['output_subdir']
    
    print(f"Generating {dataset_type} dataset...")
    print(f"Configurations: {len(config_info['configs'])}")
    print(f"Instances per config: {config_info['instances_per_config']}")
    print(f"Total instances: {len(config_info['configs']) * config_info['instances_per_config']}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Initialize generator
    generator = POFJSPDataGenerator(random_seed=42)
    
    # Start timing
    start_time = time.time()
    
    # Generate dataset
    instances = generator.generate_dataset(
        configurations=config_info['configs'],
        instances_per_config=config_info['instances_per_config'],
        output_dir=str(output_dir)
    )
    
    # End timing
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"\n✅ Dataset generation completed!")
    print(f"⏱️  Total time: {generation_time:.2f} seconds")
    print(f"📊 Generated {len(instances)} instances")
    print(f"💾 Saved to: {output_dir}")
    
    # Display summary statistics
    if instances:
        complexity_scores = [inst['metadata']['complexity_score'] for inst in instances]
        total_ops = [inst['metadata']['total_operations'] for inst in instances]
        
        print(f"\n📈 Dataset Statistics:")
        print(f"   Complexity scores: {min(complexity_scores):.1f} - {max(complexity_scores):.1f}")
        print(f"   Operations range: {min(total_ops)} - {max(total_ops)}")
        print(f"   Average complexity: {sum(complexity_scores)/len(complexity_scores):.1f}")
    
    return instances


def create_custom_dataset():
    """Interactive creation of custom dataset."""
    print("🛠️  Custom Dataset Generator")
    print("="*50)
    
    # Get user preferences
    print("Available size configurations:")
    sizes = ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge']
    for i, size in enumerate(sizes):
        print(f"  {i+1}. {size}")
    
    while True:
        try:
            size_choice = int(input("Select size configuration (1-6): ")) - 1
            if 0 <= size_choice < len(sizes):
                selected_size = sizes[size_choice]
                break
            else:
                print("Invalid choice. Please select 1-6.")
        except ValueError:
            print("Please enter a number.")
    
    print(f"\nSelected size: {selected_size}")
    
    # Get precedence pattern
    print("\nAvailable precedence patterns:")
    patterns = ['sequential', 'parallel', 'assembly', 'mixed', 'tree']
    for i, pattern in enumerate(patterns):
        print(f"  {i+1}. {pattern}")
    
    while True:
        try:
            pattern_choice = int(input("Select precedence pattern (1-5): ")) - 1
            if 0 <= pattern_choice < len(patterns):
                selected_pattern = patterns[pattern_choice]
                break
            else:
                print("Invalid choice. Please select 1-5.")
        except ValueError:
            print("Please enter a number.")
    
    print(f"Selected pattern: {selected_pattern}")
    
    # Get number of instances
    while True:
        try:
            num_instances = int(input("\nNumber of instances to generate (1-100): "))
            if 1 <= num_instances <= 100:
                break
            else:
                print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Please enter a number.")
    
    # Create configuration
    config = {
        'name': f'custom_{selected_size}_{selected_pattern}',
        'size_config': selected_size,
        'precedence_pattern': selected_pattern,
        'time_distribution': 'uniform',
        'machine_capability_prob': 0.8
    }
    
    print(f"\n🎯 Generating {num_instances} instances with configuration:")
    print(f"   Size: {selected_size}")
    print(f"   Pattern: {selected_pattern}")
    print("-" * 40)
    
    # Generate
    generator = POFJSPDataGenerator(random_seed=42)
    output_dir = './data/custom'
    
    instances = generator.generate_dataset(
        configurations=[config],
        instances_per_config=num_instances,
        output_dir=output_dir
    )
    
    print(f"\n✅ Custom dataset generated: {len(instances)} instances")
    print(f"💾 Saved to: {output_dir}")
    
    return instances


def main():
    parser = argparse.ArgumentParser(description='Generate POFJSP datasets')
    parser.add_argument('--dataset', 
                       choices=['development', 'benchmark', 'performance', 'stress', 'algorithmic', 'all'],
                       help='Type of dataset to generate')
    parser.add_argument('--custom', action='store_true',
                       help='Create custom dataset interactively')
    parser.add_argument('--output', default='./data',
                       help='Output directory (default: ./data)')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.custom:
        print("❌ Please specify --dataset or --custom")
        parser.print_help()
        return
    
    if args.custom:
        create_custom_dataset()
        return
    
    if args.dataset == 'all':
        dataset_types = ['development', 'benchmark', 'performance', 'algorithmic']
        print("🚀 Generating all dataset types...")
        
        for dataset_type in dataset_types:
            print(f"\n{'='*60}")
            print(f"GENERATING {dataset_type.upper()} DATASET")
            print(f"{'='*60}")
            generate_dataset(dataset_type, args.output)
            
        print(f"\n🎉 All datasets generated successfully!")
        
    else:
        generate_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main() 