#!/usr/bin/env python3
"""
Script to visualize dataset statistics and distribution.

This script provides a quick way to analyze and visualize a POFJSP dataset
without running the algorithm.

Usage:
    python scripts/visualize_dataset.py --dataset development
    python scripts/visualize_dataset.py --dataset benchmark --output figures/benchmark_analysis
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import POFJSPDataLoader
from visualization.analysis import visualize_dataset_distribution, plot_complexity_characteristics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize POFJSP dataset statistics')
    parser.add_argument('--dataset', required=True, help='Name of the dataset to visualize')
    parser.add_argument('--output', default=None, help='Output directory for visualizations')
    parser.add_argument('--formats', default='png,pdf', help='Output formats (comma-separated)')
    args = parser.parse_args()
    
    # Configure paths
    data_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'figures', 
                                     'dataset', args.dataset))
    
    # Parse formats
    file_formats = args.formats.split(',')
    
    print(f"Visualizing dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset metadata
    try:
        loader = POFJSPDataLoader(str(data_dir))
        metadata_df = loader.load_metadata(args.dataset)
        
        # Get dataset statistics
        stats = loader.get_dataset_statistics(args.dataset)
        
        # Print basic statistics
        print(f"\nDataset Statistics:")
        print(f"  Total instances: {stats['total_instances']}")
        print(f"  Size configurations: {', '.join(f'{k}:{v}' for k, v in stats['size_configs'].items())}")
        print(f"  Precedence patterns: {', '.join(f'{k}:{v}' for k, v in stats['precedence_patterns'].items())}")
        print(f"  Complexity range: {stats['complexity_stats']['min']:.1f} - {stats['complexity_stats']['max']:.1f}")
        print(f"  Operations range: {stats['operations_stats']['min']} - {stats['operations_stats']['max']}")
        
        # Generate visualizations
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating dataset distribution visualization...")
        visualize_dataset_distribution(
            metadata_df=metadata_df,
            output_dir=str(output_dir),
            file_formats=file_formats
        )
        
        print("Generating complexity characteristics visualization...")
        plot_complexity_characteristics(
            metadata_df=metadata_df,
            output_dir=str(output_dir),
            file_formats=file_formats
        )
        
        print(f"\nVisualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error visualizing dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 