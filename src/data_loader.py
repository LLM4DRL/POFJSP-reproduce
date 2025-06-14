"""
Data Loader for POFJSP Instances

This module provides utilities to load generated POFJSP datasets from parquet
and JSON files, converting them back into ProblemInstance objects for algorithm evaluation.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import namedtuple

from algorithms import Operation, ProblemInstance


class POFJSPDataLoader:
    """Loader for POFJSP datasets generated by POFJSPDataGenerator."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets in the data directory."""
        datasets = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and (item / 'metadata.parquet').exists():
                datasets.append(item.name)
        return sorted(datasets)
    
    def load_dataset_summary(self, dataset_name: str) -> Optional[Dict]:
        """Load dataset summary information."""
        summary_file = self.data_dir / dataset_name / 'dataset_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return None
    
    def load_metadata(self, dataset_name: str) -> pd.DataFrame:
        """Load metadata for all instances in a dataset."""
        metadata_file = self.data_dir / dataset_name / 'metadata.parquet'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        return pd.read_parquet(metadata_file)
    
    def load_problems_dataframe(self, dataset_name: str) -> pd.DataFrame:
        """Load problems dataframe (serialized format)."""
        problems_file = self.data_dir / dataset_name / 'problems.parquet'
        if not problems_file.exists():
            raise FileNotFoundError(f"Problems file not found: {problems_file}")
        
        return pd.read_parquet(problems_file)
    
    def load_instance_from_json(self, dataset_name: str, instance_id: str) -> ProblemInstance:
        """Load a specific instance from JSON file."""
        json_file = self.data_dir / dataset_name / f"{instance_id}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Instance file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return self._convert_json_to_problem_instance(data)
    
    def load_instance_from_dataframe(self, problems_df: pd.DataFrame, instance_id: str) -> ProblemInstance:
        """Load a specific instance from the problems dataframe."""
        instance_row = problems_df[problems_df['instance_id'] == instance_id]
        if instance_row.empty:
            raise ValueError(f"Instance not found: {instance_id}")
        
        row = instance_row.iloc[0]
        
        # Reconstruct problem data
        problem_data = {
            'num_jobs': row['num_jobs'],
            'num_machines': row['num_machines'],
            'num_operations_per_job': json.loads(row['num_operations_per_job']),
            'processing_times': json.loads(row['processing_times']),
            'predecessors_map': json.loads(row['predecessors_map']),
            'successors_map': json.loads(row['successors_map'])
        }
        
        data = {'problem': problem_data}
        return self._convert_json_to_problem_instance(data)
    
    def _convert_json_to_problem_instance(self, data: Dict) -> ProblemInstance:
        """Convert JSON data back to ProblemInstance object."""
        problem = data['problem']
        
        # Convert processing times back to numpy arrays
        processing_times = []
        for job_times in problem['processing_times']:
            job_array = np.array(job_times)
            # Convert None back to np.inf
            job_array = np.where(np.isnan(job_array), np.inf, job_array)
            processing_times.append(job_array)
        
        # Convert precedence maps back to Operation objects
        predecessors_map = {}
        successors_map = {}
        
        # Parse operation strings like "(0,1)" back to Operation objects
        for op_str, pred_list in problem['predecessors_map'].items():
            # Parse "(job_idx,op_idx)" format
            op_str_clean = op_str.strip('()')
            job_idx, op_idx = map(int, op_str_clean.split(','))
            op = Operation(job_idx, op_idx)
            
            pred_ops = set()
            for pred_str in pred_list:
                pred_str_clean = pred_str.strip('()')
                pred_job_idx, pred_op_idx = map(int, pred_str_clean.split(','))
                pred_ops.add(Operation(pred_job_idx, pred_op_idx))
            
            predecessors_map[op] = pred_ops
        
        for op_str, succ_list in problem['successors_map'].items():
            op_str_clean = op_str.strip('()')
            job_idx, op_idx = map(int, op_str_clean.split(','))
            op = Operation(job_idx, op_idx)
            
            succ_ops = set()
            for succ_str in succ_list:
                succ_str_clean = succ_str.strip('()')
                succ_job_idx, succ_op_idx = map(int, succ_str_clean.split(','))
                succ_ops.add(Operation(succ_job_idx, succ_op_idx))
            
            successors_map[op] = succ_ops
        
        return ProblemInstance(
            num_jobs=problem['num_jobs'],
            num_machines=problem['num_machines'],
            num_operations_per_job=problem['num_operations_per_job'],
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
    
    def load_instances_by_criteria(self, 
                                   dataset_name: str, 
                                   size_config: Optional[str] = None,
                                   precedence_pattern: Optional[str] = None,
                                   complexity_range: Optional[Tuple[float, float]] = None,
                                   max_instances: Optional[int] = None) -> List[Tuple[str, ProblemInstance]]:
        """
        Load instances matching specific criteria.
        
        Args:
            dataset_name: Name of the dataset to load from
            size_config: Filter by size configuration (e.g., 'small', 'medium')
            precedence_pattern: Filter by precedence pattern (e.g., 'mixed', 'parallel')
            complexity_range: Filter by complexity score range (min, max)
            max_instances: Maximum number of instances to load
        
        Returns:
            List of (instance_id, ProblemInstance) tuples
        """
        metadata_df = self.load_metadata(dataset_name)
        problems_df = self.load_problems_dataframe(dataset_name)
        
        # Apply filters
        filtered_df = metadata_df.copy()
        
        if size_config:
            filtered_df = filtered_df[filtered_df['size_config'] == size_config]
        
        if precedence_pattern:
            filtered_df = filtered_df[filtered_df['precedence_pattern'] == precedence_pattern]
        
        if complexity_range:
            min_complexity, max_complexity = complexity_range
            filtered_df = filtered_df[
                (filtered_df['complexity_score'] >= min_complexity) &
                (filtered_df['complexity_score'] <= max_complexity)
            ]
        
        if max_instances:
            filtered_df = filtered_df.head(max_instances)
        
        # Load the selected instances
        instances = []
        for _, row in filtered_df.iterrows():
            instance_id = row['instance_id']
            try:
                problem_instance = self.load_instance_from_dataframe(problems_df, instance_id)
                instances.append((instance_id, problem_instance))
            except Exception as e:
                print(f"Warning: Failed to load instance {instance_id}: {e}")
        
        return instances
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict:
        """Get comprehensive statistics about a dataset."""
        metadata_df = self.load_metadata(dataset_name)
        
        stats = {
            'total_instances': len(metadata_df),
            'size_configs': metadata_df['size_config'].value_counts().to_dict(),
            'precedence_patterns': metadata_df['precedence_pattern'].value_counts().to_dict(),
            'time_distributions': metadata_df['time_distribution'].value_counts().to_dict(),
            'complexity_stats': {
                'min': metadata_df['complexity_score'].min(),
                'max': metadata_df['complexity_score'].max(),
                'mean': metadata_df['complexity_score'].mean(),
                'std': metadata_df['complexity_score'].std(),
                'median': metadata_df['complexity_score'].median()
            },
            'operations_stats': {
                'min': metadata_df['total_operations'].min(),
                'max': metadata_df['total_operations'].max(),
                'mean': metadata_df['total_operations'].mean(),
                'std': metadata_df['total_operations'].std(),
                'median': metadata_df['total_operations'].median()
            }
        }
        
        return stats


def create_sample_loader_script():
    """Create a sample script demonstrating data loading usage."""
    script_content = '''#!/usr/bin/env python3
"""
Sample script demonstrating POFJSP data loading and algorithm evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import POFJSPDataLoader
from algorithms import iaoa_gns_algorithm

def main():
    # Initialize data loader
    loader = POFJSPDataLoader('./data')
    
    # List available datasets
    datasets = loader.list_available_datasets()
    print(f"Available datasets: {datasets}")
    
    if not datasets:
        print("No datasets found. Run dataset generation first.")
        return
    
    # Use the first available dataset
    dataset_name = datasets[0]
    print(f"\\nUsing dataset: {dataset_name}")
    
    # Get dataset statistics
    stats = loader.get_dataset_statistics(dataset_name)
    print(f"Dataset statistics:")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Complexity range: {stats['complexity_stats']['min']:.1f} - {stats['complexity_stats']['max']:.1f}")
    print(f"  Operations range: {stats['operations_stats']['min']} - {stats['operations_stats']['max']}")
    
    # Load some small instances for testing
    small_instances = loader.load_instances_by_criteria(
        dataset_name=dataset_name,
        size_config='small',
        max_instances=3
    )
    
    print(f"\\nLoaded {len(small_instances)} small instances for testing:")
    
    # Test algorithm on loaded instances
    for instance_id, problem_instance in small_instances:
        print(f"\\nTesting instance: {instance_id}")
        print(f"  Jobs: {problem_instance.num_jobs}, Machines: {problem_instance.num_machines}")
        print(f"  Total operations: {problem_instance.total_operations}")
        
        # Run algorithm with small parameters for quick test
        try:
            solution = iaoa_gns_algorithm(problem_instance, pop_size=20, max_iterations=10)
            print(f"  Best makespan: {solution.makespan}")
        except Exception as e:
            print(f"  Error running algorithm: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open('examples/load_and_test_datasets.py', 'w') as f:
        f.write(script_content)
    
    print("Created sample loader script: examples/load_and_test_datasets.py")


if __name__ == "__main__":
    # Example usage
    try:
        loader = POFJSPDataLoader('./data')
        datasets = loader.list_available_datasets()
        print(f"Available datasets: {datasets}")
        
        if datasets:
            dataset_name = datasets[0]
            stats = loader.get_dataset_statistics(dataset_name)
            print(f"\\nStatistics for {dataset_name}:")
            print(f"  Total instances: {stats['total_instances']}")
            print(f"  Size configs: {stats['size_configs']}")
            print(f"  Complexity range: {stats['complexity_stats']['min']:.1f} - {stats['complexity_stats']['max']:.1f}")
        
        # Create sample script
        create_sample_loader_script()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to generate datasets first using: python scripts/generate_datasets.py --dataset development") 