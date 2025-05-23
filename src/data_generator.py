"""
Data Generator for Partially Ordered Flexible Job Shop Scheduling Problems (POFJSP)

This module provides tools for generating random POFJSP instances with various
characteristics including different precedence patterns, problem sizes, and
processing time distributions.
"""

import numpy as np
import pandas as pd
import json
import random
from typing import Dict, List, Tuple, Set, Optional, Union
from collections import namedtuple
from pathlib import Path
import datetime
import uuid

# Import from our algorithms module
from algorithms import Operation, ProblemInstance

class POFJSPDataGenerator:
    """
    Generator for random POFJSP instances with configurable characteristics.
    
    Supports various precedence patterns:
    - Sequential: Linear chain of operations within jobs
    - Parallel: Operations that can be executed in parallel after common predecessors
    - Assembly: Multiple streams converging to assembly operations
    - Mixed: Combination of above patterns
    - Tree: Hierarchical dependency structure
    """
    
    def __init__(self, random_seed: int = None):
        """
        Initialize the data generator.
        
        Args:
            random_seed: Random seed for reproducible generation
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.random_seed = random_seed
        
        # Predefined problem size configurations
        self.size_configs = {
            'tiny': {'jobs': (2, 4), 'machines': (2, 3), 'ops_per_job': (2, 4)},
            'small': {'jobs': (5, 10), 'machines': (3, 6), 'ops_per_job': (3, 8)},
            'medium': {'jobs': (10, 20), 'machines': (6, 12), 'ops_per_job': (5, 12)},
            'large': {'jobs': (20, 50), 'machines': (10, 20), 'ops_per_job': (8, 15)},
            'xlarge': {'jobs': (50, 100), 'machines': (15, 30), 'ops_per_job': (10, 20)},
            'xxlarge': {'jobs': (100, 200), 'machines': (20, 50), 'ops_per_job': (12, 25)}
        }
        
        # Processing time distribution parameters
        self.time_distributions = {
            'uniform': {'low': 1, 'high': 10},
            'normal': {'mean': 5, 'std': 2, 'min_val': 1, 'max_val': 15},
            'exponential': {'scale': 3, 'min_val': 1, 'max_val': 20},
            'bimodal': {'modes': [2, 8], 'weights': [0.6, 0.4], 'spread': 1.5}
        }
    
    def generate_problem_sizes(self, config_name: str) -> Tuple[int, int, List[int]]:
        """Generate random problem dimensions based on configuration."""
        config = self.size_configs[config_name]
        
        num_jobs = random.randint(*config['jobs'])
        num_machines = random.randint(*config['machines'])
        
        # Generate operations per job with some variation
        ops_per_job = []
        for _ in range(num_jobs):
            ops_count = random.randint(*config['ops_per_job'])
            ops_per_job.append(ops_count)
        
        return num_jobs, num_machines, ops_per_job
    
    def generate_processing_times(self, 
                                  num_jobs: int, 
                                  num_machines: int, 
                                  ops_per_job: List[int],
                                  distribution: str = 'uniform',
                                  machine_capability_prob: float = 0.8) -> List[np.ndarray]:
        """
        Generate processing times with specified distribution.
        
        Args:
            num_jobs: Number of jobs
            num_machines: Number of machines
            ops_per_job: Operations per job
            distribution: Type of time distribution
            machine_capability_prob: Probability that a machine can process an operation
        """
        processing_times = []
        dist_params = self.time_distributions[distribution]
        
        for job_idx in range(num_jobs):
            job_times = np.full((ops_per_job[job_idx], num_machines), np.inf)
            
            for op_idx in range(ops_per_job[job_idx]):
                for machine_idx in range(num_machines):
                    # Determine if this machine can process this operation
                    if random.random() < machine_capability_prob:
                        # Generate processing time based on distribution
                        if distribution == 'uniform':
                            time = random.uniform(dist_params['low'], dist_params['high'])
                        elif distribution == 'normal':
                            time = np.random.normal(dist_params['mean'], dist_params['std'])
                            time = max(dist_params['min_val'], min(dist_params['max_val'], time))
                        elif distribution == 'exponential':
                            time = np.random.exponential(dist_params['scale'])
                            time = max(dist_params['min_val'], min(dist_params['max_val'], time))
                        elif distribution == 'bimodal':
                            mode = np.random.choice(dist_params['modes'], p=dist_params['weights'])
                            time = np.random.normal(mode, dist_params['spread'])
                            time = max(1, time)
                        else:
                            # Fallback to uniform distribution
                            time = random.uniform(1, 10)
                        
                        job_times[op_idx, machine_idx] = round(time, 2)
            
            # Ensure each operation can be processed on at least one machine
            for op_idx in range(ops_per_job[job_idx]):
                if np.all(np.isinf(job_times[op_idx, :])):
                    # Assign to a random machine if no machine was capable
                    random_machine = random.randint(0, num_machines - 1)
                    if distribution == 'uniform':
                        time = random.uniform(dist_params['low'], dist_params['high'])
                    else:
                        time = random.uniform(1, 10)  # Fallback
                    job_times[op_idx, random_machine] = round(time, 2)
            
            processing_times.append(job_times)
        
        return processing_times
    
    def generate_sequential_precedence(self, ops_per_job: List[int]) -> Tuple[Dict, Dict]:
        """Generate simple sequential precedence within each job."""
        predecessors_map = {}
        successors_map = {}
        
        for job_idx, num_ops in enumerate(ops_per_job):
            for op_idx in range(num_ops):
                op = Operation(job_idx, op_idx)
                
                if op_idx == 0:
                    predecessors_map[op] = set()
                else:
                    predecessors_map[op] = {Operation(job_idx, op_idx - 1)}
                
                if op_idx == num_ops - 1:
                    successors_map[op] = set()
                else:
                    successors_map[op] = {Operation(job_idx, op_idx + 1)}
        
        return predecessors_map, successors_map
    
    def generate_parallel_precedence(self, ops_per_job: List[int], 
                                     parallel_prob: float = 0.3) -> Tuple[Dict, Dict]:
        """Generate precedence with parallel operations after common predecessors."""
        predecessors_map = {}
        successors_map = {}
        
        for job_idx, num_ops in enumerate(ops_per_job):
            if num_ops <= 2:
                # Too small for parallel structure, use sequential
                return self.generate_sequential_precedence([num_ops])
            
            # Initialize all operations
            for op_idx in range(num_ops):
                op = Operation(job_idx, op_idx)
                predecessors_map[op] = set()
                successors_map[op] = set()
            
            # Create parallel structure: start -> parallel group -> end
            if num_ops >= 4:
                start_op = Operation(job_idx, 0)
                end_op = Operation(job_idx, num_ops - 1)
                
                # Middle operations can be parallel
                parallel_ops = [Operation(job_idx, i) for i in range(1, num_ops - 1)]
                
                # Randomly decide which operations are parallel
                for op in parallel_ops:
                    if random.random() < parallel_prob:
                        # This operation depends on start
                        predecessors_map[op] = {start_op}
                        successors_map[start_op].add(op)
                        
                        # End depends on this operation
                        successors_map[op].add(end_op)
                        predecessors_map[end_op].add(op)
                    else:
                        # Sequential dependency on previous operation
                        prev_op = Operation(job_idx, op.op_idx_in_job - 1)
                        predecessors_map[op] = {prev_op}
                        successors_map[prev_op].add(op)
            else:
                # Fallback to sequential for small jobs
                for op_idx in range(1, num_ops):
                    curr_op = Operation(job_idx, op_idx)
                    prev_op = Operation(job_idx, op_idx - 1)
                    predecessors_map[curr_op] = {prev_op}
                    successors_map[prev_op].add(curr_op)
        
        return predecessors_map, successors_map
    
    def generate_assembly_precedence(self, ops_per_job: List[int]) -> Tuple[Dict, Dict]:
        """Generate assembly-line precedence pattern."""
        predecessors_map = {}
        successors_map = {}
        
        for job_idx, num_ops in enumerate(ops_per_job):
            # Initialize
            for op_idx in range(num_ops):
                op = Operation(job_idx, op_idx)
                predecessors_map[op] = set()
                successors_map[op] = set()
            
            if num_ops >= 5:
                # Create assembly pattern: multiple start ops -> assembly op -> final ops
                assembly_point = num_ops // 2
                
                # First phase: independent operations
                for op_idx in range(assembly_point):
                    op = Operation(job_idx, op_idx)
                    # These are independent (no predecessors)
                
                # Assembly operation depends on all previous
                assembly_op = Operation(job_idx, assembly_point)
                for op_idx in range(assembly_point):
                    pred_op = Operation(job_idx, op_idx)
                    predecessors_map[assembly_op].add(pred_op)
                    successors_map[pred_op].add(assembly_op)
                
                # Final operations are sequential after assembly
                for op_idx in range(assembly_point + 1, num_ops):
                    curr_op = Operation(job_idx, op_idx)
                    prev_op = Operation(job_idx, op_idx - 1)
                    predecessors_map[curr_op] = {prev_op}
                    successors_map[prev_op].add(curr_op)
            else:
                # Too small for assembly, use sequential
                preds, succs = self.generate_sequential_precedence([num_ops])
                for job_idx_inner, num_ops_inner in enumerate([num_ops]):
                    for op_idx in range(num_ops_inner):
                        op = Operation(job_idx, op_idx)
                        if op in preds:
                            predecessors_map[op] = preds[op]
                        if op in succs:
                            successors_map[op] = succs[op]
        
        return predecessors_map, successors_map
    
    def generate_mixed_precedence(self, ops_per_job: List[int]) -> Tuple[Dict, Dict]:
        """Generate mixed precedence patterns across jobs."""
        predecessors_map = {}
        successors_map = {}
        
        for job_idx, num_ops in enumerate(ops_per_job):
            # Choose pattern based on job characteristics
            if num_ops <= 3:
                pattern = 'sequential'
            elif num_ops <= 6:
                pattern = random.choice(['sequential', 'parallel'])
            else:
                pattern = random.choice(['sequential', 'parallel', 'assembly'])
            
            if pattern == 'sequential':
                preds, succs = self.generate_sequential_precedence([num_ops])
            elif pattern == 'parallel':
                preds, succs = self.generate_parallel_precedence([num_ops])
            else:  # assembly
                preds, succs = self.generate_assembly_precedence([num_ops])
            
            # Merge into main maps
            for op_idx in range(num_ops):
                op = Operation(job_idx, op_idx)
                if op in preds:
                    predecessors_map[op] = preds[op]
                else:
                    predecessors_map[op] = set()
                if op in succs:
                    successors_map[op] = succs[op]
                else:
                    successors_map[op] = set()
        
        return predecessors_map, successors_map
    
    def generate_tree_precedence(self, ops_per_job: List[int]) -> Tuple[Dict, Dict]:
        """Generate tree-like precedence structure."""
        predecessors_map = {}
        successors_map = {}
        
        for job_idx, num_ops in enumerate(ops_per_job):
            # Initialize
            for op_idx in range(num_ops):
                op = Operation(job_idx, op_idx)
                predecessors_map[op] = set()
                successors_map[op] = set()
            
            if num_ops >= 4:
                # Create binary tree-like structure
                for op_idx in range(1, num_ops):
                    curr_op = Operation(job_idx, op_idx)
                    # Each operation depends on 1-2 previous operations
                    possible_preds = [Operation(job_idx, i) for i in range(op_idx)]
                    
                    # Choose 1-2 predecessors randomly
                    num_preds = min(random.randint(1, 2), len(possible_preds))
                    selected_preds = random.sample(possible_preds, num_preds)
                    
                    for pred_op in selected_preds:
                        predecessors_map[curr_op].add(pred_op)
                        successors_map[pred_op].add(curr_op)
            else:
                # Sequential for small jobs
                preds, succs = self.generate_sequential_precedence([num_ops])
                for op_idx in range(num_ops):
                    op = Operation(job_idx, op_idx)
                    if op in preds:
                        predecessors_map[op] = preds[op]
                    if op in succs:
                        successors_map[op] = succs[op]
        
        return predecessors_map, successors_map
    
    def generate_precedence_constraints(self, 
                                        ops_per_job: List[int], 
                                        pattern: str = 'mixed') -> Tuple[Dict, Dict]:
        """
        Generate precedence constraints based on specified pattern.
        
        Args:
            ops_per_job: Number of operations per job
            pattern: Precedence pattern type
        """
        if pattern == 'sequential':
            return self.generate_sequential_precedence(ops_per_job)
        elif pattern == 'parallel':
            return self.generate_parallel_precedence(ops_per_job)
        elif pattern == 'assembly':
            return self.generate_assembly_precedence(ops_per_job)
        elif pattern == 'tree':
            return self.generate_tree_precedence(ops_per_job)
        elif pattern == 'mixed':
            return self.generate_mixed_precedence(ops_per_job)
        else:
            raise ValueError(f"Unknown precedence pattern: {pattern}")
    
    def generate_single_instance(self, 
                                 size_config: str = 'medium',
                                 precedence_pattern: str = 'mixed',
                                 time_distribution: str = 'uniform',
                                 machine_capability_prob: float = 0.8,
                                 instance_id: str = None) -> Dict:
        """
        Generate a single POFJSP instance.
        
        Args:
            size_config: Problem size configuration
            precedence_pattern: Type of precedence constraints
            time_distribution: Processing time distribution
            machine_capability_prob: Probability that a machine can process an operation
            instance_id: Unique identifier for this instance
        
        Returns:
            Dictionary containing the complete problem instance and metadata
        """
        if instance_id is None:
            instance_id = str(uuid.uuid4())[:8]
        
        # Generate problem dimensions
        num_jobs, num_machines, ops_per_job = self.generate_problem_sizes(size_config)
        
        # Generate processing times
        processing_times = self.generate_processing_times(
            num_jobs, num_machines, ops_per_job, 
            time_distribution, machine_capability_prob
        )
        
        # Generate precedence constraints
        predecessors_map, successors_map = self.generate_precedence_constraints(
            ops_per_job, precedence_pattern
        )
        
        # Create problem instance
        problem_instance = ProblemInstance(
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_operations_per_job=ops_per_job,
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
        
        # Calculate some statistics
        total_operations = sum(ops_per_job)
        avg_processing_time = 0
        feasible_assignments = 0
        
        for j in range(num_jobs):
            for o in range(ops_per_job[j]):
                times = processing_times[j][o, :]
                finite_times = times[np.isfinite(times)]
                if len(finite_times) > 0:
                    avg_processing_time += np.mean(finite_times)
                    feasible_assignments += len(finite_times)
        
        avg_processing_time = avg_processing_time / total_operations if total_operations > 0 else 0
        machine_utilization = feasible_assignments / (total_operations * num_machines)
        
        # Prepare data for serialization
        instance_data = {
            'metadata': {
                'instance_id': instance_id,
                'generation_time': datetime.datetime.now().isoformat(),
                'random_seed': self.random_seed,
                'size_config': size_config,
                'precedence_pattern': precedence_pattern,
                'time_distribution': time_distribution,
                'machine_capability_prob': machine_capability_prob,
                'total_operations': total_operations,
                'avg_processing_time': round(avg_processing_time, 2),
                'machine_utilization': round(machine_utilization, 3),
                'complexity_score': self._calculate_complexity_score(problem_instance)
            },
            'problem': {
                'num_jobs': num_jobs,
                'num_machines': num_machines,
                'num_operations_per_job': ops_per_job,
                'processing_times': [pt.tolist() for pt in processing_times],
                'predecessors_map': {
                    f"({op.job_idx},{op.op_idx_in_job})": [
                        f"({p.job_idx},{p.op_idx_in_job})" for p in preds
                    ] for op, preds in predecessors_map.items()
                },
                'successors_map': {
                    f"({op.job_idx},{op.op_idx_in_job})": [
                        f"({s.job_idx},{s.op_idx_in_job})" for s in succs
                    ] for op, succs in successors_map.items()
                }
            },
            'problem_instance': problem_instance  # For direct use
        }
        
        return instance_data
    
    def _calculate_complexity_score(self, problem_instance: ProblemInstance) -> float:
        """Calculate a complexity score for the problem instance."""
        total_ops = problem_instance.total_operations
        num_machines = problem_instance.num_machines
        
        # Count precedence edges
        precedence_edges = sum(len(preds) for preds in problem_instance.predecessors_map.values())
        
        # Calculate flexibility (average machines per operation)
        total_feasible = 0
        for j in range(problem_instance.num_jobs):
            for o in range(problem_instance.num_operations_per_job[j]):
                feasible = np.sum(np.isfinite(problem_instance.processing_times[j][o, :]))
                total_feasible += feasible
        
        avg_flexibility = total_feasible / total_ops if total_ops > 0 else 0
        
        # Complexity score combines size, precedence density, and flexibility
        size_factor = total_ops * num_machines
        precedence_density = precedence_edges / total_ops if total_ops > 0 else 0
        flexibility_factor = avg_flexibility / num_machines if num_machines > 0 else 0
        
        complexity = size_factor * (1 + precedence_density) * (2 - flexibility_factor)
        return round(complexity, 2)
    
    def generate_dataset(self, 
                         configurations: List[Dict],
                         instances_per_config: int = 10,
                         output_dir: str = './data/instances') -> List[Dict]:
        """
        Generate a dataset of POFJSP instances.
        
        Args:
            configurations: List of configuration dictionaries
            instances_per_config: Number of instances to generate per configuration
            output_dir: Directory to save the dataset
        
        Returns:
            List of generated instance data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_instances = []
        
        for config_idx, config in enumerate(configurations):
            print(f"Generating configuration {config_idx + 1}/{len(configurations)}: {config}")
            
            for instance_idx in range(instances_per_config):
                instance_id = f"{config.get('name', f'config_{config_idx}')}_{instance_idx:03d}"
                
                instance_data = self.generate_single_instance(
                    size_config=config.get('size_config', 'medium'),
                    precedence_pattern=config.get('precedence_pattern', 'mixed'),
                    time_distribution=config.get('time_distribution', 'uniform'),
                    machine_capability_prob=config.get('machine_capability_prob', 0.8),
                    instance_id=instance_id
                )
                
                all_instances.append(instance_data)
                
                # Save individual instance
                instance_file = output_path / f"{instance_id}.json"
                self._save_instance_json(instance_data, instance_file)
        
        # Save dataset summary
        self._save_dataset_summary(all_instances, output_path)
        
        # Save as parquet for efficient loading
        self._save_dataset_parquet(all_instances, output_path)
        
        print(f"Generated {len(all_instances)} instances and saved to {output_path}")
        return all_instances
    
    def _save_instance_json(self, instance_data: Dict, file_path: Path):
        """Save a single instance as JSON."""
        # Remove the problem_instance object for JSON serialization
        json_data = {
            'metadata': instance_data['metadata'],
            'problem': instance_data['problem']
        }
        
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _save_dataset_summary(self, instances: List[Dict], output_dir: Path):
        """Save dataset summary statistics."""
        summary = {
            'generation_info': {
                'total_instances': len(instances),
                'generation_time': datetime.datetime.now().isoformat(),
                'random_seed': self.random_seed
            },
            'statistics': {
                'size_configs': {},
                'precedence_patterns': {},
                'time_distributions': {},
                'complexity_scores': []
            }
        }
        
        # Collect statistics
        for instance in instances:
            metadata = instance['metadata']
            
            # Count configurations
            for key in ['size_config', 'precedence_pattern', 'time_distribution']:
                if key not in summary['statistics'][f"{key}s"]:
                    summary['statistics'][f"{key}s"][metadata[key]] = 0
                summary['statistics'][f"{key}s"][metadata[key]] += 1
            
            summary['statistics']['complexity_scores'].append(metadata['complexity_score'])
        
        # Calculate complexity statistics
        scores = summary['statistics']['complexity_scores']
        summary['statistics']['complexity_stats'] = {
            'min': min(scores),
            'max': max(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores)
        }
        
        # Save summary
        with open(output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _save_dataset_parquet(self, instances: List[Dict], output_dir: Path):
        """Save dataset as parquet files for efficient loading."""
        # Prepare metadata DataFrame
        metadata_records = []
        for instance in instances:
            metadata = instance['metadata'].copy()
            metadata_records.append(metadata)
        
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_parquet(output_dir / 'metadata.parquet', index=False)
        
        # Prepare problem data (flattened for tabular storage)
        problem_records = []
        for instance in instances:
            problem = instance['problem']
            record = {
                'instance_id': instance['metadata']['instance_id'],
                'num_jobs': problem['num_jobs'],
                'num_machines': problem['num_machines'],
                'num_operations_per_job': json.dumps(problem['num_operations_per_job']),
                'processing_times': json.dumps(problem['processing_times']),
                'predecessors_map': json.dumps(problem['predecessors_map']),
                'successors_map': json.dumps(problem['successors_map'])
            }
            problem_records.append(record)
        
        problems_df = pd.DataFrame(problem_records)
        problems_df.to_parquet(output_dir / 'problems.parquet', index=False)
        
        print(f"Saved parquet files: metadata.parquet, problems.parquet")


def create_standard_configurations() -> List[Dict]:
    """Create a set of standard configurations for dataset generation."""
    configurations = []
    
    # Size variations
    sizes = ['tiny', 'small', 'medium', 'large']
    
    # Pattern variations
    patterns = ['sequential', 'parallel', 'assembly', 'mixed', 'tree']
    
    # Time distribution variations
    distributions = ['uniform', 'normal', 'exponential']
    
    # Machine capability variations
    capabilities = [0.6, 0.8, 1.0]
    
    config_id = 0
    for size in sizes:
        for pattern in patterns:
            for dist in distributions:
                for cap in capabilities:
                    config = {
                        'name': f'config_{config_id:03d}_{size}_{pattern}_{dist}_{int(cap*100)}',
                        'size_config': size,
                        'precedence_pattern': pattern,
                        'time_distribution': dist,
                        'machine_capability_prob': cap
                    }
                    configurations.append(config)
                    config_id += 1
    
    return configurations


if __name__ == "__main__":
    # Example usage
    generator = POFJSPDataGenerator(random_seed=42)
    
    # Generate a few test instances
    test_configs = [
        {
            'name': 'test_small_mixed',
            'size_config': 'small',
            'precedence_pattern': 'mixed',
            'time_distribution': 'uniform',
            'machine_capability_prob': 0.8
        },
        {
            'name': 'test_medium_parallel',
            'size_config': 'medium',
            'precedence_pattern': 'parallel',
            'time_distribution': 'normal',
            'machine_capability_prob': 0.7
        }
    ]
    
    dataset = generator.generate_dataset(test_configs, instances_per_config=3)
    print(f"Generated {len(dataset)} test instances") 