"""
Curriculum Learning Manager

Handles progressive difficulty scaling for RL training with
automated stage advancement based on performance metrics.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from problems.problem_instance import ProblemInstance, Operation
from training.config import TrainingConfig, CurriculumStageConfig
from validation import validate_inputs, Validators

logger = logging.getLogger(__name__)


@dataclass
class PerformanceRecord:
    """Record of performance at a specific curriculum stage."""
    stage_name: str
    success_rate: float
    avg_makespan: float
    avg_episode_steps: float
    timestamp: float
    additional_metrics: Dict = field(default_factory=dict)


class CurriculumManager:
    """Manages curriculum learning progression and stage transitions."""
    
    def __init__(self, config: TrainingConfig, stages: Optional[List[CurriculumStageConfig]] = None):
        self.config = config
        self.stages = stages or self._create_default_stages()
        self.current_stage_idx = 0
        self.performance_history = []
        self.stage_performance = {stage.name: deque(maxlen=10) for stage in self.stages}
        self.advancement_history = []
        
        logger.info(f"Curriculum initialized with {len(self.stages)} stages")
        for i, stage in enumerate(self.stages):
            logger.info(f"  Stage {i+1}: {stage.name} - {stage.job_range[0]}-{stage.job_range[1]} jobs, "
                       f"{stage.machine_range[0]}-{stage.machine_range[1]} machines")
    
    def _create_default_stages(self) -> List[CurriculumStageConfig]:
        """Create default curriculum stages if none provided."""
        stages = []
        
        # Stage 1: Small problems
        stages.append(CurriculumStageConfig(
            name='small_problems',
            job_range=(self.config.min_jobs, min(15, self.config.max_jobs)),
            machine_range=(self.config.min_machines, min(10, self.config.max_machines)),
            instances_per_eval=50,
            target_success_rate=0.7,
            max_episode_steps=200,
            complexity="simple"
        ))
        
        # Stage 2: Medium problems
        if self.config.curriculum_stages > 1:
            stages.append(CurriculumStageConfig(
                name='medium_problems',
                job_range=(15, min(30, self.config.max_jobs)),
                machine_range=(10, min(20, self.config.max_machines)),
                instances_per_eval=30,
                target_success_rate=0.6,
                max_episode_steps=300,
                complexity="medium"
            ))
        
        # Stage 3: Large problems
        if self.config.curriculum_stages > 2:
            stages.append(CurriculumStageConfig(
                name='large_problems',
                job_range=(30, self.config.max_jobs),
                machine_range=(20, self.config.max_machines),
                instances_per_eval=20,
                target_success_rate=0.5,
                max_episode_steps=400,
                complexity="complex"
            ))
        
        return stages
    
    def get_current_stage(self) -> CurriculumStageConfig:
        """Get the current curriculum stage."""
        return self.stages[min(self.current_stage_idx, len(self.stages) - 1)]
    
    def record_performance(self, success_rate: float, avg_makespan: float, 
                          avg_episode_steps: float, **additional_metrics) -> None:
        """Record performance metrics for the current stage."""
        current_stage = self.get_current_stage()
        
        record = PerformanceRecord(
            stage_name=current_stage.name,
            success_rate=success_rate,
            avg_makespan=avg_makespan,
            avg_episode_steps=avg_episode_steps,
            timestamp=time.time(),
            additional_metrics=additional_metrics
        )
        
        self.performance_history.append(record)
        self.stage_performance[current_stage.name].append(record)
        
        logger.info(f"Recorded performance for {current_stage.name}: "
                   f"success_rate={success_rate:.3f}, avg_makespan={avg_makespan:.2f}")
    
    def should_advance_stage(self) -> Tuple[bool, str]:
        """
        Determine if should advance to next curriculum stage.
        
        Returns:
            Tuple of (should_advance, reason)
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False, "Already at final stage"
        
        current_stage = self.get_current_stage()
        recent_performance = list(self.stage_performance[current_stage.name])
        
        if len(recent_performance) < 3:
            return False, f"Need at least 3 evaluations (have {len(recent_performance)})"
        
        # Check recent performance trend
        recent_success_rates = [p.success_rate for p in recent_performance[-3:]]
        avg_recent_success = np.mean(recent_success_rates)
        
        # Check if consistently meeting target
        if avg_recent_success >= current_stage.target_success_rate:
            # Additional check: ensure performance is stable (not declining)
            if len(recent_success_rates) >= 3:
                trend = np.polyfit(range(len(recent_success_rates)), recent_success_rates, 1)[0]
                if trend < -0.1:  # Declining trend
                    return False, f"Performance declining (trend: {trend:.3f})"
            
            return True, f"Target achieved: {avg_recent_success:.3f} >= {current_stage.target_success_rate:.3f}"
        
        # Check if performance has plateaued (no improvement in recent evaluations)
        if len(recent_performance) >= 6:
            older_performance = recent_performance[-6:-3]
            recent_performance_subset = recent_performance[-3:]
            
            older_avg = np.mean([p.success_rate for p in older_performance])
            recent_avg = np.mean([p.success_rate for p in recent_performance_subset])
            
            # If no significant improvement and reasonable performance, consider advancing
            improvement = recent_avg - older_avg
            if abs(improvement) < 0.05 and recent_avg >= current_stage.target_success_rate * 0.8:
                return True, f"Performance plateaued at acceptable level: {recent_avg:.3f}"
        
        return False, f"Current performance {avg_recent_success:.3f} < target {current_stage.target_success_rate:.3f}"
    
    def advance_stage(self) -> bool:
        """
        Advance to the next curriculum stage.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            logger.warning("Cannot advance: already at final curriculum stage")
            return False
        
        old_stage = self.get_current_stage()
        self.current_stage_idx += 1
        new_stage = self.get_current_stage()
        
        # Record advancement
        advancement_record = {
            'timestamp': time.time(),
            'from_stage': old_stage.name,
            'to_stage': new_stage.name,
            'stage_index': self.current_stage_idx,
            'performance_history': list(self.stage_performance[old_stage.name])
        }
        self.advancement_history.append(advancement_record)
        
        logger.info(f"Advanced curriculum: {old_stage.name} -> {new_stage.name}")
        logger.info(f"New stage parameters: {new_stage.job_range[0]}-{new_stage.job_range[1]} jobs, "
                   f"{new_stage.machine_range[0]}-{new_stage.machine_range[1]} machines")
        
        return True
    
    def get_stage_statistics(self) -> Dict[str, Dict]:
        """Get comprehensive statistics for all stages."""
        stats = {}
        
        for stage_name, performance_records in self.stage_performance.items():
            if not performance_records:
                stats[stage_name] = {
                    'evaluations': 0,
                    'avg_success_rate': 0.0,
                    'best_success_rate': 0.0,
                    'avg_makespan': 0.0,
                    'best_makespan': float('inf')
                }
                continue
            
            success_rates = [p.success_rate for p in performance_records]
            makespans = [p.avg_makespan for p in performance_records if p.avg_makespan > 0]
            
            stats[stage_name] = {
                'evaluations': len(performance_records),
                'avg_success_rate': np.mean(success_rates),
                'best_success_rate': np.max(success_rates),
                'latest_success_rate': success_rates[-1] if success_rates else 0.0,
                'avg_makespan': np.mean(makespans) if makespans else 0.0,
                'best_makespan': np.min(makespans) if makespans else float('inf'),
                'performance_trend': self._calculate_trend(success_rates),
                'time_in_stage': self._calculate_time_in_stage(stage_name)
            }
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate performance trend (positive = improving)."""
        if len(values) < 2:
            return 0.0
        
        try:
            # Linear regression slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except:
            return 0.0
    
    def _calculate_time_in_stage(self, stage_name: str) -> float:
        """Calculate total time spent in a stage."""
        stage_records = [p for p in self.performance_history if p.stage_name == stage_name]
        if len(stage_records) < 2:
            return 0.0
        
        return stage_records[-1].timestamp - stage_records[0].timestamp
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has been completed (advanced beyond)."""
        current_stage = self.get_current_stage()
        
        # Find stage index
        stage_idx = next((i for i, s in enumerate(self.stages) if s.name == stage_name), -1)
        
        return stage_idx != -1 and self.current_stage_idx > stage_idx
    
    def reset_to_stage(self, stage_idx: int) -> bool:
        """Reset curriculum to a specific stage (for debugging/retraining)."""
        if not 0 <= stage_idx < len(self.stages):
            logger.error(f"Invalid stage index: {stage_idx}")
            return False
        
        old_stage = self.get_current_stage()
        self.current_stage_idx = stage_idx
        new_stage = self.get_current_stage()
        
        logger.info(f"Reset curriculum: {old_stage.name} -> {new_stage.name}")
        return True


class ProblemInstanceGenerator:
    """Generates problem instances for curriculum training."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        logger.info(f"Problem generator initialized with seed: {seed}")
    
    def generate_curriculum_instances(self, stage: CurriculumStageConfig, 
                                    num_instances: int) -> List[ProblemInstance]:
        """Generate problem instances for a curriculum stage."""
        instances = []
        
        job_min, job_max = stage.job_range
        machine_min, machine_max = stage.machine_range
        
        logger.info(f"Generating {num_instances} instances for {stage.name}: "
                   f"{job_min}-{job_max} jobs, {machine_min}-{machine_max} machines")
        
        for i in range(num_instances):
            # Vary problem size within stage range
            n_jobs = self.rng.randint(job_min, job_max + 1)
            n_machines = self.rng.randint(machine_min, machine_max + 1)
            
            instance = self._create_instance(n_jobs, n_machines, stage.complexity)
            instances.append(instance)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i + 1}/{num_instances} instances")
        
        return instances
    
    def _create_instance(self, n_jobs: int, n_machines: int, complexity: str) -> ProblemInstance:
        """Create a single problem instance with specified complexity."""
        # Generate operations per job based on complexity
        if complexity == "simple":
            ops_range = (2, 4)
            infeasible_ratio = 0.1
            inter_job_constraints = 0
        elif complexity == "medium":
            ops_range = (3, 6)
            infeasible_ratio = 0.3
            inter_job_constraints = min(2, n_jobs // 3)
        else:  # complex
            ops_range = (4, 8)
            infeasible_ratio = 0.5
            inter_job_constraints = min(5, n_jobs // 2)
        
        num_operations_per_job = [
            self.rng.randint(ops_range[0], ops_range[1] + 1) 
            for _ in range(n_jobs)
        ]
        
        # Generate processing times
        processing_times = []
        for job_idx in range(n_jobs):
            n_ops = num_operations_per_job[job_idx]
            
            # Create processing time matrix
            job_times = self.rng.randint(10, 100, size=(n_ops, n_machines)).astype(float)
            
            # Add infeasible assignments
            for op_idx in range(n_ops):
                if infeasible_ratio > 0:
                    n_infeasible = int(n_machines * infeasible_ratio)
                    if n_infeasible > 0 and n_infeasible < n_machines:  # Ensure at least one feasible machine
                        infeasible_machines = self.rng.choice(
                            n_machines, size=n_infeasible, replace=False
                        )
                        job_times[op_idx, infeasible_machines] = np.inf
            
            processing_times.append(job_times)
        
        # Generate precedence constraints
        predecessors_map, successors_map = self._generate_precedence_constraints(
            n_jobs, num_operations_per_job, inter_job_constraints
        )
        
        return ProblemInstance(
            num_jobs=n_jobs,
            num_machines=n_machines,
            num_operations_per_job=num_operations_per_job,
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
    
    def _generate_precedence_constraints(self, n_jobs: int, num_operations_per_job: List[int], 
                                       inter_job_constraints: int) -> Tuple[Dict, Dict]:
        """Generate precedence constraint maps."""
        predecessors_map = {}
        successors_map = {}
        
        # Initialize all operations
        all_operations = []
        for j in range(n_jobs):
            for o in range(num_operations_per_job[j]):
                op = Operation(j, o)
                all_operations.append(op)
                predecessors_map[op] = set()
                successors_map[op] = set()
        
        # Add intra-job precedence constraints (sequential within each job)
        for job_idx in range(n_jobs):
            for op_idx in range(1, num_operations_per_job[job_idx]):
                current_op = Operation(job_idx, op_idx)
                prev_op = Operation(job_idx, op_idx - 1)
                
                predecessors_map[current_op].add(prev_op)
                successors_map[prev_op].add(current_op)
        
        # Add inter-job precedence constraints
        if inter_job_constraints > 0 and n_jobs > 1:
            for _ in range(inter_job_constraints):
                # Select two different jobs
                job1, job2 = self.rng.choice(n_jobs, 2, replace=False)
                
                # Select random operations from each job
                op1 = Operation(job1, self.rng.randint(num_operations_per_job[job1]))
                op2 = Operation(job2, self.rng.randint(num_operations_per_job[job2]))
                
                # Avoid creating cycles by ensuring op1 is not already a successor of op2
                if op1 not in self._get_all_successors(op2, successors_map):
                    predecessors_map[op2].add(op1)
                    successors_map[op1].add(op2)
        
        return predecessors_map, successors_map
    
    def _get_all_successors(self, op: Operation, successors_map: Dict) -> set:
        """Get all successors of an operation (transitive closure)."""
        all_successors = set()
        to_visit = list(successors_map.get(op, set()))
        
        while to_visit:
            current = to_visit.pop()
            if current not in all_successors:
                all_successors.add(current)
                to_visit.extend(successors_map.get(current, set()))
        
        return all_successors
    
    def create_evaluation_suite(self, stages: List[CurriculumStageConfig]) -> Dict[str, List[ProblemInstance]]:
        """Create a comprehensive evaluation suite for all stages."""
        evaluation_suite = {}
        
        for stage in stages:
            # Generate fewer instances for evaluation to save time
            num_eval_instances = min(stage.instances_per_eval, 20)
            instances = self.generate_curriculum_instances(stage, num_eval_instances)
            evaluation_suite[stage.name] = instances
        
        logger.info(f"Created evaluation suite with {sum(len(instances) for instances in evaluation_suite.values())} total instances")
        return evaluation_suite


# Example usage and testing
if __name__ == "__main__":
    from src.training.config import TrainingConfig, config_manager
    
    # Test curriculum manager
    config = config_manager.get_config('fast')
    stages = config_manager.create_curriculum_stages(config)
    
    curriculum = CurriculumManager(config, stages)
    generator = ProblemInstanceGenerator(seed=42)
    
    print("Testing curriculum progression:")
    
    # Simulate training progression
    for evaluation in range(10):
        current_stage = curriculum.get_current_stage()
        print(f"\nEvaluation {evaluation + 1} - Stage: {current_stage.name}")
        
        # Generate test performance (simulated)
        success_rate = min(1.0, 0.3 + evaluation * 0.1 + np.random.random() * 0.1)
        avg_makespan = 100 - evaluation * 5 + np.random.random() * 10
        avg_steps = 50 + np.random.randint(-10, 10)
        
        curriculum.record_performance(success_rate, avg_makespan, avg_steps)
        
        should_advance, reason = curriculum.should_advance_stage()
        print(f"  Performance: {success_rate:.3f} success rate, {avg_makespan:.1f} avg makespan")
        print(f"  Should advance: {should_advance} ({reason})")
        
        if should_advance:
            advanced = curriculum.advance_stage()
            if advanced:
                print(f"  Advanced to: {curriculum.get_current_stage().name}")
            else:
                print("  Cannot advance further")
                break
    
    # Show final statistics
    print("\nFinal curriculum statistics:")
    stats = curriculum.get_stage_statistics()
    for stage_name, stage_stats in stats.items():
        print(f"\n{stage_name}:")
        print(f"  Evaluations: {stage_stats['evaluations']}")
        print(f"  Best success rate: {stage_stats['best_success_rate']:.3f}")
        print(f"  Performance trend: {stage_stats['performance_trend']:+.3f}")
        print(f"  Time in stage: {stage_stats['time_in_stage']:.1f}s")
    
    # Test problem generation
    print(f"\nTesting problem generation:")
    for stage in stages[:2]:  # Test first two stages
        instances = generator.generate_curriculum_instances(stage, 3)
        print(f"{stage.name}: Generated {len(instances)} instances")
        for i, instance in enumerate(instances):
            print(f"  Instance {i+1}: {instance.num_jobs} jobs, {instance.num_machines} machines, {instance.total_operations} operations")