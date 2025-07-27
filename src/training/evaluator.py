"""
Training Evaluation System

Provides comprehensive evaluation of RL agents against baselines
with timeout protection and detailed performance analysis.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import signal
import threading

import numpy as np
import torch

from problems.problem_instance import ProblemInstance, Solution
from rl.environments.pofjsp_env import POFJSPEnv
from algorithms.iaoa_gns import IAOAGNSAlgorithm
from validation import validate_inputs, Validators
from performance.monitor import performance_tracker

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating an agent on problem instances."""
    agent_name: str
    num_instances: int
    completed_instances: int
    failed_instances: int
    timeout_instances: int
    
    # Performance metrics
    makespans: List[float]
    execution_times: List[float]
    episode_lengths: List[int]
    
    # Summary statistics
    avg_makespan: float = 0.0
    best_makespan: float = float('inf')
    worst_makespan: float = 0.0
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    
    def __post_init__(self):
        """Calculate summary statistics after initialization."""
        if self.makespans:
            self.avg_makespan = np.mean(self.makespans)
            self.best_makespan = np.min(self.makespans)
            self.worst_makespan = np.max(self.makespans)
        
        if self.execution_times:
            self.avg_execution_time = np.mean(self.execution_times)
        
        self.success_rate = self.completed_instances / self.num_instances if self.num_instances > 0 else 0.0


@dataclass
class ComparisonResult:
    """Results from comparing two agents."""
    agent1_result: EvaluationResult
    agent2_result: EvaluationResult
    
    # Head-to-head comparison
    agent1_wins: int = 0
    agent2_wins: int = 0
    ties: int = 0
    win_rate: float = 0.0
    
    # Performance improvements
    makespan_improvement: float = 0.0
    time_speedup: float = 0.0
    
    def __post_init__(self):
        """Calculate comparison statistics."""
        # Calculate head-to-head comparison
        valid_comparisons = min(len(self.agent1_result.makespans), len(self.agent2_result.makespans))
        
        if valid_comparisons > 0:
            for i in range(valid_comparisons):
                makespan1 = self.agent1_result.makespans[i]
                makespan2 = self.agent2_result.makespans[i]
                
                if makespan1 < makespan2:
                    self.agent1_wins += 1
                elif makespan2 < makespan1:
                    self.agent2_wins += 1
                else:
                    self.ties += 1
            
            self.win_rate = self.agent1_wins / valid_comparisons
        
        # Calculate performance improvements
        if self.agent2_result.avg_makespan > 0:
            self.makespan_improvement = (self.agent2_result.avg_makespan - self.agent1_result.avg_makespan) / self.agent2_result.avg_makespan
        
        if self.agent1_result.avg_execution_time > 0:
            self.time_speedup = self.agent2_result.avg_execution_time / self.agent1_result.avg_execution_time


class TimeoutHandler:
    """Handles timeout for evaluation tasks."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.is_timeout = False
        self.timer = None
    
    def __enter__(self):
        def timeout_handler():
            self.is_timeout = True
            logger.warning(f"Evaluation timeout after {self.timeout_seconds}s")
        
        self.timer = threading.Timer(self.timeout_seconds, timeout_handler)
        self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()


class RLAgentEvaluator:
    """Evaluates RL agents with timeout protection and error handling."""
    
    def __init__(self, timeout_per_instance: float = 30.0, max_episode_steps: int = 500):
        self.timeout_per_instance = timeout_per_instance
        self.max_episode_steps = max_episode_steps
    
    def evaluate_agent(self, agent, instances: List[ProblemInstance], 
                      max_jobs: int, max_machines: int) -> EvaluationResult:
        """Evaluate RL agent on problem instances."""
        logger.info(f"Evaluating RL agent on {len(instances)} instances")
        
        makespans = []
        execution_times = []
        episode_lengths = []
        failed_count = 0
        timeout_count = 0
        
        for i, instance in enumerate(instances):
            try:
                with TimeoutHandler(self.timeout_per_instance) as timeout_handler:
                    start_time = time.time()
                    
                    # Create environment
                    env = POFJSPEnv(instance, time_limit=self.max_episode_steps)
                    
                    # Run episode
                    makespan, episode_length = self._run_rl_episode(
                        agent, env, max_jobs, max_machines, timeout_handler
                    )
                    
                    execution_time = time.time() - start_time
                    
                    if timeout_handler.is_timeout:
                        timeout_count += 1
                        logger.debug(f"Instance {i+1} timeout")
                        continue
                    
                    if makespan != float('inf'):
                        makespans.append(makespan)
                        execution_times.append(execution_time)
                        episode_lengths.append(episode_length)
                    else:
                        failed_count += 1
                        logger.debug(f"Instance {i+1} failed (infinite makespan)")
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"Instance {i+1} evaluation error: {e}")
        
        result = EvaluationResult(
            agent_name="RL_Agent",
            num_instances=len(instances),
            completed_instances=len(makespans),
            failed_instances=failed_count,
            timeout_instances=timeout_count,
            makespans=makespans,
            execution_times=execution_times,
            episode_lengths=episode_lengths
        )
        
        logger.info(f"RL evaluation completed: {result.completed_instances}/{len(instances)} successful, "
                   f"success rate: {result.success_rate:.3f}")
        
        return result
    
    def _run_rl_episode(self, agent, env: POFJSPEnv, max_jobs: int, max_machines: int,
                       timeout_handler: TimeoutHandler) -> Tuple[float, int]:
        """Run a single RL episode."""
        obs, _ = env.reset()
        done = False
        episode_length = 0
        
        while not done and episode_length < self.max_episode_steps and not timeout_handler.is_timeout:
            try:
                with torch.no_grad():
                    # Pad observation
                    device = next(agent.actor.parameters()).device
                    obs_padded = self._pad_observation(obs, max_jobs, max_machines, device)
                    
                    # Get action
                    action_result = agent.get_action(
                        obs_padded['x'],
                        obs_padded['edge_index'],
                        obs_padded['batch'],
                        obs_padded['job_mask'],
                        obs_padded['machine_mask'],
                        obs_padded['processing_times']
                    )
                    
                    # Extract action
                    if isinstance(action_result, tuple) and len(action_result) >= 2:
                        job_action = action_result[0]
                        machine_action = action_result[1]
                        
                        # Convert to numpy
                        if hasattr(job_action, 'cpu'):
                            job_action = job_action.cpu().numpy()
                        if hasattr(machine_action, 'cpu'):
                            machine_action = machine_action.cpu().numpy()
                        
                        # Ensure scalar
                        if isinstance(job_action, np.ndarray):
                            job_action = job_action.item()
                        if isinstance(machine_action, np.ndarray):
                            machine_action = machine_action.item()
                        
                        action = np.array([job_action, machine_action])
                    else:
                        # Fallback to random action
                        action = env.action_space.sample()
                    
                    # Take step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_length += 1
                    
            except Exception as e:
                logger.debug(f"Episode step error: {e}")
                return float('inf'), episode_length
        
        makespan = info.get('makespan', float('inf')) if info else float('inf')
        return makespan, episode_length
    
    def _pad_observation(self, obs: Dict, max_jobs: int, max_machines: int, device: torch.device) -> Dict:
        """Pad observation tensors to maximum dimensions."""
        # Pad job mask
        job_mask_padded = torch.zeros(max_jobs, dtype=torch.bool, device=device)
        actual_jobs = min(obs['job_mask'].size(0), max_jobs)
        job_mask_padded[:actual_jobs] = obs['job_mask'][:actual_jobs].to(device)
        
        # Pad machine mask
        machine_mask_padded = torch.zeros(max_machines, dtype=torch.bool, device=device)
        actual_machines = min(obs['machine_mask'].size(0), max_machines)
        machine_mask_padded[:actual_machines] = obs['machine_mask'][:actual_machines].to(device)
        
        # Pad processing times
        processing_times_padded = torch.zeros(max_jobs, max_machines, dtype=torch.float32, device=device)
        actual_jobs_pt = min(obs['processing_times'].size(0), max_jobs)
        actual_machines_pt = min(obs['processing_times'].size(1), max_machines)
        processing_times_padded[:actual_jobs_pt, :actual_machines_pt] = \
            obs['processing_times'][:actual_jobs_pt, :actual_machines_pt].to(device)
        
        return {
            'x': obs['x'].to(device),
            'edge_index': obs['edge_index'].to(device),
            'batch': obs['batch'].to(device),
            'job_mask': job_mask_padded,
            'machine_mask': machine_mask_padded,
            'processing_times': processing_times_padded
        }


class BaselineEvaluator:
    """Evaluates baseline algorithms (IAOA+GNS) with timeout protection."""
    
    def __init__(self, timeout_per_instance: float = 30.0):
        self.timeout_per_instance = timeout_per_instance
    
    def evaluate_iaoa_gns(self, instances: List[ProblemInstance], 
                         pop_size: int = 20, max_iterations: int = 10) -> EvaluationResult:
        """Evaluate IAOA+GNS algorithm on problem instances."""
        logger.info(f"Evaluating IAOA+GNS on {len(instances)} instances "
                   f"(pop_size={pop_size}, max_iter={max_iterations})")
        
        makespans = []
        execution_times = []
        failed_count = 0
        timeout_count = 0
        
        algorithm = IAOAGNSAlgorithm(pop_size=pop_size, max_iterations=max_iterations)
        
        for i, instance in enumerate(instances):
            try:
                with TimeoutHandler(self.timeout_per_instance) as timeout_handler:
                    start_time = time.time()
                    
                    # Run algorithm
                    solution = algorithm.solve(instance, verbose=False)
                    
                    execution_time = time.time() - start_time
                    
                    if timeout_handler.is_timeout:
                        timeout_count += 1
                        logger.debug(f"Instance {i+1} timeout")
                        continue
                    
                    if solution and hasattr(solution, 'makespan') and solution.makespan != float('inf'):
                        makespans.append(solution.makespan)
                        execution_times.append(execution_time)
                    else:
                        failed_count += 1
                        logger.debug(f"Instance {i+1} failed (no solution or infinite makespan)")
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"Instance {i+1} IAOA evaluation error: {e}")
        
        result = EvaluationResult(
            agent_name="IAOA_GNS",
            num_instances=len(instances),
            completed_instances=len(makespans),
            failed_instances=failed_count,
            timeout_instances=timeout_count,
            makespans=makespans,
            execution_times=execution_times,
            episode_lengths=[]  # Not applicable for IAOA
        )
        
        logger.info(f"IAOA evaluation completed: {result.completed_instances}/{len(instances)} successful, "
                   f"success rate: {result.success_rate:.3f}")
        
        return result


class ComprehensiveEvaluator:
    """Comprehensive evaluation system comparing RL agents with baselines."""
    
    def __init__(self, timeout_per_instance: float = 30.0, max_episode_steps: int = 500):
        self.rl_evaluator = RLAgentEvaluator(timeout_per_instance, max_episode_steps)
        self.baseline_evaluator = BaselineEvaluator(timeout_per_instance)
        self.timeout_per_instance = timeout_per_instance
    
    def evaluate_agent_vs_baseline(self, agent, instances: List[ProblemInstance],
                                  max_jobs: int, max_machines: int,
                                  iaoa_pop_size: int = 20, iaoa_max_iter: int = 10) -> ComparisonResult:
        """Compare RL agent against IAOA+GNS baseline."""
        logger.info(f"Comprehensive evaluation: RL vs IAOA+GNS on {len(instances)} instances")
        
        with performance_tracker("comprehensive_evaluation") as tracker:
            # Evaluate RL agent
            tracker.record_algorithm_metric("evaluation_stage", "rl_agent")
            rl_result = self.rl_evaluator.evaluate_agent(agent, instances, max_jobs, max_machines)
            
            # Evaluate IAOA+GNS baseline
            tracker.record_algorithm_metric("evaluation_stage", "iaoa_baseline")
            iaoa_result = self.baseline_evaluator.evaluate_iaoa_gns(
                instances, pop_size=iaoa_pop_size, max_iterations=iaoa_max_iter
            )
            
            # Create comparison
            comparison = ComparisonResult(rl_result, iaoa_result)
            
            # Log results
            self._log_comparison_results(comparison)
            
            tracker.record_algorithm_metric("rl_success_rate", rl_result.success_rate)
            tracker.record_algorithm_metric("iaoa_success_rate", iaoa_result.success_rate)
            tracker.record_algorithm_metric("win_rate", comparison.win_rate)
            tracker.record_algorithm_metric("makespan_improvement", comparison.makespan_improvement)
            tracker.record_algorithm_metric("time_speedup", comparison.time_speedup)
        
        return comparison
    
    def _log_comparison_results(self, comparison: ComparisonResult):
        """Log detailed comparison results."""
        rl_result = comparison.agent1_result
        iaoa_result = comparison.agent2_result
        
        logger.info("Evaluation Comparison Results:")
        logger.info(f"  RL Agent:")
        logger.info(f"    Success rate: {rl_result.success_rate:.3f}")
        logger.info(f"    Avg makespan: {rl_result.avg_makespan:.2f}")
        logger.info(f"    Avg time: {rl_result.avg_execution_time:.3f}s")
        
        logger.info(f"  IAOA+GNS:")
        logger.info(f"    Success rate: {iaoa_result.success_rate:.3f}")
        logger.info(f"    Avg makespan: {iaoa_result.avg_makespan:.2f}")
        logger.info(f"    Avg time: {iaoa_result.avg_execution_time:.3f}s")
        
        logger.info(f"  Head-to-head comparison:")
        logger.info(f"    RL wins: {comparison.agent1_wins}")
        logger.info(f"    IAOA wins: {comparison.agent2_wins}")
        logger.info(f"    Ties: {comparison.ties}")
        logger.info(f"    Win rate: {comparison.win_rate:.3f}")
        logger.info(f"    Makespan improvement: {comparison.makespan_improvement:.3f}")
        logger.info(f"    Time speedup: {comparison.time_speedup:.2f}x")
    
    def evaluate_multiple_agents(self, agents: Dict[str, Any], instances: List[ProblemInstance],
                                max_jobs: int, max_machines: int) -> Dict[str, EvaluationResult]:
        """Evaluate multiple agents on the same set of instances."""
        results = {}
        
        logger.info(f"Evaluating {len(agents)} agents on {len(instances)} instances")
        
        for agent_name, agent in agents.items():
            logger.info(f"Evaluating agent: {agent_name}")
            
            if agent_name.lower() == 'iaoa_gns':
                # Evaluate as baseline
                result = self.baseline_evaluator.evaluate_iaoa_gns(instances)
                result.agent_name = agent_name
            else:
                # Evaluate as RL agent
                result = self.rl_evaluator.evaluate_agent(agent, instances, max_jobs, max_machines)
                result.agent_name = agent_name
            
            results[agent_name] = result
            
            logger.info(f"  {agent_name}: {result.success_rate:.3f} success rate, "
                       f"{result.avg_makespan:.2f} avg makespan")
        
        return results
    
    def create_evaluation_report(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Create detailed evaluation report."""
        report = {
            'evaluation_timestamp': time.time(),
            'evaluation_config': {
                'timeout_per_instance': self.timeout_per_instance,
                'max_episode_steps': self.rl_evaluator.max_episode_steps
            },
            'rl_agent_performance': {
                'success_rate': comparison.agent1_result.success_rate,
                'completed_instances': comparison.agent1_result.completed_instances,
                'failed_instances': comparison.agent1_result.failed_instances,
                'timeout_instances': comparison.agent1_result.timeout_instances,
                'avg_makespan': comparison.agent1_result.avg_makespan,
                'best_makespan': comparison.agent1_result.best_makespan,
                'avg_execution_time': comparison.agent1_result.avg_execution_time,
                'avg_episode_length': np.mean(comparison.agent1_result.episode_lengths) if comparison.agent1_result.episode_lengths else 0
            },
            'baseline_performance': {
                'success_rate': comparison.agent2_result.success_rate,
                'completed_instances': comparison.agent2_result.completed_instances,
                'failed_instances': comparison.agent2_result.failed_instances,
                'timeout_instances': comparison.agent2_result.timeout_instances,
                'avg_makespan': comparison.agent2_result.avg_makespan,
                'best_makespan': comparison.agent2_result.best_makespan,
                'avg_execution_time': comparison.agent2_result.avg_execution_time
            },
            'comparison_results': {
                'rl_wins': comparison.agent1_wins,
                'baseline_wins': comparison.agent2_wins,
                'ties': comparison.ties,
                'win_rate': comparison.win_rate,
                'makespan_improvement': comparison.makespan_improvement,
                'time_speedup': comparison.time_speedup
            },
            'detailed_results': {
                'rl_makespans': comparison.agent1_result.makespans,
                'baseline_makespans': comparison.agent2_result.makespans,
                'rl_execution_times': comparison.agent1_result.execution_times,
                'baseline_execution_times': comparison.agent2_result.execution_times
            }
        }
        
        return report


# Example usage and testing
if __name__ == "__main__":
    from src.training.curriculum import ProblemInstanceGenerator
    from src.training.config import CurriculumStageConfig
    
    # Create test instances
    generator = ProblemInstanceGenerator(seed=42)
    stage_config = CurriculumStageConfig(
        name='test_stage',
        job_range=(8, 12),
        machine_range=(6, 8),
        instances_per_eval=5,
        target_success_rate=0.6,
        max_episode_steps=200
    )
    
    test_instances = generator.generate_curriculum_instances(stage_config, 5)
    
    # Test baseline evaluator
    baseline_evaluator = BaselineEvaluator(timeout_per_instance=10.0)
    baseline_result = baseline_evaluator.evaluate_iaoa_gns(test_instances, pop_size=10, max_iterations=5)
    
    print("Baseline Evaluation Results:")
    print(f"  Success rate: {baseline_result.success_rate:.3f}")
    print(f"  Avg makespan: {baseline_result.avg_makespan:.2f}")
    print(f"  Avg time: {baseline_result.avg_execution_time:.3f}s")
    print(f"  Completed: {baseline_result.completed_instances}/{baseline_result.num_instances}")
    
    # Test evaluation report
    comprehensive_evaluator = ComprehensiveEvaluator(timeout_per_instance=10.0)
    
    # Create a dummy comparison for testing
    rl_result = EvaluationResult(
        agent_name="RL_Test",
        num_instances=5,
        completed_instances=4,
        failed_instances=1,
        timeout_instances=0,
        makespans=[95, 88, 102, 91],
        execution_times=[0.5, 0.6, 0.4, 0.7],
        episode_lengths=[45, 52, 38, 60]
    )
    
    comparison = ComparisonResult(rl_result, baseline_result)
    report = comprehensive_evaluator.create_evaluation_report(comparison)
    
    print(f"\nComparison Results:")
    print(f"  Win rate: {comparison.win_rate:.3f}")
    print(f"  Makespan improvement: {comparison.makespan_improvement:.3f}")
    print(f"  Time speedup: {comparison.time_speedup:.2f}x")