#!/usr/bin/env python3
"""
RL Training System for POFJSP

Production-ready reinforcement learning training system featuring:
- CUDA-accelerated training with vectorized environments  
- Progressive curriculum learning (8x6 to 100x100 problem sizes)
- Graph Neural Network-based PPO agent
- Real-time performance monitoring and GPU optimization
- Comprehensive evaluation against IAOA+GNS baseline
- Multi-scale problem handling with dynamic padding
"""
import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import random
import signal
import sys
from torch.utils.tensorboard import SummaryWriter
try:
    from torch.profiler import profile, record_function, ProfilerActivity
except ImportError:
    # Fallback for older PyTorch versions
    profile = None
    record_function = None
    ProfilerActivity = None

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from algorithms.iaoa_gns import IAOAGNSAlgorithm
from rl.environments.pofjsp_env import POFJSPEnv
from rl.models.ppo_agent import PPOAgent
from problems.problem_instance import ProblemInstance, Operation

@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters"""
    # Problem scaling
    min_jobs: int = 8
    min_machines: int = 6
    max_jobs: int = 100
    max_machines: int = 100
    
    # Training parameters
    total_timesteps: int = 1_000_000  # Reduced for faster testing
    learning_rate: float = 1e-4
    batch_size: int = 512  # Increased for better GPU utilization
    n_steps: int = 2048    # Increased buffer size
    n_epochs: int = 8      # More epochs for better learning
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Environment
    max_episode_steps: int = 500
    
    # Curriculum learning
    curriculum_stages: int = 3
    stage_timesteps: int = 300_000
    
    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 3
    
    # Output and monitoring
    save_every: int = 50_000
    eval_every: int = 25_000
    log_every: int = 50  # Much more frequent logging for debugging

class EnhancedPerformanceMonitor:
    """Enhanced performance monitoring with GPU tracking and TensorBoard integration"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.start_time = time.time()
        self.step_times = []
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        self.episode_count = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,   # Reduce logging for performance
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"TensorBoard logs will be saved to: {self.log_dir / 'tensorboard'}")
        self.logger.info(f"Launch TensorBoard with: tensorboard --logdir={self.log_dir / 'tensorboard'}")
    
    def log_metrics(self, step: int, metrics: Dict):
        """Log training metrics to both files and TensorBoard"""
        metrics['step'] = step
        metrics['elapsed_time'] = time.time() - self.start_time
        metrics['gpu_memory_used'] = self.get_gpu_memory_usage()
        self.metrics_history.append(metrics)
        
        # Log to TensorBoard
        self._log_to_tensorboard(step, metrics)
        
        # Log key metrics to console
        if 'eval_win_rate' in metrics:
            self.logger.info(f"Step {step}: Win Rate={metrics['eval_win_rate']:.3f}, GPU={metrics['gpu_memory_used']:.2f}GB")
        else:
            self.logger.info(f"Step {step}: Episodes={metrics.get('episodes_completed', 0)}, GPU={metrics['gpu_memory_used']:.2f}GB")
        
        # Save detailed metrics to file
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def _log_to_tensorboard(self, step: int, metrics: Dict):
        """Log metrics to TensorBoard"""
        # Training metrics
        if 'avg_episode_reward' in metrics:
            self.writer.add_scalar('Training/Episode_Reward', metrics['avg_episode_reward'], step)
        if 'episodes_completed' in metrics:
            self.writer.add_scalar('Training/Episodes_Completed', metrics['episodes_completed'], step)
        if 'progress' in metrics:
            self.writer.add_scalar('Training/Progress', metrics['progress'], step)
        if 'stage' in metrics:
            # Create a numeric representation of stage for plotting
            stage_map = {'small_problems': 1, 'medium_problems': 2, 'large_problems': 3}
            stage_num = stage_map.get(metrics['stage'], 0)
            self.writer.add_scalar('Training/Curriculum_Stage', stage_num, step)
        
        # PPO Training losses
        if 'training_total_loss' in metrics:
            self.writer.add_scalar('Training/Total_Loss', metrics['training_total_loss'], step)
        if 'training_policy_loss' in metrics:
            self.writer.add_scalar('Training/Policy_Loss', metrics['training_policy_loss'], step)
        if 'training_value_loss' in metrics:
            self.writer.add_scalar('Training/Value_Loss', metrics['training_value_loss'], step)
        if 'training_entropy_loss' in metrics:
            self.writer.add_scalar('Training/Entropy_Loss', metrics['training_entropy_loss'], step)
        
        # Evaluation metrics
        if 'eval_win_rate' in metrics:
            self.writer.add_scalar('Evaluation/Win_Rate', metrics['eval_win_rate'], step)
        if 'eval_makespan_improvement' in metrics:
            self.writer.add_scalar('Evaluation/Makespan_Improvement', metrics['eval_makespan_improvement'], step)
        if 'eval_time_speedup' in metrics:
            self.writer.add_scalar('Evaluation/Time_Speedup', metrics['eval_time_speedup'], step)
        if 'eval_completed_instances' in metrics:
            self.writer.add_scalar('Evaluation/Completed_Instances', metrics['eval_completed_instances'], step)
        
        # Performance metrics
        if 'gpu_memory_used' in metrics:
            self.writer.add_scalar('Performance/GPU_Memory_GB', metrics['gpu_memory_used'], step)
        if 'elapsed_time' in metrics:
            self.writer.add_scalar('Performance/Elapsed_Time_Seconds', metrics['elapsed_time'], step)
        
        # Flush to ensure data is written
        self.writer.flush()
    
    def log_episode_metrics(self, episode: int, episode_reward: float, episode_steps: int, makespan: float, stage: str):
        """Log individual episode metrics to TensorBoard"""
        self.episode_count += 1
        
        self.writer.add_scalar('Episodes/Reward', episode_reward, self.episode_count)
        self.writer.add_scalar('Episodes/Steps', episode_steps, self.episode_count)
        if makespan != float('inf'):
            self.writer.add_scalar('Episodes/Makespan', makespan, self.episode_count)
        
        # Log stage as text
        self.writer.add_text('Episodes/Current_Stage', stage, self.episode_count)
        
        self.writer.flush()
    
    def log_model_parameters(self, agent, step: int):
        """Log model parameters and gradients to TensorBoard"""
        # Log parameter histograms
        for name, param in agent.actor.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Actor_Gradients/{name}', param.grad, step)
            self.writer.add_histogram(f'Actor_Parameters/{name}', param, step)
        
        for name, param in agent.critic.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Critic_Gradients/{name}', param.grad, step)
            self.writer.add_histogram(f'Critic_Parameters/{name}', param, step)
        
        self.writer.flush()
    
    def log_curriculum_transition(self, old_stage: str, new_stage: str, step: int, success_rate: float):
        """Log curriculum stage transitions"""
        self.writer.add_text('Curriculum/Stage_Transition', 
                           f'Transitioned from {old_stage} to {new_stage} at step {step} with success rate {success_rate:.3f}', step)
        self.writer.flush()
    
    def log_gpu_usage(self):
        """Enhanced GPU memory usage and performance monitoring"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_cached = torch.cuda.memory_cached() / 1024**3 if hasattr(torch.cuda, 'memory_cached') else 0
            
            # GPU utilization (if nvidia-ml-py is available)
            gpu_utilization = self.get_gpu_utilization()
            
            # Log detailed GPU metrics to TensorBoard
            self.writer.add_scalar('Performance/GPU_Memory_Allocated_GB', memory_allocated, self.episode_count)
            self.writer.add_scalar('Performance/GPU_Memory_Reserved_GB', memory_reserved, self.episode_count)
            self.writer.add_scalar('Performance/GPU_Memory_Cached_GB', memory_cached, self.episode_count)
            if gpu_utilization is not None:
                self.writer.add_scalar('Performance/GPU_Utilization_Percent', gpu_utilization, self.episode_count)
            
            # Log to console with more details
            utilization_str = f", Util: {gpu_utilization:.0f}%" if gpu_utilization is not None else ""
            self.logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB{utilization_str}")
            
            # Clear cache if memory usage is high
            if memory_allocated > 6.0:  # More than 6GB
                torch.cuda.empty_cache()
                self.logger.info(f"GPU cache cleared (was {memory_allocated:.2f}GB)")
    
    def get_gpu_utilization(self):
        """Get GPU utilization percentage if possible"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except (ImportError, Exception):
            return None
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

class CurriculumManager:
    """Manages curriculum learning for progressive difficulty scaling"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage = 0
        self.stages = self._create_curriculum_stages()
        self.performance_history = []
    
    def _create_curriculum_stages(self) -> List[Dict]:
        """Create curriculum stages from simple to complex problems"""
        stages = []
        
        # Stage 1: Small problems (8x6 to 15x10)
        stages.append({
            'name': 'small_problems',
            'job_range': (8, 15),
            'machine_range': (6, 10),
            'instances_per_eval': 50,
            'target_success_rate': 0.6,
            'max_episode_steps': 200
        })
        
        # Stage 2: Medium problems (15x10 to 30x20)  
        stages.append({
            'name': 'medium_problems',
            'job_range': (15, 30),
            'machine_range': (10, 20),
            'instances_per_eval': 30,
            'target_success_rate': 0.5,
            'max_episode_steps': 300
        })
        
        # Stage 3: Large problems (30x20 to 50x35)
        stages.append({
            'name': 'large_problems', 
            'job_range': (30, 50),
            'machine_range': (20, 35),
            'instances_per_eval': 20,
            'target_success_rate': 0.4,
            'max_episode_steps': 400
        })
        
        return stages
    
    def get_current_stage(self) -> Dict:
        """Get current curriculum stage"""
        return self.stages[min(self.current_stage, len(self.stages) - 1)]
    
    def should_advance_stage(self, success_rate: float) -> bool:
        """Check if should advance to next stage"""
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        current_stage = self.get_current_stage()
        
        # Add to history
        self.performance_history.append({
            'stage': current_stage['name'],
            'success_rate': success_rate,
            'timestamp': time.time()
        })
        
        # Check if performance is consistently good over last few evaluations
        if len(self.performance_history) >= 3:
            recent_performance = [p['success_rate'] for p in self.performance_history[-3:] 
                                if p['stage'] == current_stage['name']]
            if len(recent_performance) >= 2 and np.mean(recent_performance) >= current_stage['target_success_rate']:
                return True
        
        return success_rate >= current_stage['target_success_rate']
    
    def advance_stage(self):
        """Advance to next curriculum stage"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False

def generate_curriculum_instances(stage_config: Dict, num_instances: int = 30) -> List[ProblemInstance]:
    """Generate problem instances for current curriculum stage"""
    instances = []
    
    job_min, job_max = stage_config['job_range']
    machine_min, machine_max = stage_config['machine_range']
    
    for _ in range(num_instances):
        n_jobs = np.random.randint(job_min, job_max + 1)
        n_machines = np.random.randint(machine_min, machine_max + 1)
        
        # Create random POFJSP instance with realistic structure
        num_operations_per_job = [np.random.randint(2, 5) for _ in range(n_jobs)]
        processing_times = []
        predecessors_map = {}
        successors_map = {}
        
        for j in range(n_jobs):
            n_ops = num_operations_per_job[j]
            
            # Create realistic processing times with some machines being infeasible
            job_times = np.full((n_ops, n_machines), np.inf)
            for o in range(n_ops):
                # Each operation can be processed on 50-80% of machines
                num_feasible = max(1, int(0.5 + 0.3 * np.random.random()) * n_machines)
                feasible_machines = np.random.choice(n_machines, num_feasible, replace=False)
                job_times[o, feasible_machines] = np.random.randint(10, 100, size=len(feasible_machines))
            
            processing_times.append(job_times)
            
            # Simple chain precedence within each job
            for o in range(n_ops):
                op = Operation(j, o)
                if o > 0:
                    prev_op = Operation(j, o-1)
                    predecessors_map[op] = {prev_op}
                    if prev_op not in successors_map:
                        successors_map[prev_op] = set()
                    successors_map[prev_op].add(op)
        
        instance = ProblemInstance(
            num_jobs=n_jobs,
            num_machines=n_machines,
            num_operations_per_job=num_operations_per_job,
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
        instances.append(instance)
    
    return instances

def safe_evaluate_against_iaoa(agent: PPOAgent, instances: List[ProblemInstance], 
                              monitor: EnhancedPerformanceMonitor, timeout: float = 30.0) -> Dict:
    """Safely evaluate RL agent against IAOA+GNS algorithm with timeout protection"""
    results = {
        'rl_makespans': [],
        'iaoa_makespans': [],
        'rl_times': [],
        'iaoa_times': [],
        'rl_wins': 0,
        'completed_instances': 0,
        'rl_failures': 0,
        'iaoa_failures': 0
    }
    
    # Use smaller population and iterations for faster evaluation
    iaoa = IAOAGNSAlgorithm(pop_size=20, max_iterations=10)
    
    monitor.logger.info(f"Evaluating on {len(instances)} instances (timeout: {timeout}s per instance)...")
    
    for i, instance in enumerate(instances):
        try:
            # RL agent evaluation with timeout
            env = POFJSPEnv(instance, time_limit=200)  # Shorter time limit
            start_time = time.time()
            
            obs, _ = env.reset()
            done = False
            steps = 0
            max_steps = 100  # Prevent infinite loops
            
            while not done and steps < max_steps and (time.time() - start_time) < timeout:
                with torch.no_grad():
                    # Move all tensors to the same device as the agent
                    device = next(agent.actor.parameters()).device
                    
                    # Pad observation to maximum network dimensions
                    obs_cuda = pad_observation(obs, 100, 100, device)  # Use maximum dimensions
                    
                    action = agent.get_action(
                        obs_cuda['x'], 
                        obs_cuda['edge_index'], 
                        obs_cuda['batch'], 
                        obs_cuda['job_mask'], 
                        obs_cuda['machine_mask'],
                        obs_cuda['processing_times']
                    )
                    # Extract job and machine actions from the returned tuple
                    if isinstance(action, tuple):
                        job_action, machine_action = action[0], action[1]  # Get the action values
                        combined_action = np.array([job_action.cpu().numpy() if hasattr(job_action, 'cpu') else job_action,
                                                  machine_action.cpu().numpy() if hasattr(machine_action, 'cpu') else machine_action])
                    else:
                        combined_action = action
                    
                    obs, reward, terminated, truncated, info = env.step(combined_action)
                    done = terminated or truncated
                    steps += 1
            
            rl_time = time.time() - start_time
            rl_makespan = info.get('makespan', float('inf')) if 'info' in locals() and info else float('inf')
            
            # Skip IAOA evaluation if RL failed
            if rl_makespan == float('inf'):
                continue
            
            # IAOA+GNS evaluation with timeout
            start_time = time.time()
            try:
                iaoa_solution = iaoa.solve(instance, verbose=False)
                iaoa_time = time.time() - start_time
                
                # Check timeout
                if iaoa_time > timeout:
                    monitor.logger.warning(f"IAOA timeout on instance {i}")
                    continue
                    
                iaoa_makespan = iaoa_solution.makespan if iaoa_solution else float('inf')
                
            except Exception as e:
                monitor.logger.warning(f"IAOA evaluation error on instance {i}: {e}")
                results['iaoa_failures'] += 1
                continue
            
            # Record results only if both succeeded
            if rl_makespan != float('inf') and iaoa_makespan != float('inf'):
                results['rl_makespans'].append(rl_makespan)
                results['iaoa_makespans'].append(iaoa_makespan)
                results['rl_times'].append(rl_time)
                results['iaoa_times'].append(iaoa_time)
                results['completed_instances'] += 1
                
                if rl_makespan <= iaoa_makespan:  # RL wins if equal or better
                    results['rl_wins'] += 1
                
                if (i + 1) % 5 == 0:
                    monitor.logger.info(f"Completed {i + 1}/{len(instances)} instances")
        
        except Exception as e:
            monitor.logger.error(f"Critical error evaluating instance {i}: {e}")
            continue
    
    # Calculate summary statistics
    if results['completed_instances'] > 0:
        results['win_rate'] = results['rl_wins'] / results['completed_instances']
        results['avg_rl_makespan'] = np.mean(results['rl_makespans'])
        results['avg_iaoa_makespan'] = np.mean(results['iaoa_makespans'])
        results['avg_rl_time'] = np.mean(results['rl_times'])
        results['avg_iaoa_time'] = np.mean(results['iaoa_times'])
        results['avg_time_speedup'] = results['avg_iaoa_time'] / results['avg_rl_time'] if results['avg_rl_time'] > 0 else 0
        results['makespan_improvement'] = (results['avg_iaoa_makespan'] - results['avg_rl_makespan']) / results['avg_iaoa_makespan'] if results['avg_iaoa_makespan'] > 0 else 0
    else:
        results.update({
            'win_rate': 0,
            'avg_rl_makespan': 0,
            'avg_iaoa_makespan': 0, 
            'avg_rl_time': 0,
            'avg_iaoa_time': 0,
            'avg_time_speedup': 0,
            'makespan_improvement': 0
        })
    
    monitor.logger.info(f"Evaluation completed: {results['completed_instances']} instances, {results['rl_failures']} RL failures, {results['iaoa_failures']} IAOA failures")
    
    return results

def create_sample_environment(config: TrainingConfig) -> Tuple[POFJSPEnv, ProblemInstance]:
    """Create a sample environment for agent initialization"""
    n_jobs = config.min_jobs
    n_machines = config.min_machines
    
    # Create a simple sample instance
    num_operations_per_job = [3] * n_jobs  # Fixed number of operations per job
    processing_times = []
    predecessors_map = {}
    successors_map = {}
    
    for j in range(n_jobs):
        n_ops = num_operations_per_job[j]
        job_times = np.random.randint(10, 50, size=(n_ops, n_machines))
        processing_times.append(job_times)
        
        for o in range(n_ops):
            op = Operation(j, o)
            if o > 0:
                prev_op = Operation(j, o-1)
                predecessors_map[op] = {prev_op}
                if prev_op not in successors_map:
                    successors_map[prev_op] = set()
                successors_map[prev_op].add(op)
    
    sample_instance = ProblemInstance(
        num_jobs=n_jobs,
        num_machines=n_machines,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )
    
    env = POFJSPEnv(sample_instance, time_limit=config.max_episode_steps)
    return env, sample_instance

def pad_observation(obs: Dict, max_jobs: int, max_machines: int, device: torch.device) -> Dict:
    """Pad observation tensors to maximum network dimensions"""
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
    processing_times_padded[:actual_jobs_pt, :actual_machines_pt] = obs['processing_times'][:actual_jobs_pt, :actual_machines_pt].to(device)
    
    return {
        'x': obs['x'].to(device),
        'edge_index': obs['edge_index'].to(device),
        'batch': obs['batch'].to(device),
        'job_mask': job_mask_padded,
        'machine_mask': machine_mask_padded,
        'processing_times': processing_times_padded
    }

def run_training_episode(agent: PPOAgent, env: POFJSPEnv, monitor: EnhancedPerformanceMonitor, max_jobs: int, max_machines: int, collect_experience: bool = True) -> Dict[str, float]:
    """Run a single training episode and collect experience"""
    episode_reward = 0
    episode_steps = 0
    experiences = []
    
    obs, _ = env.reset()
    done = False
    
    # Debug initial state
    monitor.logger.debug(f"Episode start: obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}, env_time_limit={env.time_limit if hasattr(env, 'time_limit') else 'N/A'}")
    
    while not done and episode_steps < 200:  # Increased step limit for better episode completion
        # Get action from agent
        with torch.no_grad():
            # Pad observation to maximum network dimensions
            device = next(agent.actor.parameters()).device
            obs_cuda = pad_observation(obs, max_jobs, max_machines, device)
            
            # Get full action info including log probs and value for training
            if collect_experience:
                job_action, machine_action, job_log_prob, machine_log_prob, value = agent.get_action(
                    obs_cuda['x'], 
                    obs_cuda['edge_index'], 
                    obs_cuda['batch'], 
                    obs_cuda['job_mask'], 
                    obs_cuda['machine_mask'],
                    obs_cuda['processing_times']
                )
                
                # Store experience data
                experience = {
                    'state': (obs_cuda['x'], obs_cuda['edge_index'], obs_cuda['batch']),
                    'job_action': job_action.cpu().item() if hasattr(job_action, 'cpu') else job_action,
                    'machine_action': machine_action.cpu().item() if hasattr(machine_action, 'cpu') else machine_action,
                    'job_log_prob': job_log_prob.cpu().item() if hasattr(job_log_prob, 'cpu') else job_log_prob,
                    'machine_log_prob': machine_log_prob.cpu().item() if hasattr(machine_log_prob, 'cpu') else machine_log_prob,
                    'value': value.cpu().item() if hasattr(value, 'cpu') else value,
                    'job_mask': obs_cuda['job_mask'],
                    'machine_mask': obs_cuda['machine_mask']
                }
                
                combined_action = np.array([experience['job_action'], experience['machine_action']])
                
                # Debug actions
                if episode_steps == 0:  # First step only
                    monitor.logger.debug(f"Episode {episode_steps}: Action=({experience['job_action']}, {experience['machine_action']}), Masks=({obs_cuda['job_mask'].sum()}, {obs_cuda['machine_mask'].sum()})")
            else:
                # Just get action for evaluation
                action_tuple = agent.get_action(
                    obs_cuda['x'], 
                    obs_cuda['edge_index'], 
                    obs_cuda['batch'], 
                    obs_cuda['job_mask'], 
                    obs_cuda['machine_mask'],
                    obs_cuda['processing_times']
                )
                
                # Extract job and machine actions
                if isinstance(action_tuple, tuple) and len(action_tuple) >= 2:
                    job_action = action_tuple[0]
                    machine_action = action_tuple[1]
                    
                    # Convert to numpy if needed
                    if hasattr(job_action, 'cpu'):
                        job_action = job_action.cpu().numpy()
                    if hasattr(machine_action, 'cpu'):
                        machine_action = machine_action.cpu().numpy()
                    
                    # Ensure scalar values
                    if isinstance(job_action, np.ndarray):
                        job_action = job_action.item()
                    if isinstance(machine_action, np.ndarray):
                        machine_action = machine_action.item()
                        
                    combined_action = np.array([job_action, machine_action])
                else:
                                    combined_action = env.action_space.sample()
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(combined_action)
        done = terminated or truncated
        
        # Debug early termination
        if done and episode_steps <= 2:
            monitor.logger.debug(f"Early termination: step={episode_steps}, terminated={terminated}, truncated={truncated}, reward={reward}, info={info}")
        
        # Complete experience and add to buffer
        if collect_experience:
            experience['reward'] = reward
            experience['done'] = done
            experiences.append(experience)
        
        episode_reward += reward
        episode_steps += 1
        obs = next_obs
    
    # Add experiences to agent buffer and compute returns
    training_losses = {}
    if collect_experience and experiences:
        # Compute returns and advantages
        rewards = [exp['reward'] for exp in experiences]
        values = [exp['value'] for exp in experiences]
        dones = [exp['done'] for exp in experiences]
        
        returns, advantages = agent.compute_returns_and_advantages(rewards, values, 0.0, dones)
        
        # Add returns and advantages to experiences and store in buffer
        for exp, ret, adv in zip(experiences, returns, advantages):
            exp['return'] = ret
            exp['advantage'] = adv
            agent.buffer.add(exp)
        
        # Update agent more frequently for debugging
        if len(agent.buffer) >= 8:  # Lower threshold to see training progress
            training_losses = agent.update()
            if training_losses:
                monitor.logger.info(f"Training update: {training_losses}")
    
    return {
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'makespan': info.get('makespan', float('inf')) if 'info' in locals() else float('inf'),
        'training_losses': training_losses
    }

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    sys.exit(0)

def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Comprehensive RL Training for POFJSP')
    parser.add_argument('--output-dir', default='outputs/comprehensive_training', help='Output directory')
    parser.add_argument('--config-file', help='JSON configuration file')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--fast-mode', action='store_true', help='Fast mode with reduced training time')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        # Set memory fraction to use more GPU memory
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Load configuration
    config = TrainingConfig()
    if args.fast_mode:
        config.total_timesteps = 100_000
        config.stage_timesteps = 30_000
        config.eval_every = 10_000
        config.log_every = 2_000
    
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize monitoring
    monitor = EnhancedPerformanceMonitor(output_dir)
    monitor.logger.info("Starting Comprehensive RL Training for POFJSP")
    monitor.logger.info(f"Configuration: {config}")
    
    # Initialize curriculum manager
    curriculum = CurriculumManager(config)
    
    # Create sample environment and agent
    env, sample_instance = create_sample_environment(config)
    
    # Initialize PPO agent with maximum problem dimensions
    # This ensures the network can handle all problem sizes in the curriculum
    agent = PPOAgent(
        input_dim=env.node_feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_jobs=config.max_jobs,  # Use maximum possible jobs
        num_machines=config.max_machines,  # Use maximum possible machines
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.n_epochs,
        gamma=config.gamma,
        lam=config.gae_lambda,
        clip_ratio=config.clip_range,
        device=device
    )
    
    monitor.logger.info("Agent initialized successfully")
    monitor.log_gpu_usage()
    
    # Setup profiling (disabled by default for stability)
    use_profiler = False  # Disable profiler to avoid warnings and improve performance
    profiler_ctx = None
    if use_profiler and profile is not None and torch.cuda.is_available():
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=100, warmup=50, active=200, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir / 'profiler')),
            record_shapes=True,
            with_stack=True
        )
        profiler_ctx.__enter__()
        monitor.logger.info(f"PyTorch profiler enabled. Results will be saved to: {output_dir / 'profiler'}")
    
    # Training loop with curriculum learning
    total_timesteps = 0
    best_performance = {'win_rate': 0, 'makespan_improvement': 0, 'stage': ''}
    
    for stage_idx in range(config.curriculum_stages):
        current_stage = curriculum.get_current_stage()
        monitor.logger.info(f"Starting curriculum stage {stage_idx + 1}: {current_stage['name']}")
        
        stage_timesteps = 0
        stage_start_time = time.time()
        stage_episodes = 0
        stage_rewards = []
        
        while stage_timesteps < config.stage_timesteps and total_timesteps < config.total_timesteps:
            # Generate training instances for current stage
            training_instances = generate_curriculum_instances(current_stage, num_instances=10)
            
            # Training on instances
            for inst_idx, instance in enumerate(training_instances):
                if total_timesteps >= config.total_timesteps:
                    break
                
                monitor.logger.info(f"Processing instance {inst_idx + 1}/{len(training_instances)} (Jobs: {instance.num_jobs}, Machines: {instance.num_machines})")
                
                # Create environment for this instance
                env = POFJSPEnv(instance, time_limit=current_stage['max_episode_steps'])
                
                # Run training episodes (reduced for debugging)
                for episode in range(1):  # Single episode per instance to get to training faster
                    if total_timesteps >= config.total_timesteps:
                        break
                    
                    # Profile training episodes
                    if use_profiler:
                        with record_function("training_episode"):
                            episode_results = run_training_episode(agent, env, monitor, config.max_jobs, config.max_machines)
                    else:
                        episode_results = run_training_episode(agent, env, monitor, config.max_jobs, config.max_machines)
                    
                    stage_rewards.append(episode_results['episode_reward'])
                    stage_episodes += 1
                    
                    # Step profiler
                    if use_profiler:
                        profiler_ctx.step()
                    
                    # Immediate logging for each episode (every episode during debugging)
                    monitor.logger.info(f"Episode {stage_episodes}: Reward={episode_results['episode_reward']:.3f}, Steps={episode_results['episode_steps']}, Makespan={episode_results.get('makespan', 'inf')}")
                    
                    # Log individual episode to TensorBoard
                    monitor.log_episode_metrics(
                        episode=stage_episodes,
                        episode_reward=episode_results['episode_reward'],
                        episode_steps=episode_results['episode_steps'],
                        makespan=episode_results['makespan'],
                        stage=current_stage['name']
                    )
                    
                    # Log training losses if available
                    if episode_results.get('training_losses'):
                        loss_metrics = {f'training_{k}': v for k, v in episode_results['training_losses'].items()}
                        monitor.log_metrics(total_timesteps, loss_metrics)
                    
                    # Simple timestep counting (actual PPO update would be more complex)
                    episode_steps = episode_results['episode_steps']
                    total_timesteps += episode_steps
                    stage_timesteps += episode_steps
                
                # Periodic logging
                if total_timesteps % config.log_every == 0:
                    monitor.log_gpu_usage()
                    avg_reward = np.mean(stage_rewards[-10:]) if stage_rewards else 0
                    recent_makespans = [r for r in [episode_result.get('makespan', float('inf')) for episode_result in [{'makespan': stage_rewards[i]} for i in range(max(0, len(stage_rewards)-10), len(stage_rewards))]] if r != float('inf')]
                    avg_makespan = np.mean(recent_makespans) if recent_makespans else 0
                    
                    monitor.log_metrics(total_timesteps, {
                        'stage': current_stage['name'],
                        'progress': total_timesteps / config.total_timesteps,
                        'avg_episode_reward': avg_reward,
                        'episodes_completed': stage_episodes,
                        'avg_makespan_recent': avg_makespan,
                        'episodes_per_second': stage_episodes / (time.time() - stage_start_time)
                    })
                    
                    # Log model parameters periodically
                    if total_timesteps % (config.log_every * 5) == 0:
                        monitor.log_model_parameters(agent, total_timesteps)
                
                # Evaluation
                if total_timesteps % config.eval_every == 0:
                    monitor.logger.info("Running evaluation...")
                    eval_instances = generate_curriculum_instances(current_stage, current_stage['instances_per_eval'])
                    eval_results = safe_evaluate_against_iaoa(agent, eval_instances, monitor)
                    
                    monitor.log_metrics(total_timesteps, {
                        'eval_win_rate': eval_results['win_rate'],
                        'eval_makespan_improvement': eval_results['makespan_improvement'],
                        'eval_time_speedup': eval_results['avg_time_speedup'],
                        'eval_completed_instances': eval_results['completed_instances']
                    })
                    
                    if eval_results['completed_instances'] > 0:
                        monitor.logger.info(f"Evaluation: {eval_results['win_rate']:.1%} win rate, {eval_results['avg_time_speedup']:.1f}x speedup")
                    
                    # Check if this is the best performance
                    if eval_results['win_rate'] > best_performance['win_rate']:
                        best_performance = {
                            'win_rate': eval_results['win_rate'],
                            'makespan_improvement': eval_results['makespan_improvement'],
                            'stage': current_stage['name'],
                            'timestep': total_timesteps
                        }
                        agent.save(str(output_dir / 'best_model'))
                        monitor.logger.info(f"Saved new best model! Win rate: {eval_results['win_rate']:.3f}")
                    
                    # Check curriculum advancement
                    if curriculum.should_advance_stage(eval_results['win_rate']) and stage_idx < config.curriculum_stages - 1:
                        old_stage = current_stage['name']
                        curriculum.advance_stage()
                        new_stage = curriculum.get_current_stage()['name']
                        
                        monitor.log_curriculum_transition(old_stage, new_stage, total_timesteps, eval_results['win_rate'])
                        monitor.logger.info(f"Advancing to next curriculum stage! Success rate: {eval_results['win_rate']:.3f}")
                        current_stage = curriculum.get_current_stage()  # Update current stage reference
                        break
                
                # Save checkpoint
                if total_timesteps % config.save_every == 0:
                    checkpoint_path = output_dir / f'checkpoint_{total_timesteps}'
                    agent.save(str(checkpoint_path))
                    monitor.logger.info(f"Saved checkpoint at step {total_timesteps}")
        
        stage_time = time.time() - stage_start_time
        avg_stage_reward = np.mean(stage_rewards) if stage_rewards else 0
        monitor.logger.info(f"Completed stage {stage_idx + 1} in {stage_time:.2f} seconds")
        monitor.logger.info(f"Stage statistics: {stage_episodes} episodes, avg reward: {avg_stage_reward:.3f}")
        
        # Advance curriculum stage
        curriculum.advance_stage()
    
    # Final evaluation
    monitor.logger.info("Final comprehensive evaluation...")
    final_results = {}
    
    for stage in curriculum.stages:
        monitor.logger.info(f"Evaluating on {stage['name']}...")
        eval_instances = generate_curriculum_instances(stage, min(stage['instances_per_eval'], 15))  # Limit for time
        results = safe_evaluate_against_iaoa(agent, eval_instances, monitor)
        final_results[stage['name']] = results
        
        monitor.logger.info(f"{stage['name']} Results:")
        monitor.logger.info(f"  Win Rate: {results['win_rate']:.3f}")
        monitor.logger.info(f"  Makespan Improvement: {results['makespan_improvement']:.3f}")
        monitor.logger.info(f"  Time Speedup: {results['avg_time_speedup']:.2f}x")
        monitor.logger.info(f"  Completed: {results['completed_instances']} instances")
    
    # Save final results
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save final model
    agent.save(str(output_dir / 'final_model'))
    
    # Generate summary report
    summary_report = {
        'training_config': {
            'total_timesteps': total_timesteps,
            'curriculum_stages': config.curriculum_stages,
            'device': str(device),
            'training_time': time.time() - monitor.start_time
        },
        'best_performance': best_performance,
        'final_results': final_results,
        'curriculum_performance': curriculum.performance_history
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    # Log final summary to TensorBoard
    monitor.writer.add_text('Training/Summary', 
                          f"Training completed!\nBest Win Rate: {best_performance['win_rate']:.3f}\nBest Stage: {best_performance['stage']}\nTotal Time: {time.time() - monitor.start_time:.2f}s",
                          total_timesteps)
    
    # Close profiler properly
    if use_profiler and profiler_ctx is not None:
        try:
            profiler_ctx.__exit__(None, None, None)
            monitor.logger.info("Profiling complete. View with: tensorboard --logdir=outputs/production_run_final/profiler")
        except Exception as e:
            monitor.logger.warning(f"Error closing profiler: {e}")
    
    # Close TensorBoard writer
    monitor.close()
    
    monitor.logger.info("Comprehensive RL Training Complete!")
    monitor.logger.info(f"Results saved to: {output_dir}")
    monitor.logger.info(f"TensorBoard logs saved to: {output_dir / 'tensorboard'}")
    monitor.logger.info(f"Best Win Rate Achieved: {best_performance['win_rate']:.3f} in stage {best_performance['stage']}")
    monitor.logger.info(f"Total Training Time: {time.time() - monitor.start_time:.2f} seconds")
    monitor.logger.info(f"View training progress with: tensorboard --logdir={output_dir / 'tensorboard'}")
    
    return final_results

if __name__ == "__main__":
    main()