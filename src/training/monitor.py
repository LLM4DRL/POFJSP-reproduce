"""
Training Monitoring and Logging

Comprehensive monitoring system for RL training with TensorBoard integration,
performance tracking, and detailed logging capabilities.
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from performance.monitor import ResourceMonitor, PerformanceMetrics
from validation import validate_inputs, Validators

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific timestep."""
    timestep: int
    episode: int
    stage: str
    
    # Episode metrics
    episode_reward: float = 0.0
    episode_steps: int = 0
    episode_makespan: float = float('inf')
    
    # Training metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    total_loss: float = 0.0
    
    # Performance metrics
    gpu_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    
    # Evaluation metrics
    eval_win_rate: Optional[float] = None
    eval_makespan_improvement: Optional[float] = None
    eval_time_speedup: Optional[float] = None
    
    # Additional metrics
    additional: Dict[str, Any] = field(default_factory=dict)


class TensorBoardLogger:
    """Enhanced TensorBoard logging with hierarchical organization."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))
        self.episode_count = 0
        
        logger.info(f"TensorBoard writer initialized: {log_dir / 'tensorboard'}")
        logger.info(f"Launch with: tensorboard --logdir={log_dir / 'tensorboard'}")
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """Log training-related metrics."""
        step = metrics.timestep
        
        # Episode metrics
        if metrics.episode_reward != 0:
            self.writer.add_scalar('Training/Episode_Reward', metrics.episode_reward, step)
        if metrics.episode_steps > 0:
            self.writer.add_scalar('Training/Episode_Steps', metrics.episode_steps, step)
        if metrics.episode_makespan != float('inf'):
            self.writer.add_scalar('Training/Episode_Makespan', metrics.episode_makespan, step)
        
        # Loss metrics
        if metrics.policy_loss != 0:
            self.writer.add_scalar('Training/Policy_Loss', metrics.policy_loss, step)
        if metrics.value_loss != 0:
            self.writer.add_scalar('Training/Value_Loss', metrics.value_loss, step)
        if metrics.entropy_loss != 0:
            self.writer.add_scalar('Training/Entropy_Loss', metrics.entropy_loss, step)
        if metrics.total_loss != 0:
            self.writer.add_scalar('Training/Total_Loss', metrics.total_loss, step)
        
        # Curriculum stage
        stage_map = {'small_problems': 1, 'medium_problems': 2, 'large_problems': 3}
        stage_num = stage_map.get(metrics.stage, 0)
        self.writer.add_scalar('Curriculum/Current_Stage', stage_num, step)
        
        self.writer.flush()
    
    def log_evaluation_metrics(self, metrics: TrainingMetrics):
        """Log evaluation-related metrics."""
        step = metrics.timestep
        
        if metrics.eval_win_rate is not None:
            self.writer.add_scalar('Evaluation/Win_Rate', metrics.eval_win_rate, step)
        if metrics.eval_makespan_improvement is not None:
            self.writer.add_scalar('Evaluation/Makespan_Improvement', metrics.eval_makespan_improvement, step)
        if metrics.eval_time_speedup is not None:
            self.writer.add_scalar('Evaluation/Time_Speedup', metrics.eval_time_speedup, step)
        
        self.writer.flush()
    
    def log_performance_metrics(self, metrics: TrainingMetrics):
        """Log system performance metrics."""
        step = metrics.timestep
        
        if metrics.gpu_memory_mb > 0:
            self.writer.add_scalar('Performance/GPU_Memory_MB', metrics.gpu_memory_mb, step)
        if metrics.cpu_percent > 0:
            self.writer.add_scalar('Performance/CPU_Percent', metrics.cpu_percent, step)
        
        self.writer.flush()
    
    def log_model_parameters(self, model: torch.nn.Module, step: int, prefix: str = ""):
        """Log model parameters and gradients."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'{prefix}Gradients/{name}', param.grad, step)
            self.writer.add_histogram(f'{prefix}Parameters/{name}', param, step)
        
        self.writer.flush()
    
    def log_curriculum_transition(self, old_stage: str, new_stage: str, 
                                 step: int, success_rate: float):
        """Log curriculum stage transitions."""
        transition_text = f'Stage transition: {old_stage} -> {new_stage} (success rate: {success_rate:.3f})'
        self.writer.add_text('Curriculum/Transitions', transition_text, step)
        self.writer.flush()
    
    def log_custom_metrics(self, metrics: Dict[str, float], step: int, category: str = "Custom"):
        """Log custom metrics under specified category."""
        for name, value in metrics.items():
            self.writer.add_scalar(f'{category}/{name}', value, step)
        self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


class TrainingMonitor:
    """Comprehensive training monitoring system."""
    
    def __init__(self, output_dir: Union[str, Path], config: Optional[Dict] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tensorboard = TensorBoardLogger(self.output_dir)
        self.resource_monitor = ResourceMonitor(sampling_interval=1.0)
        
        # Metrics storage
        self.metrics_history = []
        self.episode_metrics = deque(maxlen=1000)
        self.evaluation_history = []
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Configuration
        self.config = config or {}
        
        # Statistics tracking
        self.episode_count = 0
        self.total_timesteps = 0
        self.current_stage = "initialization"
        
        # Setup file logging
        self._setup_logging()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        logger.info(f"Training monitor initialized: {self.output_dir}")
        logger.info(f"TensorBoard: tensorboard --logdir={self.output_dir / 'tensorboard'}")
    
    def _setup_logging(self):
        """Setup file logging configuration."""
        log_file = self.output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        logger.info(f"File logging enabled: {log_file}")
    
    def log_episode(self, episode_reward: float, episode_steps: int, 
                   episode_makespan: float, stage: str, **additional_metrics):
        """Log metrics for a completed episode."""
        self.episode_count += 1
        self.total_timesteps += episode_steps
        self.current_stage = stage
        
        # Create metrics object
        metrics = TrainingMetrics(
            timestep=self.total_timesteps,
            episode=self.episode_count,
            stage=stage,
            episode_reward=episode_reward,
            episode_steps=episode_steps,
            episode_makespan=episode_makespan,
            additional=additional_metrics
        )
        
        # Add resource metrics
        if torch.cuda.is_available():
            metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Store metrics
        self.episode_metrics.append(metrics)
        self.metrics_history.append(metrics)
        
        # Log to TensorBoard
        self.tensorboard.log_training_metrics(metrics)
        
        # Log to console (periodic)
        if self.episode_count % 10 == 0:
            self._log_episode_summary()
        
        # Save metrics to file (periodic)
        if self.episode_count % 50 == 0:
            self._save_metrics_to_file()
    
    def log_training_step(self, policy_loss: float, value_loss: float, 
                         entropy_loss: float, total_loss: float, **additional_metrics):
        """Log metrics for a training step."""
        metrics = TrainingMetrics(
            timestep=self.total_timesteps,
            episode=self.episode_count,
            stage=self.current_stage,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            total_loss=total_loss,
            additional=additional_metrics
        )
        
        # Add resource metrics
        resource_metrics = self.resource_monitor.stop_monitoring()
        self.resource_monitor.start_monitoring()  # Restart monitoring
        
        metrics.gpu_memory_mb = resource_metrics.get('memory_peak_mb', 0)
        metrics.cpu_percent = resource_metrics.get('cpu_avg_percent', 0)
        
        # Log to TensorBoard
        self.tensorboard.log_training_metrics(metrics)
        self.tensorboard.log_performance_metrics(metrics)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        logger.debug(f"Training step: policy_loss={policy_loss:.4f}, "
                    f"value_loss={value_loss:.4f}, total_loss={total_loss:.4f}")
    
    def log_evaluation(self, win_rate: float, makespan_improvement: float, 
                      time_speedup: float, completed_instances: int, **additional_metrics):
        """Log evaluation results."""
        metrics = TrainingMetrics(
            timestep=self.total_timesteps,
            episode=self.episode_count,
            stage=self.current_stage,
            eval_win_rate=win_rate,
            eval_makespan_improvement=makespan_improvement,
            eval_time_speedup=time_speedup,
            additional={'completed_instances': completed_instances, **additional_metrics}
        )
        
        # Store evaluation
        eval_record = {
            'timestep': self.total_timesteps,
            'episode': self.episode_count,
            'stage': self.current_stage,
            'win_rate': win_rate,
            'makespan_improvement': makespan_improvement,
            'time_speedup': time_speedup,
            'completed_instances': completed_instances,
            'timestamp': time.time(),
            **additional_metrics
        }
        self.evaluation_history.append(eval_record)
        
        # Log to TensorBoard
        self.tensorboard.log_evaluation_metrics(metrics)
        
        # Log to console
        logger.info(f"Evaluation completed - Win Rate: {win_rate:.3f}, "
                   f"Makespan Improvement: {makespan_improvement:.3f}, "
                   f"Time Speedup: {time_speedup:.2f}x")
        
        # Save evaluation history
        self._save_evaluation_history()
    
    def log_curriculum_transition(self, old_stage: str, new_stage: str, success_rate: float):
        """Log curriculum stage transition."""
        self.current_stage = new_stage
        
        # Log to TensorBoard
        self.tensorboard.log_curriculum_transition(
            old_stage, new_stage, self.total_timesteps, success_rate
        )
        
        # Log to console
        logger.info(f"Curriculum transition: {old_stage} -> {new_stage} "
                   f"(success rate: {success_rate:.3f})")
        
        # Record transition
        transition_record = {
            'timestep': self.total_timesteps,
            'episode': self.episode_count,
            'old_stage': old_stage,
            'new_stage': new_stage,
            'success_rate': success_rate,
            'timestamp': time.time()
        }
        
        # Save to transitions file
        transitions_file = self.output_dir / 'curriculum_transitions.json'
        if transitions_file.exists():
            with open(transitions_file, 'r') as f:
                transitions = json.load(f)
        else:
            transitions = []
        
        transitions.append(transition_record)
        
        with open(transitions_file, 'w') as f:
            json.dump(transitions, f, indent=2)
    
    def log_model_parameters(self, agent, step: Optional[int] = None):
        """Log model parameters and gradients."""
        if step is None:
            step = self.total_timesteps
        
        if hasattr(agent, 'actor'):
            self.tensorboard.log_model_parameters(agent.actor, step, "Actor_")
        if hasattr(agent, 'critic'):
            self.tensorboard.log_model_parameters(agent.critic, step, "Critic_")
    
    def log_gpu_usage(self):
        """Log detailed GPU usage information."""
        if not torch.cuda.is_available():
            return
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            # Log to TensorBoard
            self.tensorboard.log_custom_metrics({
                'Memory_Allocated_GB': memory_allocated,
                'Memory_Reserved_GB': memory_reserved
            }, self.total_timesteps, 'GPU')
            
            # Log to console if high usage
            if memory_allocated > 4.0:  # More than 4GB
                logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, "
                           f"{memory_reserved:.2f}GB reserved")
            
            # Clear cache if very high usage
            if memory_allocated > 6.0:
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared due to high memory usage")
                
        except Exception as e:
            logger.warning(f"Error logging GPU usage: {e}")
    
    def _log_episode_summary(self):
        """Log periodic episode summary."""
        if not self.episode_metrics:
            return
        
        recent_episodes = list(self.episode_metrics)[-10:]  # Last 10 episodes
        
        avg_reward = np.mean([e.episode_reward for e in recent_episodes])
        avg_steps = np.mean([e.episode_steps for e in recent_episodes])
        valid_makespans = [e.episode_makespan for e in recent_episodes 
                          if e.episode_makespan != float('inf')]
        avg_makespan = np.mean(valid_makespans) if valid_makespans else float('inf')
        
        elapsed_time = time.time() - self.start_time
        episodes_per_minute = self.episode_count / (elapsed_time / 60) if elapsed_time > 0 else 0
        
        logger.info(f"Episode {self.episode_count} Summary:")
        logger.info(f"  Stage: {self.current_stage}")
        logger.info(f"  Avg reward (last 10): {avg_reward:.3f}")
        logger.info(f"  Avg steps (last 10): {avg_steps:.1f}")
        logger.info(f"  Avg makespan (last 10): {avg_makespan:.2f}")
        logger.info(f"  Episodes/minute: {episodes_per_minute:.1f}")
        logger.info(f"  Total timesteps: {self.total_timesteps:,}")
        logger.info(f"  Elapsed time: {elapsed_time:.1f}s")
    
    def _save_metrics_to_file(self):
        """Save metrics history to JSON file."""
        metrics_file = self.output_dir / 'training_metrics.json'
        
        # Convert metrics to serializable format
        serializable_metrics = []
        for m in self.metrics_history[-100:]:  # Last 100 entries
            metric_dict = {
                'timestep': m.timestep,
                'episode': m.episode,
                'stage': m.stage,
                'episode_reward': m.episode_reward,
                'episode_steps': m.episode_steps,
                'episode_makespan': m.episode_makespan if m.episode_makespan != float('inf') else None,
                'policy_loss': m.policy_loss,
                'value_loss': m.value_loss,
                'entropy_loss': m.entropy_loss,
                'total_loss': m.total_loss,
                'gpu_memory_mb': m.gpu_memory_mb,
                'cpu_percent': m.cpu_percent,
                'eval_win_rate': m.eval_win_rate,
                'eval_makespan_improvement': m.eval_makespan_improvement,
                'eval_time_speedup': m.eval_time_speedup,
                'additional': m.additional
            }
            serializable_metrics.append(metric_dict)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def _save_evaluation_history(self):
        """Save evaluation history to JSON file."""
        eval_file = self.output_dir / 'evaluation_history.json'
        
        with open(eval_file, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        elapsed_time = time.time() - self.start_time
        
        # Calculate recent performance
        recent_episodes = list(self.episode_metrics)[-50:] if self.episode_metrics else []
        recent_evaluations = self.evaluation_history[-10:] if self.evaluation_history else []
        
        summary = {
            'training_time': elapsed_time,
            'total_episodes': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'current_stage': self.current_stage,
            'episodes_per_minute': self.episode_count / (elapsed_time / 60) if elapsed_time > 0 else 0,
            'recent_performance': {
                'avg_episode_reward': np.mean([e.episode_reward for e in recent_episodes]) if recent_episodes else 0,
                'avg_episode_steps': np.mean([e.episode_steps for e in recent_episodes]) if recent_episodes else 0,
                'success_rate': len([e for e in recent_episodes if e.episode_makespan != float('inf')]) / len(recent_episodes) if recent_episodes else 0
            },
            'evaluation_summary': {
                'total_evaluations': len(self.evaluation_history),
                'best_win_rate': max([e['win_rate'] for e in recent_evaluations], default=0),
                'latest_win_rate': recent_evaluations[-1]['win_rate'] if recent_evaluations else 0,
                'avg_time_speedup': np.mean([e['time_speedup'] for e in recent_evaluations]) if recent_evaluations else 0
            }
        }
        
        return summary
    
    def save_final_summary(self):
        """Save final training summary."""
        summary = self.get_training_summary()
        summary['config'] = self.config
        summary['final_timestamp'] = time.time()
        
        summary_file = self.output_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Final training summary saved: {summary_file}")
    
    def close(self):
        """Close monitoring systems and save final data."""
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Save final data
        self._save_metrics_to_file()
        self._save_evaluation_history()
        self.save_final_summary()
        
        # Close TensorBoard
        self.tensorboard.close()
        
        logger.info("Training monitor closed successfully")


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import shutil
    
    # Test monitoring system
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = TrainingMonitor(temp_dir, config={'test': True})
        
        # Simulate training episodes
        for episode in range(20):
            reward = np.random.normal(10, 2)
            steps = np.random.randint(50, 150)
            makespan = np.random.normal(100, 10) if np.random.random() > 0.1 else float('inf')
            stage = 'small_problems' if episode < 10 else 'medium_problems'
            
            monitor.log_episode(reward, steps, makespan, stage)
            
            # Simulate training step
            if episode % 5 == 0:
                monitor.log_training_step(
                    policy_loss=np.random.random() * 0.1,
                    value_loss=np.random.random() * 0.1,
                    entropy_loss=np.random.random() * 0.01,
                    total_loss=np.random.random() * 0.2
                )
            
            # Simulate evaluation
            if episode % 8 == 0:
                monitor.log_evaluation(
                    win_rate=np.random.random() * 0.8,
                    makespan_improvement=np.random.random() * 0.2,
                    time_speedup=1 + np.random.random() * 5,
                    completed_instances=np.random.randint(15, 25)
                )
            
            # Simulate curriculum transition
            if episode == 10:
                monitor.log_curriculum_transition('small_problems', 'medium_problems', 0.75)
        
        # Get summary
        summary = monitor.get_training_summary()
        print("Training Summary:")
        print(f"  Episodes: {summary['total_episodes']}")
        print(f"  Timesteps: {summary['total_timesteps']}")
        print(f"  Training time: {summary['training_time']:.1f}s")
        print(f"  Episodes/min: {summary['episodes_per_minute']:.1f}")
        print(f"  Recent avg reward: {summary['recent_performance']['avg_episode_reward']:.3f}")
        print(f"  Success rate: {summary['recent_performance']['success_rate']:.3f}")
        
        monitor.close()
        print(f"Test completed. Files created in: {temp_dir}")