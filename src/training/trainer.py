"""
Main Training Orchestrator

Coordinates all training components to provide a clean, modular
training system for POFJSP reinforcement learning.
"""

import time
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

import torch
import numpy as np

from training.config import TrainingConfig, get_training_config, config_manager
from training.curriculum import CurriculumManager, ProblemInstanceGenerator
from training.monitor import TrainingMonitor
from training.evaluator import ComprehensiveEvaluator
from rl.models.ppo_agent import PPOAgent
from rl.environments.pofjsp_env import POFJSPEnv
from problems.problem_instance import ProblemInstance
from validation import validate_inputs, Validators

logger = logging.getLogger(__name__)


class TrainingSession:
    """Manages a complete training session with all components."""
    
    def __init__(self, config: TrainingConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Initialize components
        self.curriculum_stages = config_manager.create_curriculum_stages(config)
        self.curriculum = CurriculumManager(config, self.curriculum_stages)
        self.problem_generator = ProblemInstanceGenerator(seed=42)
        self.monitor = TrainingMonitor(output_dir, config.__dict__)
        self.evaluator = ComprehensiveEvaluator(
            timeout_per_instance=config.timeout_per_instance,
            max_episode_steps=config.max_episode_steps
        )
        
        # Training state
        self.agent = None
        self.total_timesteps = 0
        self.episode_count = 0
        self.best_performance = {'win_rate': 0.0, 'stage': '', 'timestep': 0}
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info(f"Training session initialized")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Curriculum stages: {len(self.curriculum_stages)}")
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown on interruption."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Shutting down gracefully...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_agent(self) -> PPOAgent:
        """Initialize PPO agent with maximum problem dimensions."""
        # Create sample environment to get dimensions
        sample_instance = self._create_sample_instance()
        env = POFJSPEnv(sample_instance, time_limit=self.config.max_episode_steps)
        
        # Initialize with maximum dimensions for curriculum
        agent = PPOAgent(
            input_dim=env.node_feature_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_jobs=self.config.max_jobs,
            num_machines=self.config.max_machines,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            lam=self.config.gae_lambda,
            clip_ratio=self.config.clip_range,
            device=self.device
        )
        
        self.agent = agent
        logger.info("PPO agent initialized successfully")
        
        return agent
    
    def _create_sample_instance(self) -> ProblemInstance:
        """Create a sample problem instance for initialization."""
        sample_stage = self.curriculum.get_current_stage()
        instances = self.problem_generator.generate_curriculum_instances(sample_stage, 1)
        return instances[0]
    
    def run_training(self) -> Dict[str, Any]:
        """Run complete training session."""
        logger.info("Starting comprehensive training session")
        
        # Initialize agent
        if self.agent is None:
            self.initialize_agent()
        
        training_start_time = time.time()
        
        try:
            # Main training loop
            while (self.total_timesteps < self.config.total_timesteps and 
                   self.curriculum.current_stage_idx < len(self.curriculum_stages)):
                
                self._run_curriculum_stage()
                
                # Check if should advance curriculum
                should_advance, reason = self.curriculum.should_advance_stage()
                logger.info(f"Curriculum check: {should_advance} ({reason})")
                
                if should_advance:
                    old_stage = self.curriculum.get_current_stage()
                    advanced = self.curriculum.advance_stage()
                    
                    if advanced:
                        new_stage = self.curriculum.get_current_stage()
                        logger.info(f"Advanced curriculum: {old_stage.name} -> {new_stage.name}")
                        
                        # Log transition
                        recent_performance = list(self.curriculum.stage_performance[old_stage.name])
                        latest_success_rate = recent_performance[-1].success_rate if recent_performance else 0.0
                        self.monitor.log_curriculum_transition(old_stage.name, new_stage.name, latest_success_rate)
            
            # Final evaluation
            final_results = self._run_final_evaluation()
            
            training_time = time.time() - training_start_time
            
            # Create training summary
            summary = {
                'training_completed': True,
                'total_training_time': training_time,
                'total_timesteps': self.total_timesteps,
                'total_episodes': self.episode_count,
                'best_performance': self.best_performance,
                'final_evaluation': final_results,
                'curriculum_statistics': self.curriculum.get_stage_statistics(),
                'config': self.config.__dict__
            }
            
            # Save summary
            summary_file = self.output_dir / 'training_session_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Training completed successfully in {training_time:.1f}s")
            logger.info(f"Best performance: {self.best_performance['win_rate']:.3f} win rate "
                       f"at stage {self.best_performance['stage']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            self.cleanup()
    
    def _run_curriculum_stage(self):
        """Run training for the current curriculum stage."""
        current_stage = self.curriculum.get_current_stage()
        logger.info(f"Training on stage: {current_stage.name}")
        
        stage_start_time = time.time()
        stage_timesteps = 0
        stage_episodes = 0
        
        while (stage_timesteps < self.config.stage_timesteps and 
               self.total_timesteps < self.config.total_timesteps):
            
            # Generate training instances
            training_instances = self.problem_generator.generate_curriculum_instances(
                current_stage, num_instances=10
            )
            
            # Train on instances
            for instance in training_instances:
                if self.total_timesteps >= self.config.total_timesteps:
                    break
                
                episode_results = self._run_training_episode(instance, current_stage)
                
                # Update counters
                self.episode_count += 1
                episode_timesteps = episode_results['episode_steps']
                self.total_timesteps += episode_timesteps
                stage_timesteps += episode_timesteps
                stage_episodes += 1
                
                # Log episode
                self.monitor.log_episode(
                    episode_reward=episode_results['episode_reward'],
                    episode_steps=episode_results['episode_steps'],
                    episode_makespan=episode_results['makespan'],
                    stage=current_stage.name,
                    **episode_results.get('additional_metrics', {})
                )
                
                # Log training step if available
                if 'training_losses' in episode_results and episode_results['training_losses']:
                    losses = episode_results['training_losses']
                    self.monitor.log_training_step(
                        policy_loss=losses.get('policy_loss', 0.0),
                        value_loss=losses.get('value_loss', 0.0),
                        entropy_loss=losses.get('entropy_loss', 0.0),
                        total_loss=losses.get('total_loss', 0.0)
                    )
                
                # Periodic logging
                if self.total_timesteps % self.config.log_every == 0:
                    self.monitor.log_gpu_usage()
                    
                    # Log model parameters periodically
                    if self.total_timesteps % (self.config.log_every * 5) == 0:
                        self.monitor.log_model_parameters(self.agent)
                
                # Evaluation
                if self.total_timesteps % self.config.eval_every == 0:
                    self._run_evaluation(current_stage)
                
                # Save checkpoint
                if self.total_timesteps % self.config.save_every == 0:
                    self._save_checkpoint()
        
        stage_time = time.time() - stage_start_time
        logger.info(f"Completed stage {current_stage.name} in {stage_time:.1f}s "
                   f"({stage_episodes} episodes)")
    
    def _run_training_episode(self, instance: ProblemInstance, 
                            stage_config) -> Dict[str, Any]:
        """Run a single training episode."""
        # Create environment
        env = POFJSPEnv(instance, time_limit=stage_config.max_episode_steps)
        
        episode_reward = 0.0
        episode_steps = 0
        experiences = []
        
        obs, _ = env.reset()
        done = False
        
        while not done and episode_steps < stage_config.max_episode_steps:
            try:
                # Get action with training info
                with torch.no_grad():
                    obs_padded = self._pad_observation(obs)
                    
                    action_result = self.agent.get_action(
                        obs_padded['x'],
                        obs_padded['edge_index'],
                        obs_padded['batch'],
                        obs_padded['job_mask'],
                        obs_padded['machine_mask'],
                        obs_padded['processing_times']
                    )
                    
                    # Extract action components
                    if len(action_result) >= 5:  # Full training info
                        job_action, machine_action, job_log_prob, machine_log_prob, value = action_result
                        
                        # Store experience
                        experience = {
                            'state': (obs_padded['x'], obs_padded['edge_index'], obs_padded['batch']),
                            'action': (job_action, machine_action),
                            'log_prob': (job_log_prob, machine_log_prob),
                            'value': value,
                            'masks': (obs_padded['job_mask'], obs_padded['machine_mask'])
                        }
                        
                        # Convert action for environment
                        job_val = job_action.cpu().item() if hasattr(job_action, 'cpu') else job_action
                        machine_val = machine_action.cpu().item() if hasattr(machine_action, 'cpu') else machine_action
                        action = np.array([job_val, machine_val])
                        
                    else:
                        # Evaluation mode or error
                        action = env.action_space.sample()
                        experience = None
                
                # Take environment step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Complete experience if available
                if experience is not None:
                    experience['reward'] = reward
                    experience['done'] = done
                    experiences.append(experience)
                
                episode_reward += reward
                episode_steps += 1
                obs = next_obs
                
            except Exception as e:
                logger.debug(f"Episode step error: {e}")
                break
        
        # Process experiences for training
        training_losses = {}
        if experiences and len(experiences) > 5:  # Minimum batch size
            training_losses = self._process_experiences(experiences)
        
        makespan = info.get('makespan', float('inf')) if info else float('inf')
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'makespan': makespan,
            'training_losses': training_losses,
            'experiences_collected': len(experiences)
        }
    
    def _process_experiences(self, experiences: List[Dict]) -> Dict[str, float]:
        """Process collected experiences for training."""
        try:
            # Extract data from experiences
            rewards = [exp['reward'] for exp in experiences]
            values = [exp['value'].cpu().item() if hasattr(exp['value'], 'cpu') else exp['value'] 
                     for exp in experiences]
            dones = [exp['done'] for exp in experiences]
            
            # Compute returns and advantages
            returns, advantages = self.agent.compute_returns_and_advantages(
                rewards, values, 0.0, dones
            )
            
            # Add to agent buffer
            for exp, ret, adv in zip(experiences, returns, advantages):
                exp['return'] = ret
                exp['advantage'] = adv
                self.agent.buffer.add(exp)
            
            # Update agent if buffer has enough data
            if len(self.agent.buffer) >= self.config.batch_size // 4:  # Quarter batch size
                return self.agent.update()
            
            return {}
            
        except Exception as e:
            logger.debug(f"Experience processing error: {e}")
            return {}
    
    def _pad_observation(self, obs: Dict) -> Dict:
        """Pad observation to maximum network dimensions."""
        device = self.device
        max_jobs = self.config.max_jobs
        max_machines = self.config.max_machines
        
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
    
    def _run_evaluation(self, stage_config):
        """Run evaluation against baseline."""
        logger.info("Running evaluation...")
        
        # Generate evaluation instances
        eval_instances = self.problem_generator.generate_curriculum_instances(
            stage_config, stage_config.instances_per_eval
        )
        
        # Run evaluation
        comparison = self.evaluator.evaluate_agent_vs_baseline(
            self.agent, eval_instances, 
            self.config.max_jobs, self.config.max_machines,
            iaoa_pop_size=20, iaoa_max_iter=10
        )
        
        # Log results
        self.monitor.log_evaluation(
            win_rate=comparison.win_rate,
            makespan_improvement=comparison.makespan_improvement,
            time_speedup=comparison.time_speedup,
            completed_instances=comparison.agent1_result.completed_instances
        )
        
        # Record curriculum performance
        self.curriculum.record_performance(
            success_rate=comparison.win_rate,
            avg_makespan=comparison.agent1_result.avg_makespan,
            avg_episode_steps=np.mean(comparison.agent1_result.episode_lengths) 
            if comparison.agent1_result.episode_lengths else 0,
            win_rate=comparison.win_rate,
            makespan_improvement=comparison.makespan_improvement
        )
        
        # Check if best performance
        if comparison.win_rate > self.best_performance['win_rate']:
            self.best_performance = {
                'win_rate': comparison.win_rate,
                'makespan_improvement': comparison.makespan_improvement,
                'stage': stage_config.name,
                'timestep': self.total_timesteps
            }
            
            # Save best model
            best_model_path = self.output_dir / 'best_model'
            self.agent.save(str(best_model_path))
            logger.info(f"New best model saved! Win rate: {comparison.win_rate:.3f}")
        
        logger.info(f"Evaluation completed: win_rate={comparison.win_rate:.3f}, "
                   f"improvement={comparison.makespan_improvement:.3f}")
    
    def _run_final_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive final evaluation on all stages."""
        logger.info("Running final comprehensive evaluation...")
        
        final_results = {}
        
        for stage in self.curriculum_stages:
            logger.info(f"Final evaluation on {stage.name}...")
            
            # Generate evaluation instances
            eval_instances = self.problem_generator.generate_curriculum_instances(
                stage, min(stage.instances_per_eval, 20)  # Limit for time
            )
            
            # Run evaluation
            comparison = self.evaluator.evaluate_agent_vs_baseline(
                self.agent, eval_instances,
                self.config.max_jobs, self.config.max_machines
            )
            
            # Store results
            stage_results = {
                'win_rate': comparison.win_rate,
                'makespan_improvement': comparison.makespan_improvement,
                'time_speedup': comparison.time_speedup,
                'rl_success_rate': comparison.agent1_result.success_rate,
                'baseline_success_rate': comparison.agent2_result.success_rate,
                'completed_instances': comparison.agent1_result.completed_instances,
                'total_instances': comparison.agent1_result.num_instances
            }
            
            final_results[stage.name] = stage_results
            
            logger.info(f"{stage.name} final results: "
                       f"win_rate={comparison.win_rate:.3f}, "
                       f"improvement={comparison.makespan_improvement:.3f}")
        
        return final_results
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f'checkpoint_{self.total_timesteps}'
        
        checkpoint_data = {
            'agent_state': self.agent.state_dict() if self.agent else None,
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
            'curriculum_state': {
                'current_stage_idx': self.curriculum.current_stage_idx,
                'performance_history': self.curriculum.performance_history
            },
            'best_performance': self.best_performance,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint_data, str(checkpoint_path) + '.pt')
        logger.info(f"Checkpoint saved: {checkpoint_path}.pt")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint."""
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            
            # Restore agent if available
            if checkpoint_data.get('agent_state') and self.agent:
                self.agent.load_state_dict(checkpoint_data['agent_state'])
            
            # Restore training state
            self.total_timesteps = checkpoint_data.get('total_timesteps', 0)
            self.episode_count = checkpoint_data.get('episode_count', 0)
            self.best_performance = checkpoint_data.get('best_performance', self.best_performance)
            
            # Restore curriculum state
            curriculum_state = checkpoint_data.get('curriculum_state', {})
            self.curriculum.current_stage_idx = curriculum_state.get('current_stage_idx', 0)
            self.curriculum.performance_history = curriculum_state.get('performance_history', [])
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"  Timesteps: {self.total_timesteps}")
            logger.info(f"  Episodes: {self.episode_count}")
            logger.info(f"  Current stage: {self.curriculum.get_current_stage().name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.monitor:
                self.monitor.close()
            logger.info("Training session cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


class TrainingOrchestrator:
    """High-level training orchestrator with preset configurations."""
    
    @staticmethod
    def run_debug_training(output_dir: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Run debug training for development."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        config = get_training_config('debug')
        session = TrainingSession(config, output_dir, device)
        
        return session.run_training()
    
    @staticmethod
    def run_fast_training(output_dir: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Run fast training for testing."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        config = get_training_config('fast')
        session = TrainingSession(config, output_dir, device)
        
        return session.run_training()
    
    @staticmethod
    def run_production_training(output_dir: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Run production training."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        config = get_training_config('production')
        session = TrainingSession(config, output_dir, device)
        
        return session.run_training()
    
    @staticmethod
    def run_custom_training(config: TrainingConfig, output_dir: str, 
                          device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Run training with custom configuration."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        session = TrainingSession(config, output_dir, device)
        
        return session.run_training()


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import argparse
    
    parser = argparse.ArgumentParser(description='POFJSP RL Training')
    parser.add_argument('--mode', choices=['debug', 'fast', 'production'], 
                       default='debug', help='Training mode')
    parser.add_argument('--output-dir', default='outputs/training_test', 
                       help='Output directory')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run training
    if args.mode == 'debug':
        results = TrainingOrchestrator.run_debug_training(args.output_dir, device)
    elif args.mode == 'fast':
        results = TrainingOrchestrator.run_fast_training(args.output_dir, device)
    else:
        results = TrainingOrchestrator.run_production_training(args.output_dir, device)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best win rate: {results['best_performance']['win_rate']:.3f}")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Training time: {results['total_training_time']:.1f}s")