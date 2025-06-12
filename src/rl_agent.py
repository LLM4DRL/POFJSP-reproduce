"""
Reinforcement Learning Agent for POFJSP using Stable Baselines3

This module implements a PPO agent for solving the POFJSP problem.
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
import torch as th
from tqdm.auto import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy

from algorithms import ProblemInstance, Solution
from src.rl_env import POFJSPEnv


# We no longer need the custom policy since we're using a standard Discrete action space


class SaveBestSolutionCallback(BaseCallback):
    """Callback to save the best solution found during training."""
    
    def __init__(self, eval_env, verbose=0):
        super(SaveBestSolutionCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.best_makespan = float('inf')
        self.best_solution = None
    
    def _on_step(self) -> bool:
        """Called at each step during training."""
        # Periodically evaluate the agent
        if self.n_calls % 1000 == 0:
            # Run evaluation episode
            obs, _ = self.eval_env.reset()
            terminated = truncated = False
            
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
            
            # Check if better solution found
            current_makespan = info.get("makespan", float('inf'))
            if current_makespan < self.best_makespan:
                self.best_makespan = current_makespan
                self.best_solution = self.eval_env.get_solution()
                if self.verbose > 0:
                    print(f"New best makespan: {self.best_makespan}")
        
        return True


class TQDMCallback(BaseCallback):
    """
    Custom callback for updating a tqdm progress bar during training.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        
    def _on_training_start(self) -> None:
        """Initialize progress bar at the start of training."""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        """Update progress bar on each step."""
        # Calculate the steps taken since the last update
        if self.n_calls > 1:
            step_size = self.n_calls - self.pbar.n
            self.pbar.update(step_size)
        else:
            self.pbar.update(1)
        return True
        
    def _on_training_end(self) -> None:
        """Close the progress bar when training ends."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class POFJSPAgent:
    """
    RL agent for solving POFJSP using PPO.
    
    This class handles:
    - Agent initialization and training
    - Model saving and loading
    - Solution generation and evaluation
    """
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize the PPO agent.
        
        Args:
            learning_rate: Learning rate for the optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epoch when optimizing the surrogate loss
            gamma: Discount factor
            device: Device to run the model on ("auto", "cpu", "cuda")
            seed: Random seed
            verbose: Whether to print verbose output
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.device = device
        self.seed = seed
        self.verbose = verbose
        
        # Will be initialized later
        self.model = None
        self.env = None
        self.best_solution = None
        self.best_makespan = float('inf')
    
    def _make_env(self, problem_instance: ProblemInstance, seed: int) -> POFJSPEnv:
        """Create a POFJSP environment."""
        env = POFJSPEnv(problem_instance, time_limit=problem_instance.total_operations * 2)
        # Seed handled in reset method in Gymnasium
        return env
    
    def setup_environment(self, problem_instance: ProblemInstance, n_envs: int = 4) -> None:
        """Set up the environment for training."""
        # Create vectorized environments for parallel training
        env_fns = [lambda: self._make_env(problem_instance, self.seed + i) for i in range(n_envs)]
        
        if n_envs > 1:
            self.env = SubprocVecEnv(env_fns)
        else:
            self.env = DummyVecEnv(env_fns)
        
        # Create a single environment for evaluation
        self.eval_env = self._make_env(problem_instance, self.seed)
    
    def setup_model(self) -> None:
        """Initialize the PPO model."""
        if self.env is None:
            raise ValueError("Environment must be set up before model initialization")
        
        # Create policy kwargs if needed
        # policy_kwargs = dict(
        #     net_arch=[128, 128, dict(pi=[64, 64], vf=[64, 64])]
        # )
        
        # Setup TensorBoard logging
        tensorboard_log = "logs/tensorboard/"
        os.makedirs(tensorboard_log, exist_ok=True)
        
        # Simple policy network configuration
        policy_kwargs = dict(
            net_arch=[64, 64]
        )
        
        # Initialize the model with standard MlpPolicy
        self.model = PPO(
            "MlpPolicy",  # Standard policy for Discrete action spaces
            self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            verbose=1 if self.verbose else 0,
            device=self.device,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
        )
    
    def train(self, total_timesteps: int = 1000000) -> None:
        """Train the agent."""
        if self.model is None:
            raise ValueError("Model must be set up before training")
        
        # Create callbacks for best solution saving and progress tracking
        save_callback = SaveBestSolutionCallback(self.eval_env, verbose=1 if self.verbose else 0)
        tqdm_callback = TQDMCallback(total_timesteps=total_timesteps)
        callbacks = [save_callback, tqdm_callback]
        
        # Train the model
        start_time = time.time()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
        training_time = time.time() - start_time
        
        # Store best solution
        self.best_solution = save_callback.best_solution
        self.best_makespan = save_callback.best_makespan
        
        if self.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Best makespan: {self.best_makespan}")
    
    def save(self, path: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save the model
        self.model.save(path)
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load a trained model."""
        self.model = PPO.load(path)
        if self.verbose:
            print(f"Model loaded from {path}")
    
    def solve(self, problem_instance: ProblemInstance) -> Solution:
        """Solve a POFJSP instance using the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before solving")
        
        # Create environment for the problem
        env = self._make_env(problem_instance, self.seed)
        
        # Solve the problem
        obs = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
        
        # Get the solution
        solution = env.get_solution()
        
        return solution


def train_ppo_agent(
    problem_instance: ProblemInstance,
    total_timesteps: int = 1000000,
    n_envs: int = 4,
    save_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[POFJSPAgent, Solution]:
    """
    Train a PPO agent for a POFJSP instance.
    
    Args:
        problem_instance: The POFJSP instance to solve
        total_timesteps: Total number of timesteps for training
        n_envs: Number of parallel environments
        save_path: Path to save the trained model (optional)
        verbose: Whether to print verbose output
        
    Returns:
        agent: Trained PPO agent
        solution: Best solution found
    """
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create and setup agent
    agent = POFJSPAgent(verbose=verbose)
    agent.setup_environment(problem_instance, n_envs=n_envs)
    agent.setup_model()
    
    # Print summary info before training
    print(f"Starting RL training for problem with {problem_instance.num_jobs} jobs, " 
          f"{problem_instance.num_machines} machines, {problem_instance.total_operations} operations")
    print(f"Training with {n_envs} environments for {total_timesteps} timesteps...")
    
    # Train the agent
    agent.train(total_timesteps=total_timesteps)
    
    # Save the model if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    
    return agent, agent.best_solution 