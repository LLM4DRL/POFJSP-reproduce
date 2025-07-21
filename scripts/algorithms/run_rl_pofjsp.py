#!/usr/bin/env python3
"""
Standalone RL Runner for POFJSP with GNN+PPO

This script provides a command-line interface to train and evaluate
GNN+PPO agents for Partially Ordered Flexible Job Shop Scheduling.
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.problems.problem_instance import ProblemInstance
from src.rl.environments.pofjsp_env import POFJSPEnv
from src.rl.models.ppo_agent import PPOAgent
from src.algorithms.decoder import decode_solution


def load_problem(problem_path: str) -> ProblemInstance:
    """Load problem instance from file."""
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file not found: {problem_path}")
    
    # Load based on file extension
    if problem_path.endswith('.json'):
        return ProblemInstance.from_json(problem_path)
    elif problem_path.endswith('.txt'):
        return ProblemInstance.from_fjsp_file(problem_path)
    else:
        raise ValueError(f"Unsupported file format: {problem_path}")


def train_agent(
    problem: ProblemInstance,
    output_dir: str,
    total_timesteps: int = 100000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    hidden_dim: int = 128,
    num_layers: int = 3,
    device: str = "auto",
    save_interval: int = 10000,
    eval_interval: int = 5000,
    verbose: bool = True
) -> PPOAgent:
    """Train PPO agent for POFJSP."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize environment
    env = POFJSPEnv(problem, time_limit=problem.total_operations * 2)
    
    # Initialize agent
    agent = PPOAgent(
        input_dim=8,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_jobs=problem.num_jobs,
        num_machines=problem.num_machines,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    best_makespan = float('inf')
    best_solution = None
    
    # Training loop
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    
    if verbose:
        print(f"Starting training for POFJSP with:")
        print(f"  Jobs: {problem.num_jobs}")
        print(f"  Machines: {problem.num_machines}")
        print(f"  Total operations: {problem.total_operations}")
        print(f"  Device: {device}")
        print(f"  Total timesteps: {total_timesteps}")
        print("-" * 50)
    
    start_time = time.time()
    
    for step in range(total_timesteps):
        # Get action - ensure all tensors are on the same device as the agent
        device = agent.device
        job_action, machine_action, job_log_prob, machine_log_prob, value = agent.get_action(
            obs['x'].to(device),
            obs['edge_index'].to(device),
            obs['batch'].to(device),
            obs['job_mask'].to(device),
            obs['machine_mask'].to(device),
            obs['processing_times'].to(device)
        )
        
        # Execute action
        action = [job_action.item(), machine_action.item()]
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store experience
        agent.buffer.add({
            'state': (obs['x'], obs['edge_index'], obs['batch']),
            'job_action': job_action.item(),
            'machine_action': machine_action.item(),
            'job_log_prob': job_log_prob.item(),
            'machine_log_prob': machine_log_prob.item(),
            'reward': reward,
            'value': value.item(),
            'done': terminated or truncated,
            'job_mask': obs['job_mask'],
            'machine_mask': obs['machine_mask']
        })
        
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
        # Update agent
        if len(agent.buffer) >= agent.batch_size:
            agent.update()
        
        # Log episode
        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Get final solution
            solution = env.get_solution()
            makespan = solution.makespan
            
            # Update best solution
            if makespan < best_makespan:
                best_makespan = makespan
                best_solution = solution
                
                if verbose:
                    print(f"New best makespan: {best_makespan:.2f} at step {step}")
            
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
        
        # Periodic evaluation
        if step % eval_interval == 0 and step > 0:
            eval_makespan = evaluate_agent(agent, problem, num_episodes=5)
            if verbose:
                print(f"Step {step}: Average makespan = {eval_makespan:.2f}")
        
        # Save checkpoint
        if step % save_interval == 0 and step > 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pt")
            agent.save(checkpoint_path)
    
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    agent.save(final_model_path)
    
    # Save training summary
    summary = {
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'best_makespan': best_makespan,
        'avg_episode_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 0
    }
    
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best makespan: {best_makespan:.2f}")
        print(f"Model saved to: {final_model_path}")
    
    return agent, best_solution


def evaluate_agent(
    agent: PPOAgent,
    problem: ProblemInstance,
    num_episodes: int = 10,
    deterministic: bool = True
) -> float:
    """Evaluate trained agent."""
    
    env = POFJSPEnv(problem, time_limit=problem.total_operations * 2)
    
    makespans = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        while True:
            job_action, machine_action, _, _, _ = agent.get_action(
                obs['x'].to(agent.device),
                obs['edge_index'].to(agent.device),
                obs['batch'].to(agent.device),
                obs['job_mask'].to(agent.device),
                obs['machine_mask'].to(agent.device),
                obs['processing_times'].to(agent.device),
                deterministic=deterministic
            )
            
            action = [job_action.item(), machine_action.item()]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        solution = env.get_solution()
        makespans.append(solution.makespan)
    
    return np.mean(makespans)


def solve_problem(
    problem_path: str,
    model_path: str,
    output_path: str,
    deterministic: bool = True
) -> None:
    """Solve a problem instance using trained model."""
    
    # Load problem and model
    problem = load_problem(problem_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        input_dim=8,
        hidden_dim=128,
        num_layers=3,
        num_jobs=problem.num_jobs,
        num_machines=problem.num_machines,
        device=device
    )
    agent.load(model_path)
    
    # Solve problem
    env = POFJSPEnv(problem)
    obs, info = env.reset()
    
    while True:
        job_action, machine_action, _, _, _ = agent.get_action(
            obs['x'].to(agent.device),
            obs['edge_index'].to(agent.device),
            obs['batch'].to(agent.device),
            obs['job_mask'].to(agent.device),
            obs['machine_mask'].to(agent.device),
            obs['processing_times'].to(agent.device),
            deterministic=deterministic
        )
        
        action = [job_action.item(), machine_action.item()]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Get and save solution
    solution = env.get_solution()
    
    # Save solution
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    solution.save(output_path)
    
    print(f"Solution found with makespan: {solution.makespan:.2f}")
    print(f"Solution saved to: {output_path}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Train PPO agent for POFJSP")
    
    # Required arguments
    parser.add_argument("problem", type=str, help="Path to problem file (.json or .txt)")
    
    # Optional arguments
    parser.add_argument("--mode", type=str, choices=['train', 'solve'], default='train',
                        help="Mode: train or solve")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for results")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for networks")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model (for solve mode)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--save-interval", type=int, default=10000,
                        help="Model save interval")
    parser.add_argument("--eval-interval", type=int, default=5000,
                        help="Evaluation interval")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load problem
    problem = load_problem(args.problem)
    
    if args.mode == 'train':
        # Train agent
        agent, best_solution = train_agent(
            problem=problem,
            output_dir=args.output_dir,
            total_timesteps=args.total_timesteps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            device=args.device,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            verbose=args.verbose
        )
        
        # Save best solution
        if best_solution:
            best_solution.save(os.path.join(args.output_dir, "best_solution.json"))
    
    elif args.mode == 'solve':
        if not args.model_path:
            raise ValueError("--model-path is required for solve mode")
        
        # Solve problem
        output_path = os.path.join(args.output_dir, "solution.json")
        solve_problem(
            problem_path=args.problem,
            model_path=args.model_path,
            output_path=output_path,
            deterministic=True
        )
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()