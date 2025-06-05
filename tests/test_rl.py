"""
Tests for the RL implementation of POFJSP.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from algorithms import Operation, ProblemInstance
from rl_env import POFJSPEnv
from rl_agent import POFJSPAgent, train_ppo_agent


class TestRLEnvironment(unittest.TestCase):
    """Test cases for the POFJSPEnv class."""
    
    def setUp(self):
        """Set up a simple test problem."""
        # Create a small test problem instance
        # 2 jobs, 2 machines
        num_operations_per_job = [2, 2]  # J0 has 2 ops, J1 has 2 ops
        processing_times = [
            # Job 0: [[Op0_0_M0, Op0_0_M1], [Op0_1_M0, Op0_1_M1]]
            np.array([[3, 5], [6, np.inf]]),  # J0,O0: M0=3, M1=5; J0,O1: M0=6, M1=cannot
            # Job 1: [[Op1_0_M0, Op1_0_M1], [Op1_1_M0, Op1_1_M1]]
            np.array([[4, 2], [np.inf, 7]])   # J1,O0: M0=4, M1=2; J1,O1: M0=cannot, M1=7
        ]

        predecessors_map = {
            Operation(0, 1): {Operation(0, 0)},  # Op0_1 depends on Op0_0
            Operation(1, 1): {Operation(1, 0)},  # Op1_1 depends on Op1_0
        }
        
        # Create successors map (inverse of predecessors_map)
        successors_map = {}
        for succ, preds in predecessors_map.items():
            for pred in preds:
                if pred not in successors_map:
                    successors_map[pred] = set()
                successors_map[pred].add(succ)
        
        self.problem = ProblemInstance(
            num_jobs=2,
            num_machines=2,
            num_operations_per_job=num_operations_per_job,
            processing_times=processing_times,
            predecessors_map=predecessors_map,
            successors_map=successors_map
        )
        
    def test_environment_creation(self):
        """Test that the environment can be created."""
        env = POFJSPEnv(self.problem)
        self.assertIsNotNone(env)
        
    def test_reset(self):
        """Test that reset returns an observation."""
        env = POFJSPEnv(self.problem)
        obs = env.reset()
        self.assertIsNotNone(obs)
        self.assertEqual(obs.shape, env.observation_space.shape)
        
    def test_step(self):
        """Test that step works with a valid action."""
        env = POFJSPEnv(self.problem)
        env.reset()
        
        # Get available operations
        available_ops = env._get_available_operations()
        self.assertGreater(len(available_ops), 0)
        
        # Create a valid action
        op = available_ops[0]
        op_idx = env.op_to_idx[op]
        
        # Find a valid machine for this operation
        job_idx, op_idx_in_job = op.job_idx, op.op_idx_in_job
        for m in range(env.num_machines):
            if not np.isinf(env.problem.processing_times[job_idx][op_idx_in_job, m]):
                machine_idx = m
                break
        
        action = {"operation_idx": op_idx, "machine_idx": machine_idx}
        
        # Take a step
        obs, reward, done, info = env.step(action)
        
        # Check that the step worked
        self.assertIsNotNone(obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
    def test_invalid_action(self):
        """Test that invalid actions are handled correctly."""
        env = POFJSPEnv(self.problem)
        env.reset()
        
        # Create an invalid action (operation not available)
        action = {"operation_idx": 100, "machine_idx": 0}  # Invalid operation index
        
        # Take a step
        obs, reward, done, info = env.step(action)
        
        # Check that the invalid action is handled
        self.assertEqual(reward, -100)  # Penalty for invalid action
        self.assertTrue(info.get("invalid_action", False))
        
    def test_complete_episode(self):
        """Test that a complete episode can be run."""
        env = POFJSPEnv(self.problem)
        env.reset()
        
        # Run until done
        done = False
        steps = 0
        while not done and steps < 10:  # Limit to 10 steps to avoid infinite loops
            # Get available operations
            available_ops = env._get_available_operations()
            if not available_ops:
                break
                
            # Create a valid action
            op = available_ops[0]
            op_idx = env.op_to_idx[op]
            
            # Find a valid machine for this operation
            job_idx, op_idx_in_job = op.job_idx, op.op_idx_in_job
            for m in range(env.num_machines):
                if not np.isinf(env.problem.processing_times[job_idx][op_idx_in_job, m]):
                    machine_idx = m
                    break
            
            action = {"operation_idx": op_idx, "machine_idx": machine_idx}
            
            # Take a step
            obs, reward, done, info = env.step(action)
            steps += 1
        
        # Either all operations are scheduled or we hit the step limit
        self.assertTrue(done or steps == 10)
        
        # If done, we should have scheduled all operations
        if done:
            self.assertEqual(len(env.scheduled_operations), env.num_operations)
            
    def test_get_solution(self):
        """Test that a solution can be extracted from the environment."""
        env = POFJSPEnv(self.problem)
        env.reset()
        
        # Schedule all operations
        available_ops = env._get_available_operations()
        while available_ops:
            op = available_ops[0]
            op_idx = env.op_to_idx[op]
            
            # Find a valid machine for this operation
            job_idx, op_idx_in_job = op.job_idx, op.op_idx_in_job
            for m in range(env.num_machines):
                if not np.isinf(env.problem.processing_times[job_idx][op_idx_in_job, m]):
                    machine_idx = m
                    break
            
            action = {"operation_idx": op_idx, "machine_idx": machine_idx}
            
            # Take a step
            obs, reward, done, info = env.step(action)
            
            # Get available operations for next iteration
            available_ops = env._get_available_operations()
        
        # Get the solution
        solution = env.get_solution()
        
        # Check that the solution is valid
        self.assertIsNotNone(solution)
        self.assertEqual(len(solution.operation_sequence), env.num_operations)
        self.assertEqual(len(solution.machine_assignment), env.num_operations)
        self.assertGreater(solution.makespan, 0)


if __name__ == "__main__":
    unittest.main() 