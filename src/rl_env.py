"""
Reinforcement Learning Environment for Partial Order FJSP

This module implements a Gym environment for solving the POFJSP using RL.
"""

import gym
import numpy as np
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import copy

from algorithms import Operation, ProblemInstance, Solution


class POFJSPEnv(gym.Env):
    """
    POFJSP environment for reinforcement learning.
    
    This environment follows the OpenAI Gym interface and provides:
    - State representation capturing the current schedule state, available operations, and machine states
    - Action space for operation selection and machine assignment
    - Reward function to minimize makespan
    """
    
    def __init__(self, problem_instance: ProblemInstance, time_limit: int = 1000):
        """
        Initialize POFJSP environment.
        
        Args:
            problem_instance: The POFJSP instance to solve
            time_limit: Maximum number of steps before termination
        """
        super(POFJSPEnv, self).__init__()
        
        self.problem = problem_instance
        self.time_limit = time_limit
        
        # Environment state
        self.step_count = 0
        self.done = False
        self.current_time = 0
        self.scheduled_operations = set()
        self.machine_ready_times = np.zeros(self.problem.num_machines)
        self.job_ready_times = np.zeros(self.problem.num_jobs)
        self.schedule_details = {}
        
        # Extract problem dimensions
        self.num_jobs = self.problem.num_jobs
        self.num_machines = self.problem.num_machines
        self.num_operations = self.problem.total_operations
        
        # Define action and observation spaces
        self.action_space = spaces.Dict({
            "operation_idx": spaces.Discrete(self.num_operations),
            "machine_idx": spaces.Discrete(self.num_machines)
        })
        
        # Observation space includes:
        # - Machine ready times (num_machines)
        # - Job ready times (num_jobs)
        # - Operation status (scheduled/available/locked) (num_operations)
        # - Processing times for each operation on each machine (num_operations x num_machines)
        
        # For simplicity, we use a flattened observation space
        obs_dim = (
            self.num_machines +             # Machine ready times
            self.num_jobs +                 # Job ready times
            self.num_operations +           # Operation status
            self.num_operations * self.num_machines  # Processing times
        )
        
        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize mappings
        self._initialize_operation_mappings()
        
        # Initialize random number generator
        self.np_random = None
        self.seed()
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for this environment's random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def _initialize_operation_mappings(self):
        """Initialize mappings between operations and indices."""
        # Map operation objects to indices and vice versa
        self.op_to_idx = {}
        self.idx_to_op = {}
        
        idx = 0
        for job_idx in range(self.num_jobs):
            for op_idx in range(self.problem.num_operations_per_job[job_idx]):
                op = Operation(job_idx, op_idx)
                self.op_to_idx[op] = idx
                self.idx_to_op[idx] = op
                idx += 1
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.step_count = 0
        self.done = False
        self.current_time = 0
        self.scheduled_operations = set()
        self.machine_ready_times = np.zeros(self.problem.num_machines)
        self.job_ready_times = np.zeros(self.problem.num_jobs)
        self.schedule_details = {}
        
        return self._get_observation()
    
    def _get_available_operations(self) -> List[Operation]:
        """Get operations available for scheduling."""
        available_ops = []
        
        for job_idx in range(self.num_jobs):
            for op_idx in range(self.problem.num_operations_per_job[job_idx]):
                op = Operation(job_idx, op_idx)
                
                # Skip already scheduled operations
                if op in self.scheduled_operations:
                    continue
                
                # Check if all predecessors are scheduled
                predecessors = self.problem.predecessors_map.get(op, set())
                if all(pred in self.scheduled_operations for pred in predecessors):
                    available_ops.append(op)
        
        return available_ops
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Get available operations
        available_ops = self._get_available_operations()
        
        # Machine ready times
        machine_features = self.machine_ready_times.copy()
        
        # Job ready times
        job_features = self.job_ready_times.copy()
        
        # Operation status (0: scheduled, 1: available, -1: locked)
        op_status = np.zeros(self.num_operations)
        for i in range(self.num_operations):
            op = self.idx_to_op[i]
            if op in self.scheduled_operations:
                op_status[i] = 0
            elif op in available_ops:
                op_status[i] = 1
            else:
                op_status[i] = -1
        
        # Processing times
        proc_times = np.zeros((self.num_operations, self.num_machines))
        for i in range(self.num_operations):
            op = self.idx_to_op[i]
            job_idx, op_idx = op.job_idx, op.op_idx_in_job
            for m in range(self.num_machines):
                proc_time = self.problem.processing_times[job_idx][op_idx, m]
                # Replace infinity with a large value
                if np.isinf(proc_time):
                    proc_times[i, m] = 1000  # Large value to represent infeasible
                else:
                    proc_times[i, m] = proc_time
        
        # Flatten and concatenate all features
        proc_times_flat = proc_times.flatten()
        observation = np.concatenate([
            machine_features, 
            job_features, 
            op_status, 
            proc_times_flat
        ])
        
        return observation.astype(np.float32)
    
    def _is_valid_action(self, action_dict: Dict) -> bool:
        """Check if an action is valid."""
        op_idx = action_dict["operation_idx"]
        machine_idx = action_dict["machine_idx"]
        
        # Check if operation index is valid
        if op_idx >= self.num_operations:
            return False
        
        op = self.idx_to_op[op_idx]
        
        # Check if operation is available
        available_ops = self._get_available_operations()
        if op not in available_ops:
            return False
        
        # Check if machine can process this operation
        job_idx, op_idx_in_job = op.job_idx, op.op_idx_in_job
        proc_time = self.problem.processing_times[job_idx][op_idx_in_job, machine_idx]
        if np.isinf(proc_time):
            return False
        
        return True
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a scheduling action.
        
        Args:
            action: Dictionary with operation_idx and machine_idx
            
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1
        
        # Handle case when the action is invalid
        if not self._is_valid_action(action):
            return self._get_observation(), -100, False, {"invalid_action": True}
        
        # Extract action components
        op_idx = action["operation_idx"]
        machine_idx = action["machine_idx"]
        operation = self.idx_to_op[op_idx]
        
        # Get processing time
        job_idx, op_idx_in_job = operation.job_idx, operation.op_idx_in_job
        proc_time = self.problem.processing_times[job_idx][op_idx_in_job, machine_idx]
        
        # Determine start time (max of machine ready time and job ready time)
        start_time = max(
            self.machine_ready_times[machine_idx],
            self.job_ready_times[job_idx]
        )
        
        # Update state
        end_time = start_time + proc_time
        self.machine_ready_times[machine_idx] = end_time
        self.job_ready_times[job_idx] = end_time
        self.scheduled_operations.add(operation)
        
        # Record schedule details
        self.schedule_details[operation] = {
            "start_time": start_time,
            "end_time": end_time,
            "machine": machine_idx
        }
        
        # Update current makespan
        current_makespan = max(self.machine_ready_times)
        
        # Calculate reward (negative makespan improvement)
        reward = -proc_time  # Simple reward based on processing time
        
        # Check termination conditions
        if len(self.scheduled_operations) == self.num_operations:
            self.done = True
            # Additional reward for completion
            reward += 50
        elif self.step_count >= self.time_limit:
            self.done = True
            # Penalty for not completing
            reward -= 50
        
        # Prepare info dictionary
        info = {
            "makespan": current_makespan,
            "scheduled_ops": len(self.scheduled_operations),
            "total_ops": self.num_operations,
        }
        
        return self._get_observation(), reward, self.done, info
    
    def get_solution(self) -> Solution:
        """Convert current schedule to a Solution object."""
        # Extract operation sequence and machine assignments
        ops_sequence = []
        machine_assignments = []
        
        # Sort operations by start time
        sorted_ops = sorted(
            self.schedule_details.items(), 
            key=lambda x: x[1]["start_time"]
        )
        
        for op, details in sorted_ops:
            ops_sequence.append(op)
            machine_assignments.append(details["machine"])
        
        solution = Solution(ops_sequence, machine_assignments)
        solution.makespan = max(self.machine_ready_times)
        solution.schedule_details = self.schedule_details
        
        return solution 