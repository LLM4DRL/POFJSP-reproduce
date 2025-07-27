"""
POFJSP Environment for Reinforcement Learning

Gym-compatible environment for Partially Ordered Flexible Job Shop Scheduling
with graph-based state representation and hierarchical action space.
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass
import threading

from src.problems.problem_instance import ProblemInstance, Solution
from src.rl.models.graph_cnn import GraphCNN
from src.exceptions import RLTrainingError


class POFJSPEnv(gym.Env):
    """
    RL Environment for POFJSP with precedence constraints.
    
    State: Graph representation of the scheduling problem
    Action: Hierarchical (job selection, machine selection)
    Reward: Negative makespan (minimization objective)
    """
    
    def __init__(
        self,
        problem: ProblemInstance,
        time_limit: int = 1000,
        reward_type: str = "makespan",
        normalize_reward: bool = True
    ):
        """
        Initialize POFJSP environment.
        
        Args:
            problem: POFJSP problem instance
            time_limit: Maximum steps per episode
            reward_type: Type of reward function
            normalize_reward: Whether to normalize rewards
        """
        super(POFJSPEnv, self).__init__()
        
        self.problem = problem
        self.time_limit = time_limit
        self.reward_type = reward_type
        self.normalize_reward = normalize_reward
        
        # State dimensions
        self.num_jobs = problem.num_jobs
        self.num_machines = problem.num_machines
        self.num_operations = problem.total_operations
        
        # Action space: Hierarchical (job, machine)
        self.action_space = gym.spaces.MultiDiscrete([self.num_jobs, self.num_machines])
        
        # Observation space: Graph-based state
        # Node features for each operation
        self.node_feature_dim = 8  # [proc_time_min, proc_time_max, proc_time_avg, status, job_ready, machine_ready, job_id_norm, op_pos_norm]
        
        # Initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment.
        
        Returns:
            observation: Initial state as graph
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset scheduling state
        self.current_time = 0.0
        self.machine_ready_times = np.zeros(self.num_machines)
        self.job_ready_times = np.zeros(self.num_jobs)
        self.scheduled_operations = set()
        
        # Track operation status
        self.operation_start_times = {}
        self.operation_completion_times = {}
        self.operation_machine_assignments = {}
        
        # Build precedence graph
        self._build_precedence_graph()
        
        # Create initial state
        state = self._get_state()
        
        # Info dictionary
        info = {
            'current_time': self.current_time,
            'scheduled_ops': len(self.scheduled_operations),
            'total_ops': self.num_operations,
            'makespan': 0.0
        }
        
        return state, info
    
    def _build_precedence_graph(self) -> None:
        """Build precedence graph for operations."""
        self.G = nx.DiGraph()
        
        # Add nodes for all operations
        for job_idx in range(self.num_jobs):
            for op_idx in range(self.problem.num_operations_per_job[job_idx]):
                op = (job_idx, op_idx)
                self.G.add_node(op)
                
        # Add edges for precedence constraints
        for job_idx in range(self.num_jobs):
            for op_idx in range(self.problem.num_operations_per_job[job_idx]):
                op = (job_idx, op_idx)
                predecessors = self.problem.predecessors_map.get(op, set())
                for pred in predecessors:
                    self.G.add_edge(pred, op)
    
    def _get_state(self) -> Dict[str, Any]:
        """
        Get current state as graph representation.
        
        Returns:
            Dictionary containing graph data
        """
        # Create graph features using GraphCNN utility
        gnn = GraphCNN()
        x, edge_index = gnn.create_graph_features(
            self.problem,
            current_time=self.current_time,
            machine_ready_times=self.machine_ready_times,
            job_ready_times=self.job_ready_times,
            scheduled_ops=self.scheduled_operations
        )
        
        # Get valid actions
        valid_jobs, valid_machines = self._get_valid_actions()
        
        # Create masks
        job_mask = torch.zeros(self.num_jobs, dtype=torch.bool)
        job_mask[valid_jobs] = True
        
        machine_mask = torch.zeros(self.num_machines, dtype=torch.bool)
        machine_mask[valid_machines] = True
        
        # Processing times for current valid jobs
        processing_times = torch.zeros(self.num_jobs, self.num_machines, dtype=torch.float32)
        for job_idx in valid_jobs:
            for op_idx in range(self.problem.num_operations_per_job[job_idx]):
                op = (job_idx, op_idx)
                if op not in self.scheduled_operations and self._is_operation_ready(op):
                    proc_times = self.problem.processing_times[job_idx][op_idx, :]
                    valid_machines_mask = ~np.isinf(proc_times)
                    if np.any(valid_machines_mask):
                        processing_times[job_idx, valid_machines_mask] = torch.tensor(
                            proc_times[valid_machines_mask], dtype=torch.float32
                        )
                    break
        
        return {
            'x': x,
            'edge_index': edge_index,
            'batch': torch.zeros(x.size(0), dtype=torch.long),  # Single graph
            'job_mask': job_mask,
            'machine_mask': machine_mask,
            'processing_times': processing_times,
            'current_time': self.current_time,
            'machine_ready_times': self.machine_ready_times,
            'job_ready_times': self.job_ready_times
        }
    
    def _get_valid_actions(self) -> Tuple[List[int], List[int]]:
        """
        Get valid job-machine pairs for scheduling.
        
        Returns:
            valid_jobs: List of valid job indices
            valid_machines: List of valid machine indices
        """
        valid_jobs = []
        valid_machines = []
        
        # Find ready operations
        ready_ops = []
        for job_idx in range(self.num_jobs):
            for op_idx in range(self.problem.num_operations_per_job[job_idx]):
                op = (job_idx, op_idx)
                if op not in self.scheduled_operations and self._is_operation_ready(op):
                    ready_ops.append(op)
                    break  # Only first ready operation per job
        
        # Get valid job-machine pairs
        for job_idx, op_idx in ready_ops:
            valid_jobs.append(job_idx)
            
        # Get all valid machines for any ready operation
        all_valid_machines = set()
        for job_idx, op_idx in ready_ops:
            proc_times = self.problem.processing_times[job_idx][op_idx, :]
            valid_machines_for_op = np.where(~np.isinf(proc_times))[0]
            all_valid_machines.update(valid_machines_for_op)
        
        valid_jobs = list(set(valid_jobs))
        valid_machines = list(all_valid_machines)
        
        # Ensure at least one valid machine exists for each valid job
        if not valid_machines:
            valid_machines = list(range(self.num_machines))  # Fallback
        
        return valid_jobs, valid_machines
    
    def _is_operation_ready(self, op: Tuple[int, int]) -> bool:
        """Check if operation is ready to be scheduled."""
        job_idx, op_idx = op
        
        # Check if all predecessors are scheduled
        predecessors = self.problem.predecessors_map.get(op, set())
        if not all(pred in self.scheduled_operations for pred in predecessors):
            return False
            
        # Check if job is ready
        if self.job_ready_times[job_idx] > self.current_time:
            return False
            
        return True
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment.
        
        Args:
            action: [job_idx, machine_idx]
            
        Returns:
            observation: New state
            reward: Reward for action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Validate and convert action to integers
        try:
            job_idx, machine_idx = action
            job_idx = int(job_idx)
            machine_idx = int(machine_idx)
        except (ValueError, TypeError):
            # Invalid action format
            reward = -10.0
            terminated = False
            truncated = self.current_time >= self.time_limit
            state = self._get_state()
            info = {'invalid_action': True, 'error': 'Invalid action format', 'makespan': self._get_makespan()}
            return state, reward, terminated, truncated, info
        
        # Find the operation to schedule
        op_to_schedule = None
        for op_idx in range(self.problem.num_operations_per_job[job_idx]):
            op = (job_idx, op_idx)
            if op not in self.scheduled_operations and self._is_operation_ready(op):
                op_to_schedule = op
                break
        
        if op_to_schedule is None:
            # Invalid action
            reward = -10.0
            terminated = False
            truncated = self.current_time >= self.time_limit
            state = self._get_state()
            info = {'invalid_action': True, 'makespan': self._get_makespan()}
            return state, reward, terminated, truncated, info
        
        # Check if the machine can process this operation
        processing_time = self.problem.processing_times[job_idx][op_to_schedule[1], machine_idx]
        
        if np.isinf(processing_time) or processing_time <= 0:
            # Invalid machine assignment
            reward = -5.0  # Penalty for invalid machine choice
            terminated = False
            truncated = self.current_time >= self.time_limit
            state = self._get_state()
            info = {
                'current_time': self.current_time,
                'scheduled_ops': len(self.scheduled_operations),
                'total_ops': self.num_operations,
                'makespan': self._get_makespan(),
                'action_valid': False,
                'invalid_machine': True
            }
            return state, reward, terminated, truncated, info
        
        # Schedule the operation with atomic state update
        start_time = max(self.current_time, self.machine_ready_times[machine_idx], self.job_ready_times[job_idx])
        completion_time = start_time + processing_time
        
        # Create new state atomically to prevent race conditions
        new_state = self._create_atomic_state_update(
            op_to_schedule, machine_idx, job_idx, start_time, completion_time
        )
        
        # Apply all state changes atomically
        self._apply_state_update(new_state)
        
        # Check if all operations are scheduled
        all_scheduled = len(self.scheduled_operations) == self.num_operations
        terminated = all_scheduled
        truncated = self.current_time >= self.time_limit
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -1000.0, 1000.0)
        
        # Get new state
        state = self._get_state()
        
        # Info
        info = {
            'current_time': self.current_time,
            'scheduled_ops': len(self.scheduled_operations),
            'total_ops': self.num_operations,
            'makespan': self._get_makespan(),
            'action_valid': True
        }
        
        return state, reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state."""
        if len(self.scheduled_operations) == self.num_operations:
            # All operations scheduled - reward based on makespan
            makespan = self._get_makespan()
            if self.normalize_reward:
                # Normalize by number of operations or max possible makespan
                max_possible_makespan = np.sum(np.max(self.problem.processing_times, axis=2))
                reward = -makespan / max_possible_makespan
            else:
                reward = -makespan
        else:
            # Partial reward for progress
            progress = len(self.scheduled_operations) / self.num_operations
            reward = progress * 0.1  # Small positive reward for progress
            
        return reward
    
    def _get_makespan(self) -> float:
        """Calculate current makespan."""
        if not self.operation_completion_times:
            return 0.0
        return max(self.operation_completion_times.values())
    
    def get_solution(self) -> Solution:
        """
        Get the complete solution.
        
        Returns:
            Solution object with scheduling information
        """
        solution = Solution(self.problem)
        
        # Build operation sequences for each job
        job_operations = defaultdict(list)
        for op, machine_idx in self.operation_machine_assignments.items():
            job_idx, op_idx = op
            start_time = self.operation_start_times[op]
            completion_time = self.operation_completion_times[op]
            processing_time = completion_time - start_time
            
            job_operations[job_idx].append({
                'operation': op_idx,
                'machine': machine_idx,
                'start_time': start_time,
                'processing_time': processing_time,
                'completion_time': completion_time
            })
        
        # Sort operations by start time for each job
        for job_idx in range(self.num_jobs):
            job_operations[job_idx].sort(key=lambda x: x['start_time'])
            for op_info in job_operations[job_idx]:
                solution.add_operation(
                    job_idx, 
                    op_info['operation'],
                    op_info['machine'],
                    op_info['start_time']
                )
        
        return solution
    
    @dataclass
    class StateUpdate:
        """Atomic state update container."""
        scheduled_operations: set
        operation_start_times: dict
        operation_completion_times: dict
        operation_machine_assignments: dict
        machine_ready_times: np.ndarray
        job_ready_times: np.ndarray
        current_time: float
    
    def _create_atomic_state_update(self, 
                                   operation: Tuple[int, int], 
                                   machine_idx: int, 
                                   job_idx: int,
                                   start_time: float, 
                                   completion_time: float) -> 'StateUpdate':
        """Create atomic state update to prevent race conditions."""
        # Create copies of current state
        new_scheduled_ops = self.scheduled_operations.copy()
        new_start_times = self.operation_start_times.copy()
        new_completion_times = self.operation_completion_times.copy()
        new_machine_assignments = self.operation_machine_assignments.copy()
        new_machine_ready_times = self.machine_ready_times.copy()
        new_job_ready_times = self.job_ready_times.copy()
        
        # Apply updates to copies
        new_scheduled_ops.add(operation)
        new_start_times[operation] = start_time
        new_completion_times[operation] = completion_time
        new_machine_assignments[operation] = machine_idx
        new_machine_ready_times[machine_idx] = completion_time
        new_job_ready_times[job_idx] = completion_time
        new_current_time = max(self.current_time, completion_time)
        
        return self.StateUpdate(
            scheduled_operations=new_scheduled_ops,
            operation_start_times=new_start_times,
            operation_completion_times=new_completion_times,
            operation_machine_assignments=new_machine_assignments,
            machine_ready_times=new_machine_ready_times,
            job_ready_times=new_job_ready_times,
            current_time=new_current_time
        )
    
    def _apply_state_update(self, state_update: 'StateUpdate') -> None:
        """Apply atomic state update."""
        try:
            # Apply all updates atomically
            self.scheduled_operations = state_update.scheduled_operations
            self.operation_start_times = state_update.operation_start_times
            self.operation_completion_times = state_update.operation_completion_times
            self.operation_machine_assignments = state_update.operation_machine_assignments
            self.machine_ready_times = state_update.machine_ready_times
            self.job_ready_times = state_update.job_ready_times
            self.current_time = state_update.current_time
        except Exception as e:
            raise RLTrainingError("state_update", f"Failed to apply atomic state update: {e}")
    
    def render(self, mode: str = 'human') -> None:
        """Render current state."""
        print(f"Current time: {self.current_time:.2f}")
        print(f"Scheduled operations: {len(self.scheduled_operations)}/{self.num_operations}")
        print(f"Current makespan: {self._get_makespan():.2f}")
        
        # Show machine status
        for machine_idx in range(self.num_machines):
            status = "Available" if self.machine_ready_times[machine_idx] <= self.current_time else "Busy"
            print(f"Machine {machine_idx}: {status} (ready at {self.machine_ready_times[machine_idx]:.2f})")
        
        print("-" * 50)
    
    def _get_available_operations(self) -> List[Tuple[int, int]]:
        """Get list of operations that can be scheduled next."""
        available_ops = []
        
        for job_idx in range(self.num_jobs):
            for op_idx in range(self.problem.num_operations_per_job[job_idx]):
                op = (job_idx, op_idx)
                if op not in self.scheduled_operations and self._is_operation_ready(op):
                    available_ops.append(op)
        
        return available_ops