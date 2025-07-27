"""
Tensor Cache for Efficient Memory Management

Provides reusable tensor buffers to avoid repeated allocations
during RL training, significantly improving memory efficiency.
"""

import torch
from typing import Dict, Tuple, Optional
import threading
from contextlib import contextmanager

from src.exceptions import MemoryError as POFJSPMemoryError, handle_gpu_memory_error


class TensorCache:
    """
    Cache for frequently used tensors to avoid repeated allocation.
    
    Thread-safe tensor buffer management with automatic cleanup
    and memory monitoring capabilities.
    """
    
    def __init__(self, max_jobs: int, max_machines: int, device: torch.device):
        """
        Initialize tensor cache.
        
        Args:
            max_jobs: Maximum number of jobs
            max_machines: Maximum number of machines  
            device: Target device for tensors
        """
        self.max_jobs = max_jobs
        self.max_machines = max_machines
        self.device = device
        self._lock = threading.RLock()
        
        # Pre-allocate reusable buffers
        self._initialize_buffers()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.allocations = 0
    
    @handle_gpu_memory_error
    def _initialize_buffers(self) -> None:
        """Initialize reusable tensor buffers."""
        with self._lock:
            # Job-related buffers
            self.job_mask_buffer = torch.zeros(
                self.max_jobs, dtype=torch.bool, device=self.device
            )
            
            # Machine-related buffers
            self.machine_mask_buffer = torch.zeros(
                self.max_machines, dtype=torch.bool, device=self.device
            )
            
            # Processing times buffer
            self.processing_times_buffer = torch.zeros(
                self.max_jobs, self.max_machines, dtype=torch.float32, device=self.device
            )
            
            # Additional buffers for common operations
            self.temp_job_buffer = torch.zeros(
                self.max_jobs, dtype=torch.float32, device=self.device
            )
            self.temp_machine_buffer = torch.zeros(
                self.max_machines, dtype=torch.float32, device=self.device
            )
            
            self.allocations += 5
    
    @handle_gpu_memory_error
    def pad_observation(self, obs: Dict, copy_tensors: bool = True) -> Dict:
        """
        Pad observation using cached buffers.
        
        Args:
            obs: Input observation dictionary
            copy_tensors: Whether to return copies (safer) or views
            
        Returns:
            Padded observation dictionary
        """
        with self._lock:
            try:
                # Clear buffers
                self.job_mask_buffer.fill_(False)
                self.machine_mask_buffer.fill_(False)
                self.processing_times_buffer.fill_(0.0)
                
                # Pad job mask
                if 'job_mask' in obs:
                    actual_jobs = min(obs['job_mask'].size(0), self.max_jobs)
                    self.job_mask_buffer[:actual_jobs] = obs['job_mask'][:actual_jobs].to(
                        self.device, non_blocking=True
                    )
                
                # Pad machine mask
                if 'machine_mask' in obs:
                    actual_machines = min(obs['machine_mask'].size(0), self.max_machines)
                    self.machine_mask_buffer[:actual_machines] = obs['machine_mask'][:actual_machines].to(
                        self.device, non_blocking=True
                    )
                
                # Pad processing times
                if 'processing_times' in obs:
                    pt_tensor = obs['processing_times']
                    actual_jobs_pt = min(pt_tensor.size(0), self.max_jobs)
                    actual_machines_pt = min(pt_tensor.size(1), self.max_machines)
                    self.processing_times_buffer[:actual_jobs_pt, :actual_machines_pt] = pt_tensor[
                        :actual_jobs_pt, :actual_machines_pt
                    ].to(self.device, non_blocking=True)
                
                # Return padded observation
                result = {
                    'x': obs['x'].to(self.device, non_blocking=True),
                    'edge_index': obs['edge_index'].to(self.device, non_blocking=True),
                    'batch': obs['batch'].to(self.device, non_blocking=True),
                    'job_mask': self.job_mask_buffer.clone() if copy_tensors else self.job_mask_buffer,
                    'machine_mask': self.machine_mask_buffer.clone() if copy_tensors else self.machine_mask_buffer,
                    'processing_times': self.processing_times_buffer.clone() if copy_tensors else self.processing_times_buffer
                }
                
                self.cache_hits += 1
                return result
                
            except RuntimeError as e:
                self.cache_misses += 1
                # Fallback to direct allocation
                return self._fallback_pad_observation(obs)
    
    def _fallback_pad_observation(self, obs: Dict) -> Dict:
        """Fallback method without caching."""
        job_mask_padded = torch.zeros(self.max_jobs, dtype=torch.bool, device=self.device)
        machine_mask_padded = torch.zeros(self.max_machines, dtype=torch.bool, device=self.device)
        processing_times_padded = torch.zeros(
            self.max_jobs, self.max_machines, dtype=torch.float32, device=self.device
        )
        
        if 'job_mask' in obs:
            actual_jobs = min(obs['job_mask'].size(0), self.max_jobs)
            job_mask_padded[:actual_jobs] = obs['job_mask'][:actual_jobs].to(self.device)
        
        if 'machine_mask' in obs:
            actual_machines = min(obs['machine_mask'].size(0), self.max_machines)
            machine_mask_padded[:actual_machines] = obs['machine_mask'][:actual_machines].to(self.device)
        
        if 'processing_times' in obs:
            pt_tensor = obs['processing_times']
            actual_jobs_pt = min(pt_tensor.size(0), self.max_jobs)
            actual_machines_pt = min(pt_tensor.size(1), self.max_machines)
            processing_times_padded[:actual_jobs_pt, :actual_machines_pt] = pt_tensor[
                :actual_jobs_pt, :actual_machines_pt
            ].to(self.device)
        
        return {
            'x': obs['x'].to(self.device),
            'edge_index': obs['edge_index'].to(self.device),
            'batch': obs['batch'].to(self.device),
            'job_mask': job_mask_padded,
            'machine_mask': machine_mask_padded,
            'processing_times': processing_times_padded
        }
    
    @contextmanager
    def get_temp_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32):
        """
        Context manager for temporary tensors.
        
        Args:
            shape: Required tensor shape
            dtype: Tensor data type
            
        Yields:
            Temporary tensor
        """
        tensor = None
        try:
            with self._lock:
                # Check if we can reuse existing buffers
                if shape == (self.max_jobs,) and dtype == torch.float32:
                    tensor = self.temp_job_buffer
                    tensor.fill_(0.0)
                elif shape == (self.max_machines,) and dtype == torch.float32:
                    tensor = self.temp_machine_buffer
                    tensor.fill_(0.0)
                else:
                    # Allocate new tensor
                    tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                    self.allocations += 1
                
            yield tensor
            
        finally:
            # Cleanup if we allocated a new tensor
            if tensor is not None and tensor not in [self.temp_job_buffer, self.temp_machine_buffer]:
                del tensor
    
    def clear_cache(self) -> None:
        """Clear all cached tensors and free memory."""
        with self._lock:
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Reset statistics
            self.cache_hits = 0
            self.cache_misses = 0
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_ratio': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'total_allocations': self.allocations
        }
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            stats.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved(self.device) / 1024**3,  # GB
                'gpu_memory_free': (torch.cuda.get_device_properties(self.device).total_memory - 
                                   torch.cuda.memory_reserved(self.device)) / 1024**3  # GB
            })
        
        return stats
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.clear_cache()
        except Exception:
            pass  # Ignore errors during cleanup


class BatchTensorProcessor:
    """
    Efficient batch processing for tensor operations.
    
    Optimizes common tensor operations by processing data in batches
    and reusing memory allocations.
    """
    
    def __init__(self, device: torch.device, batch_size: int = 32):
        """
        Initialize batch processor.
        
        Args:
            device: Target device
            batch_size: Processing batch size
        """
        self.device = device
        self.batch_size = batch_size
        self._temp_tensors = {}
    
    @handle_gpu_memory_error
    def process_experience_batch(self, experiences: list, tensor_cache: TensorCache) -> Dict[str, torch.Tensor]:
        """
        Process experience batch efficiently.
        
        Args:
            experiences: List of experience dictionaries
            tensor_cache: Tensor cache for memory optimization
            
        Returns:
            Batched tensors
        """
        if not experiences:
            return {}
        
        batch_size = len(experiences)
        
        # Pre-allocate output tensors
        with tensor_cache.get_temp_tensor((batch_size,), torch.long) as job_actions:
            job_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            with tensor_cache.get_temp_tensor((batch_size,), torch.long) as machine_actions:
                machine_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                
                with tensor_cache.get_temp_tensor((batch_size,), torch.float32) as rewards:
                    rewards = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                    
                    # Fill tensors efficiently
                    for i, exp in enumerate(experiences):
                        job_actions[i] = exp.get('job_action', 0)
                        machine_actions[i] = exp.get('machine_action', 0)
                        rewards[i] = exp.get('reward', 0.0)
                    
                    return {
                        'job_actions': job_actions.clone(),
                        'machine_actions': machine_actions.clone(),
                        'rewards': rewards.clone()
                    }
    
    def __del__(self):
        """Cleanup temporary tensors."""
        for tensor in self._temp_tensors.values():
            del tensor
        self._temp_tensors.clear()