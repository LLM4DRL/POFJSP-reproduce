"""
Training Configuration Management

Centralized configuration for RL training with validation and presets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path

from validation import validate_inputs, Validators


@dataclass
class TrainingConfig:
    """Comprehensive training configuration with validation."""
    
    # Problem scaling
    min_jobs: int = 8
    min_machines: int = 6
    max_jobs: int = 100
    max_machines: int = 100
    
    # Training parameters
    total_timesteps: int = 1_000_000
    learning_rate: float = 1e-4
    batch_size: int = 512
    n_steps: int = 2048
    n_epochs: int = 8
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
    log_every: int = 1000
    
    # Performance
    timeout_per_instance: float = 30.0
    max_parallel_envs: int = 1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.min_jobs <= 0 or self.min_machines <= 0:
            raise ValueError("min_jobs and min_machines must be positive")
        
        if self.max_jobs < self.min_jobs or self.max_machines < self.min_machines:
            raise ValueError("max values must be >= min values")
        
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError("learning_rate must be in (0, 1]")
        
        if self.batch_size <= 0 or self.n_steps <= 0:
            raise ValueError("batch_size and n_steps must be positive")
        
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        
        if not 0 < self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in (0, 1]")
        
        if not 0 < self.clip_range <= 1:
            raise ValueError("clip_range must be in (0, 1]")
    
    @classmethod
    def from_json(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_json(self, config_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def create_fast_mode(self) -> 'TrainingConfig':
        """Create a fast mode configuration for testing."""
        config = TrainingConfig(
            total_timesteps=100_000,
            stage_timesteps=30_000,
            eval_every=10_000,
            log_every=2000,
            save_every=20_000,
            curriculum_stages=2,
            max_jobs=50,
            max_machines=30,
            batch_size=256,
            n_steps=1024
        )
        return config
    
    def create_debug_mode(self) -> 'TrainingConfig':
        """Create a debug mode configuration for development."""
        config = TrainingConfig(
            total_timesteps=10_000,
            stage_timesteps=3_000,
            eval_every=2_000,
            log_every=100,
            save_every=5_000,
            curriculum_stages=2,
            max_jobs=20,
            max_machines=15,
            batch_size=64,
            n_steps=256,
            timeout_per_instance=10.0
        )
        return config
    
    def create_production_mode(self) -> 'TrainingConfig':
        """Create a production mode configuration for full training."""
        config = TrainingConfig(
            total_timesteps=5_000_000,
            stage_timesteps=1_500_000,
            eval_every=100_000,
            log_every=5000,
            save_every=200_000,
            curriculum_stages=4,
            max_jobs=200,
            max_machines=150,
            batch_size=1024,
            n_steps=4096,
            hidden_dim=512,
            num_layers=4
        )
        return config


@dataclass
class CurriculumStageConfig:
    """Configuration for a single curriculum stage."""
    name: str
    job_range: Tuple[int, int]
    machine_range: Tuple[int, int]
    instances_per_eval: int
    target_success_rate: float
    max_episode_steps: int
    complexity: str = "medium"  # simple, medium, complex


class ConfigurationManager:
    """Manages multiple training configurations and presets."""
    
    def __init__(self):
        self.presets = {
            'debug': self._create_debug_preset,
            'fast': self._create_fast_preset,
            'standard': self._create_standard_preset,
            'production': self._create_production_preset
        }
    
    def get_config(self, preset_name: str = 'standard', **overrides) -> TrainingConfig:
        """Get configuration with optional parameter overrides."""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.presets.keys())}")
        
        config = self.presets[preset_name]()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        return config
    
    def _create_debug_preset(self) -> TrainingConfig:
        """Debug configuration for development."""
        return TrainingConfig().create_debug_mode()
    
    def _create_fast_preset(self) -> TrainingConfig:
        """Fast configuration for quick testing."""
        return TrainingConfig().create_fast_mode()
    
    def _create_standard_preset(self) -> TrainingConfig:
        """Standard configuration for regular training."""
        return TrainingConfig()
    
    def _create_production_preset(self) -> TrainingConfig:
        """Production configuration for full training runs."""
        return TrainingConfig().create_production_mode()
    
    def create_curriculum_stages(self, config: TrainingConfig) -> List[CurriculumStageConfig]:
        """Create curriculum stages based on training configuration."""
        stages = []
        
        # Calculate stage parameters based on config ranges
        job_increment = (config.max_jobs - config.min_jobs) // config.curriculum_stages
        machine_increment = (config.max_machines - config.min_machines) // config.curriculum_stages
        
        complexities = ["simple", "medium", "complex", "complex"]
        
        for i in range(config.curriculum_stages):
            job_min = config.min_jobs + i * job_increment
            job_max = config.min_jobs + (i + 1) * job_increment
            machine_min = config.min_machines + i * machine_increment
            machine_max = config.min_machines + (i + 1) * machine_increment
            
            # Adjust final stage to use exact max values
            if i == config.curriculum_stages - 1:
                job_max = config.max_jobs
                machine_max = config.max_machines
            
            stage_name = f"stage_{i+1}"
            if i == 0:
                stage_name = "small_problems"
            elif i == 1:
                stage_name = "medium_problems"
            elif i >= 2:
                stage_name = "large_problems"
            
            # Calculate evaluation parameters
            instances_per_eval = max(10, 50 - i * 10)  # Fewer instances for larger problems
            target_success_rate = max(0.3, 0.7 - i * 0.1)  # Lower target for harder stages
            max_steps = min(config.max_episode_steps, 200 + i * 100)  # More steps for complex problems
            
            stage = CurriculumStageConfig(
                name=stage_name,
                job_range=(job_min, job_max),
                machine_range=(machine_min, machine_max),
                instances_per_eval=instances_per_eval,
                target_success_rate=target_success_rate,
                max_episode_steps=max_steps,
                complexity=complexities[min(i, len(complexities) - 1)]
            )
            
            stages.append(stage)
        
        return stages
    
    def save_config_template(self, filepath: str):
        """Save a configuration template file."""
        template_config = TrainingConfig()
        
        template_dict = {
            "_description": "POFJSP RL Training Configuration Template",
            "_presets": list(self.presets.keys()),
            "config": {
                field.name: {
                    "value": getattr(template_config, field.name),
                    "type": str(field.type),
                    "description": f"Description for {field.name}"
                }
                for field in template_config.__dataclass_fields__.values()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(template_dict, f, indent=2)
    
    def validate_config_file(self, config_path: str) -> bool:
        """Validate a configuration file."""
        try:
            config = TrainingConfig.from_json(config_path)
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global configuration manager instance
config_manager = ConfigurationManager()


# Utility functions
def get_training_config(preset: str = 'standard', config_file: Optional[str] = None, **overrides) -> TrainingConfig:
    """Get training configuration with various sources."""
    if config_file and Path(config_file).exists():
        config = TrainingConfig.from_json(config_file)
        
        # Apply overrides to loaded config
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    else:
        return config_manager.get_config(preset, **overrides)


def create_config_for_hardware(gpu_memory_gb: float) -> TrainingConfig:
    """Create configuration optimized for available hardware."""
    if gpu_memory_gb < 4:
        # Low memory configuration
        return config_manager.get_config('debug', 
                                       batch_size=64, 
                                       hidden_dim=128,
                                       max_jobs=30,
                                       max_machines=20)
    elif gpu_memory_gb < 8:
        # Medium memory configuration
        return config_manager.get_config('fast',
                                       batch_size=256,
                                       hidden_dim=256,
                                       max_jobs=50,
                                       max_machines=35)
    elif gpu_memory_gb < 16:
        # High memory configuration
        return config_manager.get_config('standard',
                                       batch_size=512,
                                       hidden_dim=512,
                                       max_jobs=100,
                                       max_machines=75)
    else:
        # Very high memory configuration
        return config_manager.get_config('production',
                                       batch_size=1024,
                                       hidden_dim=768,
                                       max_jobs=200,
                                       max_machines=150)


# Example usage and testing
if __name__ == "__main__":
    # Test different configurations
    configs = {
        'debug': config_manager.get_config('debug'),
        'fast': config_manager.get_config('fast'),
        'standard': config_manager.get_config('standard'),
        'production': config_manager.get_config('production')
    }
    
    print("Available configurations:")
    for name, config in configs.items():
        print(f"\n{name.upper()}:")
        print(f"  Total timesteps: {config.total_timesteps:,}")
        print(f"  Problem size: {config.min_jobs}-{config.max_jobs} jobs, {config.min_machines}-{config.max_machines} machines")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Hidden dim: {config.hidden_dim}")
        
        # Test curriculum stages
        stages = config_manager.create_curriculum_stages(config)
        print(f"  Curriculum stages: {len(stages)}")
        for stage in stages:
            print(f"    {stage.name}: {stage.job_range[0]}-{stage.job_range[1]} jobs, target success: {stage.target_success_rate:.1%}")
    
    # Test hardware-optimized configs
    print("\nHardware-optimized configurations:")
    for gpu_mem in [2, 6, 12, 24]:
        config = create_config_for_hardware(gpu_mem)
        print(f"  {gpu_mem}GB GPU: batch_size={config.batch_size}, hidden_dim={config.hidden_dim}, max_problems={config.max_jobs}x{config.max_machines}")