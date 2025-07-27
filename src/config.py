"""
Configuration Management System

Centralized configuration for all POFJSP algorithms and training parameters.
Eliminates magic numbers and provides environment-based configuration.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

from src.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    # Learning hyperparameters
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training parameters
    gamma: float = 0.99
    lam: float = 0.95
    batch_size: int = 64
    epochs: int = 10
    min_batch_size: int = 32
    
    # Memory management
    use_mixed_precision: bool = True
    memory_cleanup_interval: int = 100
    
    # Problem size adjustments
    small_problem_lr_multiplier: float = 0.5
    large_problem_lr_multiplier: float = 2.0
    
    def get_adjusted_learning_rate(self, problem_size: int) -> float:
        """Get learning rate adjusted for problem size."""
        if problem_size <= 50:  # Small problems
            return self.learning_rate * self.small_problem_lr_multiplier
        elif problem_size >= 200:  # Large problems
            return self.learning_rate * self.large_problem_lr_multiplier
        return self.learning_rate
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.learning_rate < 1:
            raise ConfigurationError("learning_rate", self.learning_rate, "between 0 and 1")
        if not 0 < self.clip_ratio < 1:
            raise ConfigurationError("clip_ratio", self.clip_ratio, "between 0 and 1")
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size", self.batch_size, "positive integer")


@dataclass
class IAOAGNSConfig:
    """IAOA+GNS algorithm configuration."""
    # Population parameters
    pop_size: int = 80
    max_iterations: int = 60
    
    # MOA parameters
    moa_min: float = 0.2
    moa_max: float = 1.0
    
    # Crossover parameters
    crossover_probability: float = 0.5
    cluster_count: int = 3
    
    # Mutation parameters
    mutation_probability: float = 0.3
    max_mutations_per_solution: int = 3
    
    # GNS parameters
    gns_probability: float = 0.5
    max_neighborhood_size: int = 10
    
    # Performance parameters
    early_stopping_patience: int = 10
    convergence_threshold: float = 0.01
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.pop_size <= 0:
            raise ConfigurationError("pop_size", self.pop_size, "positive integer")
        if self.max_iterations <= 0:
            raise ConfigurationError("max_iterations", self.max_iterations, "positive integer")
        if not 0 <= self.moa_min <= self.moa_max <= 1:
            raise ConfigurationError("moa_range", [self.moa_min, self.moa_max], "valid range [0,1]")


@dataclass
class EnvironmentConfig:
    """RL environment configuration."""
    # Episode parameters
    max_episode_steps: int = 1000
    time_limit_multiplier: float = 2.0
    
    # Reward configuration
    reward_type: str = "makespan"  # "makespan", "utilization", "combined"
    normalize_reward: bool = True
    reward_shaping: bool = True
    
    # State representation
    node_feature_dim: int = 8
    include_processing_times: bool = True
    include_precedence_info: bool = True
    
    # Action space
    invalid_action_penalty: float = -10.0
    invalid_machine_penalty: float = -5.0
    
    def validate(self) -> None:
        """Validate environment configuration."""
        if self.max_episode_steps <= 0:
            raise ConfigurationError("max_episode_steps", self.max_episode_steps, "positive integer")
        if self.reward_type not in ["makespan", "utilization", "combined"]:
            raise ConfigurationError("reward_type", self.reward_type, "one of: makespan, utilization, combined")


@dataclass
class TrainingConfig:
    """Training process configuration."""
    # Training schedule
    num_episodes: int = 1000
    eval_interval: int = 100
    save_interval: int = 500
    
    # Performance monitoring
    log_level: str = "INFO"
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    
    # Hardware configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Memory management
    memory_monitoring: bool = True
    memory_limit_gb: float = 8.0
    cleanup_interval: int = 100
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    keep_top_k: int = 5
    
    def get_device(self) -> str:
        """Get the appropriate device for training."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.num_episodes <= 0:
            raise ConfigurationError("num_episodes", self.num_episodes, "positive integer")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ConfigurationError("log_level", self.log_level, "valid log level")


@dataclass
class ProblemConfig:
    """Problem instance configuration."""
    # Problem size parameters
    min_jobs: int = 5
    max_jobs: int = 20
    min_machines: int = 3
    max_machines: int = 10
    min_operations_per_job: int = 2
    max_operations_per_job: int = 8
    
    # Processing time parameters
    min_processing_time: float = 1.0
    max_processing_time: float = 50.0
    infeasible_probability: float = 0.3  # Probability an operation can't be processed on a machine
    
    # Precedence constraint parameters
    precedence_density: float = 0.4  # Density of precedence constraints
    max_predecessors: int = 3
    
    def validate(self) -> None:
        """Validate problem configuration."""
        if self.min_jobs <= 0 or self.max_jobs < self.min_jobs:
            raise ConfigurationError("job_range", [self.min_jobs, self.max_jobs], "valid range")
        if self.min_machines <= 0 or self.max_machines < self.min_machines:
            raise ConfigurationError("machine_range", [self.min_machines, self.max_machines], "valid range")


@dataclass
class MainConfig:
    """Main configuration container."""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    iaoa_gns: IAOAGNSConfig = field(default_factory=IAOAGNSConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    
    def validate_all(self) -> None:
        """Validate all configuration sections."""
        self.ppo.validate()
        self.iaoa_gns.validate()
        self.environment.validate()
        self.training.validate()
        self.problem.validate()
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'MainConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise ValidationError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create config objects from dictionary
            config = cls()
            
            if 'ppo' in config_dict:
                config.ppo = PPOConfig(**config_dict['ppo'])
            if 'iaoa_gns' in config_dict:
                config.iaoa_gns = IAOAGNSConfig(**config_dict['iaoa_gns'])
            if 'environment' in config_dict:
                config.environment = EnvironmentConfig(**config_dict['environment'])
            if 'training' in config_dict:
                config.training = TrainingConfig(**config_dict['training'])
            if 'problem' in config_dict:
                config.problem = ProblemConfig(**config_dict['problem'])
            
            config.validate_all()
            logger.info(f"Loaded configuration from {yaml_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML in {yaml_path}: {e}")
        except Exception as e:
            raise ValidationError(f"Error loading configuration: {e}")
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = {
            'ppo': {
                'learning_rate': self.ppo.learning_rate,
                'clip_ratio': self.ppo.clip_ratio,
                'value_loss_coef': self.ppo.value_loss_coef,
                'entropy_coef': self.ppo.entropy_coef,
                'max_grad_norm': self.ppo.max_grad_norm,
                'gamma': self.ppo.gamma,
                'lam': self.ppo.lam,
                'batch_size': self.ppo.batch_size,
                'epochs': self.ppo.epochs
            },
            'iaoa_gns': {
                'pop_size': self.iaoa_gns.pop_size,
                'max_iterations': self.iaoa_gns.max_iterations,
                'moa_min': self.iaoa_gns.moa_min,
                'moa_max': self.iaoa_gns.moa_max
            },
            'environment': {
                'max_episode_steps': self.environment.max_episode_steps,
                'reward_type': self.environment.reward_type,
                'normalize_reward': self.environment.normalize_reward
            },
            'training': {
                'num_episodes': self.training.num_episodes,
                'eval_interval': self.training.eval_interval,
                'device': self.training.device,
                'log_level': self.training.log_level
            }
        }
        
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {yaml_path}")
        except Exception as e:
            raise ValidationError(f"Error saving configuration: {e}")


class ConfigManager:
    """Configuration manager with environment variable support."""
    
    def __init__(self):
        self._config: Optional[MainConfig] = None
        self._config_path: Optional[Path] = None
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> MainConfig:
        """
        Load configuration with environment variable overrides.
        
        Args:
            config_path: Path to configuration file. If None, uses default locations.
            
        Returns:
            Loaded configuration
        """
        if config_path is None:
            config_path = self._find_default_config()
        
        config_path = Path(config_path)
        
        if config_path.exists():
            self._config = MainConfig.from_yaml(config_path)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self._config = MainConfig()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate final configuration
        self._config.validate_all()
        
        self._config_path = config_path
        return self._config
    
    def _find_default_config(self) -> Path:
        """Find default configuration file."""
        possible_paths = [
            Path("config.yaml"),
            Path("conf/config.yaml"),
            Path("configs/config.yaml"),
            Path.home() / ".pofjsp" / "config.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return first option as default location
        return possible_paths[0]
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        if not self._config:
            return
        
        # PPO overrides
        if lr := os.getenv("POFJSP_PPO_LEARNING_RATE"):
            self._config.ppo.learning_rate = float(lr)
        if batch_size := os.getenv("POFJSP_PPO_BATCH_SIZE"):
            self._config.ppo.batch_size = int(batch_size)
        
        # Training overrides
        if device := os.getenv("POFJSP_DEVICE"):
            self._config.training.device = device
        if episodes := os.getenv("POFJSP_NUM_EPISODES"):
            self._config.training.num_episodes = int(episodes)
        
        # IAOA+GNS overrides
        if pop_size := os.getenv("POFJSP_IAOA_POP_SIZE"):
            self._config.iaoa_gns.pop_size = int(pop_size)
        if iterations := os.getenv("POFJSP_IAOA_MAX_ITERATIONS"):
            self._config.iaoa_gns.max_iterations = int(iterations)
    
    @property
    def config(self) -> MainConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> MainConfig:
        """Reload configuration from file."""
        if self._config_path:
            return self.load_config(self._config_path)
        return self.load_config()


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience function for getting configuration
def get_config() -> MainConfig:
    """Get global configuration instance."""
    return config_manager.config