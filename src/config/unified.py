"""
Unified Configuration Management for POFJSP

This module provides a unified configuration system that consolidates all
configuration classes and provides a consistent interface.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.exceptions import ConfigurationError, ValidationError


@dataclass
class AlgorithmConfig:
    """Configuration for scheduling algorithms."""
    name: str = "genetic"
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 300.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        if self.timeout <= 0:
            raise ConfigurationError("timeout", self.timeout, "positive number")
        
        if not isinstance(self.parameters, dict):
            raise ConfigurationError("parameters", type(self.parameters), "dictionary")


@dataclass
class ProblemConfig:
    """Configuration for problem instances."""
    data_source: str = "json"
    validation_level: str = "standard"  # "minimal", "standard", "strict"
    max_jobs: int = 1000
    max_machines: int = 100
    max_operations_per_job: int = 50
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        valid_sources = ["json", "fjsp", "dict"]
        if self.data_source not in valid_sources:
            raise ConfigurationError("data_source", self.data_source, f"one of {valid_sources}")
        
        valid_levels = ["minimal", "standard", "strict"]
        if self.validation_level not in valid_levels:
            raise ConfigurationError("validation_level", self.validation_level, f"one of {valid_levels}")
        
        if self.max_jobs <= 0:
            raise ConfigurationError("max_jobs", self.max_jobs, "positive integer")
        
        if self.max_machines <= 0:
            raise ConfigurationError("max_machines", self.max_machines, "positive integer")


@dataclass
class TrainingConfig:
    """Configuration for training procedures."""
    algorithm_suite: str = "comprehensive"  # "quick", "standard", "comprehensive"
    parallel_execution: bool = True
    timeout_per_algorithm: float = 300.0
    max_concurrent_algorithms: int = 4
    save_results: bool = True
    results_directory: str = "./outputs"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        valid_suites = ["quick", "standard", "comprehensive"]
        if self.algorithm_suite not in valid_suites:
            raise ConfigurationError("algorithm_suite", self.algorithm_suite, f"one of {valid_suites}")
        
        if self.timeout_per_algorithm <= 0:
            raise ConfigurationError("timeout_per_algorithm", self.timeout_per_algorithm, "positive number")
        
        if self.max_concurrent_algorithms <= 0:
            raise ConfigurationError("max_concurrent_algorithms", self.max_concurrent_algorithms, "positive integer")


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    enable_monitoring: bool = True
    memory_limit_mb: Optional[float] = 4000.0
    time_limit_seconds: Optional[float] = 3600.0
    cpu_limit_percent: Optional[float] = 80.0
    enable_profiling: bool = False
    profile_output_dir: str = "./profiles"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ConfigurationError("memory_limit_mb", self.memory_limit_mb, "positive number or None")
        
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ConfigurationError("time_limit_seconds", self.time_limit_seconds, "positive number or None")
        
        if self.cpu_limit_percent is not None and not (0 < self.cpu_limit_percent <= 100):
            raise ConfigurationError("cpu_limit_percent", self.cpu_limit_percent, "between 0 and 100 or None")


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    enable_plots: bool = True
    save_plots: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"
    plot_directory: str = "./plots"
    figure_size: tuple = (12, 8)
    dpi: int = 300
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        valid_formats = ["png", "pdf", "svg", "jpg"]
        if self.plot_format not in valid_formats:
            raise ConfigurationError("plot_format", self.plot_format, f"one of {valid_formats}")
        
        if self.dpi <= 0:
            raise ConfigurationError("dpi", self.dpi, "positive integer")


@dataclass
class UnifiedConfig:
    """Unified configuration containing all subsystem configurations."""
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Global settings
    random_seed: Optional[int] = 42
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate all configuration components."""
        # Individual configs validate themselves
        
        # Validate global settings
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ConfigurationError("log_level", self.log_level, f"one of {valid_log_levels}")
        
        if self.random_seed is not None and self.random_seed < 0:
            raise ConfigurationError("random_seed", self.random_seed, "non-negative integer or None")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save_to_yaml(self, file_path: Union[str, Path]):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedConfig':
        """Create configuration from dictionary."""
        # Extract subsystem configurations
        algorithm_config = AlgorithmConfig(**config_dict.get('algorithm', {}))
        problem_config = ProblemConfig(**config_dict.get('problem', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        performance_config = PerformanceConfig(**config_dict.get('performance', {}))
        visualization_config = VisualizationConfig(**config_dict.get('visualization', {}))
        
        # Extract global settings
        global_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['algorithm', 'problem', 'training', 'performance', 'visualization']}
        
        return cls(
            algorithm=algorithm_config,
            problem=problem_config,
            training=training_config,
            performance=performance_config,
            visualization=visualization_config,
            **global_settings
        )
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'UnifiedConfig':
        """Load configuration from YAML file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError("config_file", file_path, "existing file")
        except yaml.YAMLError as e:
            raise ConfigurationError("yaml_format", str(e), "valid YAML")
        
        return cls.from_dict(config_dict or {})
    
    @classmethod
    def from_environment(cls) -> 'UnifiedConfig':
        """Create configuration from environment variables."""
        config_dict = {}
        
        # Algorithm configuration
        if os.getenv('POFJSP_ALGORITHM'):
            config_dict['algorithm'] = {'name': os.getenv('POFJSP_ALGORITHM')}
        
        if os.getenv('POFJSP_TIMEOUT'):
            if 'algorithm' not in config_dict:
                config_dict['algorithm'] = {}
            config_dict['algorithm']['timeout'] = float(os.getenv('POFJSP_TIMEOUT'))
        
        # Performance configuration
        if os.getenv('POFJSP_MEMORY_LIMIT'):
            config_dict['performance'] = {'memory_limit_mb': float(os.getenv('POFJSP_MEMORY_LIMIT'))}
        
        # Global settings
        if os.getenv('POFJSP_LOG_LEVEL'):
            config_dict['log_level'] = os.getenv('POFJSP_LOG_LEVEL')
        
        if os.getenv('POFJSP_RANDOM_SEED'):
            config_dict['random_seed'] = int(os.getenv('POFJSP_RANDOM_SEED'))
        
        return cls.from_dict(config_dict)
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        import logging
        
        # Set log level
        log_level = getattr(logging, self.log_level)
        
        # Configure logging
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
        
        # File handler if specified
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
    
    def setup_random_seed(self):
        """Setup random seed based on configuration."""
        if self.random_seed is not None:
            import random
            import numpy as np
            
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            
            # Set PyTorch seed if available
            try:
                import torch
                torch.manual_seed(self.random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_seed)
            except ImportError:
                pass
    
    def initialize(self):
        """Initialize all configuration settings."""
        self.setup_logging()
        self.setup_random_seed()
        
        # Create output directories
        os.makedirs(self.training.results_directory, exist_ok=True)
        os.makedirs(self.visualization.plot_directory, exist_ok=True)
        
        if self.performance.enable_profiling:
            os.makedirs(self.performance.profile_output_dir, exist_ok=True)


# Configuration presets
def get_quick_config() -> UnifiedConfig:
    """Get configuration for quick testing."""
    return UnifiedConfig(
        algorithm=AlgorithmConfig(name="greedy", timeout=30.0),
        training=TrainingConfig(algorithm_suite="quick", timeout_per_algorithm=30.0),
        performance=PerformanceConfig(memory_limit_mb=1000.0, time_limit_seconds=60.0)
    )


def get_standard_config() -> UnifiedConfig:
    """Get standard configuration for normal use."""
    return UnifiedConfig(
        algorithm=AlgorithmConfig(name="genetic", timeout=300.0),
        training=TrainingConfig(algorithm_suite="standard", timeout_per_algorithm=300.0),
        performance=PerformanceConfig(memory_limit_mb=4000.0, time_limit_seconds=3600.0)
    )


def get_comprehensive_config() -> UnifiedConfig:
    """Get comprehensive configuration for thorough evaluation."""
    return UnifiedConfig(
        algorithm=AlgorithmConfig(name="iaoa_gns", timeout=600.0),
        training=TrainingConfig(algorithm_suite="comprehensive", timeout_per_algorithm=600.0),
        performance=PerformanceConfig(memory_limit_mb=8000.0, time_limit_seconds=7200.0),
        visualization=VisualizationConfig(enable_plots=True, save_plots=True)
    )


def get_config(config_source: Optional[str] = None) -> UnifiedConfig:
    """
    Get configuration from various sources.
    
    Args:
        config_source: Configuration source ('quick', 'standard', 'comprehensive', 
                      file path, or None for environment/defaults)
    
    Returns:
        UnifiedConfig object
    """
    if config_source == "quick":
        return get_quick_config()
    elif config_source == "standard":
        return get_standard_config()
    elif config_source == "comprehensive":
        return get_comprehensive_config()
    elif config_source and os.path.exists(config_source):
        return UnifiedConfig.from_yaml(config_source)
    else:
        # Try environment variables, fall back to standard config
        try:
            return UnifiedConfig.from_environment()
        except:
            return get_standard_config()


# Global configuration instance
_global_config: Optional[UnifiedConfig] = None


def set_global_config(config: UnifiedConfig):
    """Set the global configuration."""
    global _global_config
    _global_config = config
    config.initialize()


def get_global_config() -> UnifiedConfig:
    """Get the global configuration."""
    global _global_config
    if _global_config is None:
        _global_config = get_config()
        _global_config.initialize()
    return _global_config


def reset_global_config():
    """Reset the global configuration."""
    global _global_config
    _global_config = None