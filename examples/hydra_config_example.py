#!/usr/bin/env python3
"""
Example script demonstrating how to use Hydra configuration for POFJSP.

This script shows how to:
1. Run with different modes
2. Override configuration parameters from the command line
3. Use different configuration files

Example usage:
    # Run with default configuration
    python examples/hydra_config_example.py
    
    # Run with generate mode
    python examples/hydra_config_example.py mode=generate
    
    # Run with custom experiment configuration
    python examples/hydra_config_example.py --config-name=custom_experiment
    
    # Override specific parameters
    python examples/hydra_config_example.py algorithm.pop_size=100 dataset.max_instances=5
"""

import sys
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Demo function showing how to access Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    print(f"{'='*50}")
    print(f"POFJSP Hydra Configuration Example")
    print(f"{'='*50}")
    
    # Print basic configuration info
    print(f"Working directory: {os.getcwd()}")
    print(f"Original working directory: {hydra.utils.get_original_cwd()}")
    print(f"Mode: {cfg.mode}")
    print(f"Random seed: {cfg.random_seed}")
    
    # Print algorithm configuration
    print(f"\nAlgorithm Configuration:")
    print(f"  Population size: {cfg.algorithm.pop_size}")
    print(f"  Max iterations: {cfg.algorithm.max_iterations}")
    print(f"  Crossover probability: {cfg.algorithm.crossover_prob}")
    print(f"  Mutation probability: {cfg.algorithm.mutation_prob}")
    
    # Print dataset configuration
    print(f"\nDataset Configuration:")
    print(f"  Dataset name: {cfg.dataset.name}")
    print(f"  Dataset path: {cfg.dataset.path}")
    if hasattr(cfg.dataset, 'size_filter') and cfg.dataset.size_filter:
        print(f"  Size filter: {cfg.dataset.size_filter}")
    
    # Print full configuration as YAML
    print(f"\nFull Configuration (YAML):")
    print(f"{OmegaConf.to_yaml(cfg)}")
    
    return 0


if __name__ == "__main__":
    main() 