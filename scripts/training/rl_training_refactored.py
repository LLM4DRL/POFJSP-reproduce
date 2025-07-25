#!/usr/bin/env python3
"""
Refactored RL Training Script for POFJSP

Clean, modular training script using the refactored training components.
Provides a simple interface while leveraging the comprehensive training system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from training.trainer import TrainingOrchestrator
from training.config import get_training_config, create_config_for_hardware


def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Refactored RL Training for POFJSP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training mode
    parser.add_argument('--mode', choices=['debug', 'fast', 'standard', 'production'], 
                       default='standard', 
                       help='Training mode (affects hyperparameters and duration)')
    
    # Hardware configuration
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use for training')
    parser.add_argument('--auto-config', action='store_true',
                       help='Automatically configure based on available hardware')
    
    # Paths and output
    parser.add_argument('--output-dir', default='outputs/rl_training',
                       help='Output directory for training results')
    parser.add_argument('--config-file', type=str,
                       help='Path to custom configuration JSON file')
    
    # Training parameters (override config)
    parser.add_argument('--total-timesteps', type=int,
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate for optimizer')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for training')
    
    # Logging and monitoring
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                       help='Enable TensorBoard logging')
    
    # Resume training
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.verbose)
    logger.info("Starting refactored RL training for POFJSP")
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Display GPU information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = get_gpu_memory_gb()
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
        
        # Set memory fraction to avoid out-of-memory errors
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    try:
        # Get configuration
        config_overrides = {}
        if args.total_timesteps:
            config_overrides['total_timesteps'] = args.total_timesteps
        if args.learning_rate:
            config_overrides['learning_rate'] = args.learning_rate
        if args.batch_size:
            config_overrides['batch_size'] = args.batch_size
        
        if args.auto_config:
            # Use hardware-optimized configuration
            gpu_memory = get_gpu_memory_gb()
            config = create_config_for_hardware(gpu_memory)
            logger.info(f"Using hardware-optimized configuration for {gpu_memory:.1f}GB GPU")
            
            # Apply overrides
            for key, value in config_overrides.items():
                setattr(config, key, value)
        else:
            # Use standard configuration modes
            config = get_training_config(
                preset=args.mode, 
                config_file=args.config_file,
                **config_overrides
            )
        
        # Log configuration
        logger.info("Training Configuration:")
        logger.info(f"  Mode: {args.mode}")
        logger.info(f"  Total timesteps: {config.total_timesteps:,}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Problem size: {config.min_jobs}-{config.max_jobs} jobs, "
                   f"{config.min_machines}-{config.max_machines} machines")
        logger.info(f"  Curriculum stages: {config.curriculum_stages}")
        logger.info(f"  Hidden dim: {config.hidden_dim}")
        
        # Run training based on mode
        if args.mode == 'debug':
            results = TrainingOrchestrator.run_debug_training(str(output_dir), device)
        elif args.mode == 'fast':
            results = TrainingOrchestrator.run_fast_training(str(output_dir), device)
        elif args.mode == 'production':
            results = TrainingOrchestrator.run_production_training(str(output_dir), device)
        else:  # standard or custom
            results = TrainingOrchestrator.run_custom_training(config, str(output_dir), device)
        
        # Display results
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total training time: {results['total_training_time']:.1f} seconds")
        logger.info(f"Total episodes: {results['total_episodes']:,}")
        logger.info(f"Total timesteps: {results['total_timesteps']:,}")
        logger.info(f"Episodes per minute: {results['total_episodes'] / (results['total_training_time'] / 60):.1f}")
        
        # Best performance
        best_perf = results['best_performance']
        logger.info(f"\nBest Performance:")
        logger.info(f"  Win rate: {best_perf['win_rate']:.3f}")
        logger.info(f"  Stage: {best_perf['stage']}")
        logger.info(f"  Timestep: {best_perf['timestep']:,}")
        
        # Final evaluation summary
        if 'final_evaluation' in results:
            logger.info(f"\nFinal Evaluation Results:")
            for stage_name, stage_results in results['final_evaluation'].items():
                logger.info(f"  {stage_name}:")
                logger.info(f"    Win rate: {stage_results['win_rate']:.3f}")
                logger.info(f"    Makespan improvement: {stage_results['makespan_improvement']:.3f}")
                logger.info(f"    Time speedup: {stage_results['time_speedup']:.2f}x")
        
        # File locations
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"TensorBoard: tensorboard --logdir={output_dir}/tensorboard")
        logger.info(f"Best model: {output_dir}/best_model")
        logger.info(f"Final model: {output_dir}/final_model")
        logger.info(f"Training log: {output_dir}/training.log")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)