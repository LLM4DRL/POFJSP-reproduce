#!/usr/bin/env python3
"""
POFJSP Control Center - Multi-Task Handler

This script serves as the central control hub for the POFJSP repository, providing:
- Algorithm execution (IAOA+GNS, GA, SA, Tabu, RL)
- Code formatting and linting
- Repository health checks
- Git pre-commit utilities
- Training pipeline management
- Data processing tools

Usage:
    # Algorithm execution
    python main.py --algorithm iaoa_gns --problem data/benchmark/sample.json
    python main.py --algorithm rl --train --fast-mode
    
    # Development tools
    python main.py --format --check
    python main.py --health-check
    python main.py --setup-hooks
    
    # Training management
    python main.py --train-rl --output-dir ./outputs/full_training
    python main.py --compare-algorithms --dataset data/benchmark
"""

import argparse
import os
import sys
import json
import subprocess
import time
from pathlib import Path
import importlib.util
from typing import Dict, List, Optional

# Add current directory to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))


class POFJSPControlCenter:
    """Central control hub for POFJSP repository operations."""
    
    def __init__(self, verbose: bool = False):
        self.repo_root = Path(__file__).parent
        self.verbose = verbose
        self.available_algorithms = {
            'iaoa_gns': 'IAOA+GNS (Hierarchical Neighborhood Strategy)',
            'ga': 'Genetic Algorithm',
            'sa': 'Simulated Annealing',
            'tabu': 'Tabu Search',
            'rl': 'Reinforcement Learning (GNN+PPO)'
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        if self.verbose or level in ["ERROR", "SUCCESS"]:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    # =============================================================================
    # ALGORITHM EXECUTION
    # =============================================================================
    
    def run_algorithm(self, algorithm: str, **kwargs) -> int:
        """Run specified algorithm with parameters."""
        self.log(f"Starting {algorithm} algorithm...")
        
        if algorithm == 'iaoa_gns':
            return self._run_iaoa_gns(**kwargs)
        elif algorithm == 'ga':
            return self._run_genetic_algorithm(**kwargs)
        elif algorithm == 'sa':
            return self._run_simulated_annealing(**kwargs)
        elif algorithm == 'tabu':
            return self._run_tabu_search(**kwargs)
        elif algorithm == 'rl':
            return self._run_reinforcement_learning(**kwargs)
        else:
            self.log(f"Unknown algorithm: {algorithm}", "ERROR")
            return 1
    
    def _run_iaoa_gns(self, **kwargs) -> int:
        """Run IAOA+GNS algorithm."""
        cmd = [sys.executable, 'src/algorithms/iaoa_gns.py']
        return self._execute_command(cmd)
    
    def _run_genetic_algorithm(self, **kwargs) -> int:
        """Run Genetic Algorithm."""
        cmd = [sys.executable, 'src/algorithms/genetic_algorithm.py']
        return self._execute_command(cmd)
    
    def _run_simulated_annealing(self, **kwargs) -> int:
        """Run Simulated Annealing."""
        cmd = [sys.executable, 'src/algorithms/simulated_annealing.py']
        return self._execute_command(cmd)
    
    def _run_tabu_search(self, **kwargs) -> int:
        """Run Tabu Search."""
        cmd = [sys.executable, 'src/algorithms/tabu_search.py']
        return self._execute_command(cmd)
    
    def _run_reinforcement_learning(self, **kwargs) -> int:
        """Run Reinforcement Learning training."""
        if kwargs.get('train'):
            cmd = [sys.executable, 'scripts/training/rl_training.py']
            if kwargs.get('output_dir'):
                cmd.extend(['--output-dir', kwargs['output_dir']])
            if kwargs.get('fast_mode'):
                cmd.append('--fast-mode')
            if not kwargs.get('cuda', True):
                cmd.append('--no-cuda')
        else:
            # Run RL solve mode (placeholder)
            self.log("RL solve mode not yet implemented", "ERROR")
            return 1
        
        return self._execute_command(cmd)
    
    # =============================================================================
    # CODE FORMATTING AND LINTING
    # =============================================================================
    
    def format_code(self, check_only: bool = False, files: Optional[List[str]] = None) -> int:
        """Format Python code using ruff and black."""
        self.log("Starting code formatting...")
        
        # Install formatting tools if not available
        if not self._check_tool_available('ruff'):
            self.log("Installing ruff...")
            if self._execute_command([sys.executable, '-m', 'pip', 'install', 'ruff']) != 0:
                self.log("Failed to install ruff", "ERROR")
                return 1
        
        if not self._check_tool_available('black'):
            self.log("Installing black...")
            if self._execute_command([sys.executable, '-m', 'pip', 'install', 'black']) != 0:
                self.log("Failed to install black", "ERROR")
                return 1
        
        # Determine files to format
        if files is None:
            files = ['src', 'scripts', 'tests', 'main.py']
        
        exit_code = 0
        
        # Run ruff for linting and import sorting
        self.log("Running ruff linter...")
        ruff_cmd = [sys.executable, '-m', 'ruff', 'check'] + files
        if not check_only:
            ruff_cmd.append('--fix')
        
        if self._execute_command(ruff_cmd) != 0:
            exit_code = 1
        
        # Run black for code formatting
        self.log("Running black formatter...")
        black_cmd = [sys.executable, '-m', 'black']
        if check_only:
            black_cmd.append('--check')
        black_cmd.extend(files)
        
        if self._execute_command(black_cmd) != 0:
            exit_code = 1
        
        if exit_code == 0:
            self.log("Code formatting completed successfully", "SUCCESS")
        else:
            self.log("Code formatting found issues", "ERROR")
        
        return exit_code
    
    def run_type_check(self) -> int:
        """Run mypy type checking."""
        self.log("Running type checking...")
        
        if not self._check_tool_available('mypy'):
            self.log("Installing mypy...")
            if self._execute_command([sys.executable, '-m', 'pip', 'install', 'mypy']) != 0:
                self.log("Failed to install mypy", "ERROR")
                return 1
        
        cmd = [sys.executable, '-m', 'mypy', 'src', '--ignore-missing-imports']
        return self._execute_command(cmd)
    
    # =============================================================================
    # REPOSITORY HEALTH CHECK
    # =============================================================================
    
    def run_health_check(self) -> int:
        """Run repository health check."""
        self.log("Running repository health check...")
        
        health_check_script = self.repo_root / 'scripts' / 'repo_health_check.py'
        if not health_check_script.exists():
            self.log("Health check script not found", "ERROR")
            return 1
        
        cmd = [sys.executable, str(health_check_script)]
        if self.verbose:
            cmd.append('--verbose')
        
        return self._execute_command(cmd)
    
    # =============================================================================
    # GIT AND PRE-COMMIT HOOKS
    # =============================================================================
    
    def setup_git_hooks(self) -> int:
        """Setup git pre-commit hooks."""
        self.log("Setting up git pre-commit hooks...")
        
        # Install pre-commit if not available
        if not self._check_tool_available('pre-commit'):
            self.log("Installing pre-commit...")
            if self._execute_command([sys.executable, '-m', 'pip', 'install', 'pre-commit']) != 0:
                self.log("Failed to install pre-commit", "ERROR")
                return 1
        
        # Create .pre-commit-config.yaml if it doesn't exist
        precommit_config = self.repo_root / '.pre-commit-config.yaml'
        if not precommit_config.exists():
            self._create_precommit_config(precommit_config)
        
        # Install the hooks
        cmd = [sys.executable, '-m', 'pre-commit', 'install']
        return self._execute_command(cmd)
    
    def _create_precommit_config(self, config_path: Path):
        """Create a pre-commit configuration file."""
        config_content = """repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10000']
      - id: check-merge-conflict
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
        self.log(f"Created pre-commit config: {config_path}")
    
    def run_pre_commit_check(self) -> int:
        """Run pre-commit checks on all files."""
        self.log("Running pre-commit checks...")
        
        if not self._check_tool_available('pre-commit'):
            self.log("pre-commit not installed. Run --setup-hooks first.", "ERROR")
            return 1
        
        cmd = [sys.executable, '-m', 'pre-commit', 'run', '--all-files']
        return self._execute_command(cmd)
    
    # =============================================================================
    # TRAINING MANAGEMENT
    # =============================================================================
    
    def run_full_rl_training(self, output_dir: str = './outputs/full_training', 
                           fast_mode: bool = False, no_cuda: bool = False) -> int:
        """Run comprehensive RL training pipeline."""
        self.log("Starting comprehensive RL training...")
        
        cmd = [sys.executable, 'scripts/training/rl_training.py', '--output-dir', output_dir]
        if fast_mode:
            cmd.append('--fast-mode')
        if no_cuda:
            cmd.append('--no-cuda')
        
        return self._execute_command(cmd)
    
    def compare_algorithms(self, dataset_dir: str) -> int:
        """Run algorithm comparison on dataset."""
        self.log("Running algorithm comparison...")
        
        # This would run all algorithms on the same dataset and compare results
        # For now, just run the algorithms sequentially
        results = {}
        
        for algo in self.available_algorithms.keys():
            if algo == 'rl':  # Skip RL for now as it needs training
                continue
            
            self.log(f"Running {algo}...")
            start_time = time.time()
            exit_code = self.run_algorithm(algo)
            end_time = time.time()
            
            results[algo] = {
                'exit_code': exit_code,
                'duration': end_time - start_time
            }
        
        # Print comparison results
        self.log("Algorithm comparison results:", "SUCCESS")
        for algo, result in results.items():
            status = "SUCCESS" if result['exit_code'] == 0 else "FAILED"
            duration = result['duration']
            print(f"  {algo}: {status} ({duration:.2f}s)")
        
        return 0
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _check_tool_available(self, tool: str) -> bool:
        """Check if a command-line tool is available."""
        try:
            subprocess.run([tool, '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _execute_command(self, cmd: List[str]) -> int:
        """Execute a command and return exit code."""
        try:
            if self.verbose:
                self.log(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, cwd=self.repo_root)
            return result.returncode
        except Exception as e:
            self.log(f"Command execution failed: {e}", "ERROR")
            return 1
    
    def list_algorithms(self):
        """List available algorithms."""
        print("Available algorithms:")
        for algo, desc in self.available_algorithms.items():
            print(f"  {algo}: {desc}")
    
    def show_status(self):
        """Show repository and system status."""
        print("POFJSP Repository Status")
        print("=" * 50)
        
        # Check git status
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.repo_root)
            if result.returncode == 0:
                changes = result.stdout.strip()
                if changes:
                    print(f"Git: {len(changes.splitlines())} uncommitted changes")
                else:
                    print("Git: Clean working directory")
            else:
                print("Git: Not a git repository or git not available")
        except FileNotFoundError:
            print("Git: Not available")
        
        # Check Python environment
        print(f"Python: {sys.version.split()[0]} ({sys.executable})")
        
        # Check key dependencies
        key_packages = ['torch', 'numpy', 'matplotlib', 'hydra-core']
        for package in key_packages:
            try:
                __import__(package)
                print(f"{package}: [OK] Available")
            except ImportError:
                print(f"{package}: [MISSING] Not installed")
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA: [OK] Available ({torch.cuda.device_count()} devices)")
            else:
                print("CUDA: [NO] Not available")
        except ImportError:
            print("CUDA: [NO] PyTorch not available")


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="POFJSP Control Center - Multi-Task Handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Algorithm execution
  python main.py --algorithm iaoa_gns
  python main.py --algorithm rl --train --fast-mode
  
  # Development tools
  python main.py --format --check
  python main.py --health-check
  python main.py --setup-hooks
  
  # Training and analysis
  python main.py --train-rl --output-dir ./outputs/production
  python main.py --compare-algorithms --dataset data/benchmark
  python main.py --status
        """
    )
    
    # Main operation modes
    parser.add_argument('--algorithm', '-a', 
                       choices=['iaoa_gns', 'ga', 'sa', 'tabu', 'rl'],
                       help='Run specific algorithm')
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List available algorithms')
    parser.add_argument('--status', action='store_true',
                       help='Show repository and system status')
    
    # Development tools
    parser.add_argument('--format', action='store_true',
                       help='Format code using ruff and black')
    parser.add_argument('--check', action='store_true',
                       help='Check code formatting without making changes')
    parser.add_argument('--type-check', action='store_true',
                       help='Run mypy type checking')
    parser.add_argument('--health-check', action='store_true',
                       help='Run repository health check')
    parser.add_argument('--setup-hooks', action='store_true',
                       help='Setup git pre-commit hooks')
    parser.add_argument('--pre-commit-check', action='store_true',
                       help='Run pre-commit checks on all files')
    
    # Training and analysis
    parser.add_argument('--train-rl', action='store_true',
                       help='Run comprehensive RL training')
    parser.add_argument('--train', action='store_true',
                       help='Enable training mode for RL')
    parser.add_argument('--compare-algorithms', action='store_true',
                       help='Compare all algorithms on dataset')
    
    # Options
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--dataset', type=str,
                       help='Dataset directory for analysis')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use fast mode (reduced training time)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA acceleration')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create control center
    control_center = POFJSPControlCenter(verbose=args.verbose)
    
    # Handle different operation modes
    if args.list_algorithms:
        control_center.list_algorithms()
        return 0
    
    if args.status:
        control_center.show_status()
        return 0
    
    if args.algorithm:
        return control_center.run_algorithm(
            args.algorithm,
            train=args.train,
            output_dir=args.output_dir,
            fast_mode=args.fast_mode,
            cuda=not args.no_cuda
        )
    
    if args.format:
        return control_center.format_code(check_only=args.check)
    
    if args.type_check:
        return control_center.run_type_check()
    
    if args.health_check:
        return control_center.run_health_check()
    
    if args.setup_hooks:
        return control_center.setup_git_hooks()
    
    if args.pre_commit_check:
        return control_center.run_pre_commit_check()
    
    if args.train_rl:
        return control_center.run_full_rl_training(
            output_dir=args.output_dir,
            fast_mode=args.fast_mode,
            no_cuda=args.no_cuda
        )
    
    if args.compare_algorithms:
        if not args.dataset:
            print("Error: --dataset required for algorithm comparison")
            return 1
        return control_center.compare_algorithms(args.dataset)
    
    # If no specific command given, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())