# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a POFJSP (Partially Ordered Flexible Job Shop Problem) research repository implementing multiple optimization algorithms including IAOA+GNS, Genetic Algorithm, Simulated Annealing, Tabu Search, and Reinforcement Learning with GNN+PPO. The project supports dataset generation, algorithm evaluation, and performance benchmarking.

## Common Commands

### Running Algorithms
```bash
# Main unified interface
python main.py --algorithm iaoa_gns --sample                # Run IAOA+GNS on sample
python main.py --algorithm ga --problem data/benchmark/sample.json    # Run GA on specific problem
python main.py --algorithm rl --problem data/benchmark/sample.json --mode train    # Train RL model
python main.py --algorithm sa --dataset data/development --dataset-mode    # Run SA on dataset
python main.py --list-algorithms                            # List available algorithms

# Direct algorithm execution
python run_iaoa_gns.py --sample                            # Run IAOA+GNS directly
python run_ga.py data/benchmark/sample.json                # Run GA directly
python run_rl_pofjsp.py data/benchmark/sample.json --mode train    # Run RL directly
python run_sa.py data/benchmark/sample.json                # Run SA directly
python run_tabu.py data/benchmark/sample.json              # Run Tabu Search directly

# Dataset generation
python generate_pofjsp_dataset.py                          # Generate POFJSP datasets
```

### Testing
```bash
python test_algorithms.py                                  # Test algorithm implementations
python -m pytest tests/                                    # Run all tests
python -m pytest tests/test_algorithms.py                  # Run specific test file
python -m pytest tests/test_rl.py                         # Test RL components
```

### Development
```bash
pip install -r requirements.txt                            # Install dependencies
python -m pytest --cov=src tests/                         # Run tests with coverage
```

## Architecture Overview

### Core Structure
- **`src/algorithms/`**: Algorithm implementations (IAOA+GNS, GA, SA, Tabu Search)
- **`src/problems/`**: Problem definition and instance management 
- **`src/rl/`**: Reinforcement Learning implementation with PPO+GNN
- **`src/utils/`**: Shared utilities and helper functions
- **`src/visualization/`**: Gantt charts, analysis, and result visualization

### Algorithm Architecture
The codebase implements a modular algorithm framework:
- **Common Interface**: All algorithms inherit from a base algorithm class
- **Problem Instance**: Unified `ProblemInstance` class handles POFJSP problems
- **Decoder**: Shared solution decoding and evaluation logic
- **Multi-modal Entry**: Both individual algorithm runners and unified `main.py` interface

### RL Architecture (PPO + GNN)
The RL implementation uses a sophisticated graph-based approach:
- **Environment**: `src/rl/environments/pofjsp_env.py` - Graph-based POFJSP environment
- **Models**: 
  - `GraphCNN`: Graph neural network for processing precedence constraints
  - `HierarchicalActor`: Two-level decision making (job → machine selection)
  - `PPOAgent`: Complete PPO implementation with GAE
- **Training**: Hierarchical action space with precedence-aware masking

### Dataset Management
- **Structured datasets**: `data/benchmark/`, `data/development/`, `data/performance/`
- **Multiple formats**: JSON problem instances with metadata
- **Hierarchical organization**: Problems organized by size, type, and difficulty

### Configuration System
Uses Hydra for configuration management:
- **Base config**: `conf/config.yaml`
- **Dataset configs**: `conf/dataset/benchmark.yaml`
- **Extensible**: Easy to add new configurations

## Key Implementation Details

### Problem Representation
- **Operations**: Represented as `Operation(job_id, operation_id)` tuples
- **Precedence**: Stored as predecessor/successor maps with networkx graphs
- **Flexibility**: Machine-operation compatibility via processing time matrices (np.inf = incompatible)

### Algorithm Parameters
- **IAOA+GNS**: `pop_size=80`, `max_iter=60` (default)
- **GA**: `population_size=100`, `max_generations=1000`
- **SA**: `initial_temp=100.0`, `cooling_rate=0.95`
- **RL**: `total_timesteps=100000`, `batch_size=64`, `learning_rate=3e-4`

### RL Status
**Fixed Issues**: 
- ✅ Missing import `torch.nn.functional as F` in `src/rl/models/ppo_agent.py` 
- ✅ Device detection works (CPU/CUDA auto-detection)
- ✅ Flexible dataset generation for variable problem sizes (30x50, 40x40, etc.)

**Current Status**: Core RL components work but need refinement for multi-instance training

### Testing Strategy
- **Unit tests**: Algorithm correctness verification
- **Integration tests**: End-to-end workflow testing
- **Sample problems**: Small test instances for development

## Development Workflow

### Adding New Algorithms
1. Implement in `src/algorithms/new_algorithm.py`
2. Create runner script `run_new_algorithm.py`
3. Add to `main.py` algorithm dispatcher
4. Add tests in `tests/`

### Working with RL
1. **Environment**: Modify `src/rl/environments/pofjsp_env.py` for new features
2. **Models**: Extend GNN architecture in `src/rl/models/`
3. **Training**: Customize PPO parameters in `run_rl_pofjsp.py`
4. **Known issue**: Fix the missing import in `ppo_agent.py` before training

### Dataset Extension
1. Use `generate_pofjsp_dataset.py` for new dataset creation
2. Follow existing naming conventions in `data/` directories
3. Update dataset configurations in `conf/dataset/`

## Dependencies

Core requirements:
- **Scientific**: `numpy>=1.20.0`, `scipy>=1.7.0`, `pandas>=1.3.0`
- **ML/RL**: `torch>=1.12.0`, `torch-geometric>=2.1.0`, `stable-baselines3>=1.6.0`
- **Optimization**: `scikit-learn>=0.24.0`, `networkx>=2.6.0`
- **Config**: `hydra-core>=1.1.0`, `omegaconf>=2.1.0`
- **Visualization**: `matplotlib>=3.4.0`, `seaborn>=0.11.0`
- **Testing**: `pytest>=6.0.0`, `pytest-cov>=2.10.0`

## Important Notes

### Code Quality
- The codebase follows object-oriented design patterns
- Comprehensive error handling and input validation
- Extensive documentation and type hints
- Modular architecture supports easy extension

### Performance Considerations
- Uses `joblib` for parallel processing in batch evaluations
- Efficient numpy operations for algorithm computations
- Graph-based representations optimize memory usage
- GPU support for RL training (when available)

### Research Context
This is a research reproduction repository focusing on the IAOA+GNS algorithm with comparative analysis against other metaheuristics and modern RL approaches. The implementation prioritizes correctness and reproducibility over production optimization.