# POFJSP Reproduction

Repository for reproducing the IAOA+GNS algorithm for Partially Ordered Flexible Job Shop Scheduling Problem (POFJSP).

## Installation

```bash
# Clone the repository
git clone https://github.com/username/POFJSP-reproduce.git
cd POFJSP-reproduce

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

This project uses Hydra for configuration management. You can run the main script with different configurations:

```bash
# Run with default configuration (evaluate mode)
python main.py

# Run in dataset generation mode
python main.py mode=generate

# Run with reproduction mode (to compare with paper results)
python main.py mode=reproduce

# Use a custom experiment configuration
python main.py --config-name=custom_experiment

# Override specific parameters
python main.py algorithm.pop_size=100 dataset.max_instances=5
```

## Configuration

The configuration system is organized as follows:

- `conf/config.yaml`: Base configuration
- `conf/mode/`: Mode-specific configurations
  - `generate.yaml`: Dataset generation settings
  - `evaluate.yaml`: Algorithm evaluation settings
  - `reproduce.yaml`: Paper result reproduction settings
- `conf/dataset/`: Dataset-specific configurations
- `conf/custom_experiment.yaml`: Example of a custom experiment

## Dataset Management

The project includes tools for generating and managing POFJSP datasets:

```bash
# Generate development dataset
python main.py mode=generate dataset=development

# Generate benchmark dataset
python main.py mode=generate dataset=benchmark
```

## Algorithm Evaluation

Evaluate the IAOA+GNS algorithm on different datasets:

```bash
# Run algorithm evaluation on benchmark dataset
python main.py mode=evaluate dataset=benchmark

# Run on custom dataset with specific parameters
python main.py mode=evaluate dataset=benchmark algorithm.pop_size=100
```

## Examples

See the `examples/` directory for example scripts:

- `examples/hydra_config_example.py`: Demonstrates Hydra configuration usage
- `examples/load_and_test_datasets.py`: Shows how to load and test datasets

## Repository Structure

```
POFJSP-reproduce/
├── conf/                 # Configuration files
├── data/                 # Generated datasets
├── docs/                 # Documentation
├── examples/             # Example scripts
├── figures/              # Generated figures
├── scripts/              # Utility scripts
├── src/                  # Source code
├── tests/                # Test files
├── main.py               # Main entry point
├── README.md             # This file
└── requirements.txt      # Dependencies
```

## Citation

If you use this code, please cite the original paper:

[基于等级邻域策略的偏序柔性车间优化调度]

## License

[Your license information] 