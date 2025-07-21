# POFJSP Examples

This folder contains focused examples demonstrating the POFJSP (Partially Ordered Flexible Job Shop Problem) solution system.

## Available Examples

### 1. `simple_example.py` - Core Algorithm Demonstration
Complete demonstration of the IAOA+GNS algorithm with:
- Simple 2-job, 2-machine problem creation and solving
- Complex 3-job, 3-machine problem with parallel operations  
- Algorithm parameter comparison
- Dataset loading and testing functionality

**Usage:**
```bash
python examples/simple_example.py
```

### 2. `algorithm_parameter_study.py` - Parameter Optimization
Systematic parameter study for algorithm tuning:
- Tests different combinations of population size, iterations, crossover/mutation rates
- Uses the control center interface for dataset generation and evaluation
- Generates performance analysis and identifies optimal parameters

**Usage:**
```bash
python examples/algorithm_parameter_study.py
```

### 3. `visualization_example.py` - Comprehensive Visualization
Advanced visualization capabilities:
- Dataset distribution and complexity analysis
- Gantt chart generation for solutions
- Comparative visualizations between algorithm configurations
- Algorithm performance metrics and convergence plots

**Usage:**
```bash
python examples/visualization_example.py
```

## Integration with Control Center

All examples now integrate with the main control center (`main.py`) and use the standardized commands:

- **Generate datasets:** `python main.py generate-data --dataset-name NAME`
- **Evaluate algorithms:** `python main.py evaluate --dataset-name NAME`
- **Format code:** `python main.py format-code`
- **Run health check:** `python main.py health-check`

## Prerequisites

Before running examples, ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

For visualization examples, you may need additional plotting libraries already included in requirements.txt.

## Getting Started

1. **Start with simple_example.py** to understand the algorithm basics
2. **Use algorithm_parameter_study.py** to optimize parameters for your specific use case
3. **Run visualization_example.py** to generate publication-ready charts and analysis

Each example is self-contained and includes detailed output explaining the results.