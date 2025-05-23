# POFJSP Reproduction - Repository Guidelines

## Project Overview

This repository contains a reproduction implementation of the paper:

**"基于等级邻域策略的偏序柔性车间优化调度" (Optimal scheduling of partially ordered flexible job shop based on hierarchical neighborhood strategy)**

*Authors: 李长云, 林多, 何频捷, 谷鹏飞*  
*Published in: 管理信息化 (Management and Informatization), 2022*

### Problem Description

The Partially Ordered Flexible Job Shop Scheduling Problem (POFJSP) addresses scheduling challenges in manufacturing environments (particularly garment production) where:
- Workpieces have complex partial order relationships in their processing sequences
- Multiple operations can be processed in parallel after completing common predecessor operations
- The objective is to minimize maximum completion time (makespan)

### Algorithm: IAOA+GNS

The implementation features an **Improved Arithmetic Optimization Algorithm (IAOA)** enhanced with **Grade Neighborhood Search (GNS)** strategy:

- **Exploration Phase**: 2D clustering crossover and effective parallel mutation
- **Development Phase**: Hierarchical neighborhood search for bottleneck jobs and machines

## Repository Structure

```
POFJSP-reproduce/
├── src/
│   ├── algorithms.py          # Main IAOA+GNS implementation
│   ├── problem_loader.py      # Problem instance loading utilities
│   ├── benchmarks.py          # Benchmark problem generators
│   └── visualization.py       # Gantt chart and result visualization
├── data/
│   ├── instances/             # Problem instance files
│   ├── benchmarks/            # Extended benchmark instances (PMk01-PMk10)
│   └── results/               # Algorithm execution results
├── tests/
│   ├── test_algorithms.py     # Unit tests for algorithms
│   ├── test_decoding.py       # Tests for solution decoding
│   └── test_instances.py      # Tests for problem instances
├── docs/
│   ├── algorithm_details.md   # Detailed algorithm description
│   ├── problem_format.md      # Problem instance format specification
│   └── paper_reproduction.md  # Notes on paper reproduction
├── figures/
│   ├── gantt_charts/          # Generated Gantt charts
│   └── convergence_plots/     # Algorithm convergence plots
├── requirements.txt           # Python dependencies
├── repo_guideline.md         # This file
└── README.md                 # Project overview and quick start
```

## Installation and Setup

### Prerequisites
- Python 3.7+
- Required packages listed in `requirements.txt`

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd POFJSP-reproduce
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -m pytest tests/
   ```

## Key Components

### 1. Problem Instance (`ProblemInstance` class)
- **Attributes:**
  - `num_jobs`: Number of jobs
  - `num_machines`: Number of machines  
  - `num_operations_per_job`: Operations per job
  - `processing_times`: Processing time matrices
  - `predecessors_map`: Predecessor constraints
  - `successors_map`: Successor relationships

### 2. Solution Representation (`Solution` class)
- **Double encoding:** Operation sequence + Machine assignment
- **Decoding:** Insertion-based strategy respecting precedence constraints
- **Fitness:** Makespan calculation

### 3. Algorithm Components

#### Exploration Phase
- **2D Clustering Crossover:** Based on fitness and DVPC (Degree of Variance of Process Completion)
- **Effective Parallel Mutation:** Machine mutation + Operation position exchange

#### Development Phase  
- **Grade Neighborhood Search (GNS):** Hierarchical search on bottleneck jobs/machines
- **Priority Levels:**
  - G1 (High): Operations with multiple successors
  - G2 (Medium): Operations with one successor  
  - G3 (Low): Operations with no successors

## Usage Examples

### Basic Usage

```python
from src.algorithms import iaoa_gns_algorithm, ProblemInstance
from src.problem_loader import load_benchmark_instance

# Load a problem instance
problem = load_benchmark_instance('PMk01')

# Run IAOA+GNS algorithm
best_solution = iaoa_gns_algorithm(
    problem=problem,
    pop_size=90,
    max_iterations=60
)

# Display results
print(f"Best makespan: {best_solution.makespan}")
print(f"Operation sequence: {best_solution.operation_sequence}")
```

### Advanced Configuration

```python
# Custom problem instance
problem = ProblemInstance(
    num_jobs=3,
    num_machines=2,
    num_operations_per_job=[3, 2, 3],
    processing_times=processing_matrices,
    predecessors_map=predecessors,
    successors_map=successors
)

# Algorithm with custom parameters
solution = iaoa_gns_algorithm(
    problem=problem,
    pop_size=100,
    max_iterations=70
)
```

## Algorithm Parameters

### Default Parameters (based on paper)
- **Population Size:** 90 (optimal for most instances)
- **Max Iterations:** 60-70 
- **MOA Range:** [0.2, 1.0]
- **Mutation Rate:** 20% of operations
- **GNS Selection:** 10% of operations (roulette wheel)

### Parameter Tuning Guidelines
- **Small instances (≤50 operations):** pop_size=50-80, iterations=30-50
- **Medium instances (51-150 operations):** pop_size=80-100, iterations=50-70  
- **Large instances (>150 operations):** pop_size=100-120, iterations=70-100

## Benchmark Instances

### Extended Benchmark Set (PMk01-PMk10)
Based on Brandimarte's instances with added partial order constraints:

| Instance | Jobs | Machines | Total Operations | Precedence Pattern |
|----------|------|----------|------------------|-------------------|
| PMk01    | 10   | 6        | 55               | ∅,{1},{1},{1},{2,3},{4,5} |
| PMk02    | 10   | 6        | 58               | ∅,{1},{1},{1},{2,3},{4,5} |
| PMk03    | 15   | 8        | 150              | Complex parallel pattern |
| ...      | ...  | ...      | ...              | ... |

## Input Data Format

### Problem Instance File Format
```json
{
  "num_jobs": 10,
  "num_machines": 6,
  "num_operations_per_job": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
  "processing_times": [
    [[3, 1, 3, null, 2, null], ...],
    ...
  ],
  "predecessors_map": {
    "(0,1)": ["(0,0)"],
    "(0,4)": ["(0,2)", "(0,3)"],
    ...
  }
}
```

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Document functions with docstrings
- Keep functions focused and modular

### Testing
- Write unit tests for new algorithms
- Test with known benchmark instances
- Verify solution feasibility
- Performance regression testing

### Adding New Features

1. **New Algorithm Variants:**
   - Inherit from base classes
   - Implement required methods
   - Add comprehensive tests
   - Document parameter sensitivity

2. **New Problem Types:**
   - Extend `ProblemInstance` class
   - Update decoder if needed
   - Create test instances
   - Validate against literature

## Performance Benchmarks

### Expected Results (30 runs average)
| Instance | IAOA+GNS | IAOA+VNS | Literature Best |
|----------|----------|----------|-----------------|
| PMk01    | 39.62    | 45.43    | ~40            |
| PMk02    | 28.56    | 31.56    | ~28            |
| PMk09    | 307.43   | 316.65   | ~310           |

### Runtime Expectations
- PMk01-PMk05: < 2 minutes
- PMk06-PMk08: 2-5 minutes  
- PMk09-PMk10: 5-8 minutes

## Known Issues and Limitations

### Current Implementation Status
- ✅ Core IAOA+GNS algorithm implemented
- ✅ Basic decoding and fitness evaluation
- ✅ 2D clustering crossover
- ✅ Grade neighborhood search
- ⚠️ Visualization tools (in progress)
- ⚠️ Comprehensive test suite (partial)
- ❌ Dynamic scheduling extensions

### Known Bugs
- Line 56 in `algorithms.py`: `problem_instance` undefined in globals() check

### Limitations
- No support for dynamic job arrivals
- Limited to deterministic processing times
- Memory usage scales with problem size

## Contributing

### Pull Request Process
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-algorithm`
3. Implement changes with tests
4. Update documentation
5. Submit pull request with clear description

### Issue Reporting
- Use issue templates for bugs/features
- Include minimal reproduction example
- Specify Python version and OS
- Attach relevant error logs

## References

### Primary Paper
李长云, 林多, 何频捷, 谷鹏飞. 基于等级邻域策略的偏序柔性车间优化调度. 管理信息化, 2022(7): 165-172.

### Related Work
- Brandimarte, P. (1993). Routing and scheduling in a flexible job shop by tabu search.
- Abdelmaguid, T.F. (2010). Representations in genetic algorithms for the job shop scheduling problem.
- Zhang, G., et al. (2019). An effective genetic algorithm for the flexible job-shop scheduling problem.

### Algorithm Foundations
- Arithmetic Optimization Algorithm (AOA): Abualigah et al. (2021)
- Flexible Job Shop Scheduling: Brucker & Schlie (1990)
- Neighborhood Search: Hansen & Mladenović (2001)

## License

This project is released under MIT License. See LICENSE file for details.

## Contact

For questions about the implementation or paper reproduction:
- Create an issue in this repository
- Refer to the original paper for theoretical questions
- Check existing issues before posting new ones

---

**Note:** This is a research reproduction project. Results may vary from the original paper due to implementation details, random number generation, and parameter settings. 