# POFJSP Problem Instance Format Specification

## Overview

This document specifies the data format for Partially Ordered Flexible Job Shop Scheduling Problem (POFJSP) instances used in this repository.

## Problem Components

### 1. Basic Parameters

```python
num_jobs: int           # Number of jobs (n)
num_machines: int       # Number of machines (m) 
num_operations_per_job: List[int]  # Operations per job [p1, p2, ..., pn]
```

### 2. Processing Times

**Format**: List of 2D NumPy arrays
```python
processing_times: List[np.ndarray]
# processing_times[j][o, m] = processing time for operation o of job j on machine m
# Use np.inf if operation cannot be processed on the machine
```

**Example**:
```python
# Job 0: 2 operations, Job 1: 2 operations, 2 machines total
processing_times = [
    np.array([
        [3, 5],      # Job 0, Operation 0: M0=3, M1=5
        [6, np.inf]  # Job 0, Operation 1: M0=6, M1=cannot process
    ]),
    np.array([
        [4, 2],      # Job 1, Operation 0: M0=4, M1=2  
        [np.inf, 7]  # Job 1, Operation 1: M0=cannot, M1=7
    ])
]
```

### 3. Precedence Constraints

**Predecessors Map**:
```python
predecessors_map: Dict[Operation, Set[Operation]]
# predecessors_map[op] = set of operations that must complete before op can start
```

**Successors Map**:
```python
successors_map: Dict[Operation, Set[Operation]]  
# successors_map[op] = set of operations that can only start after op completes
```

**Operation Representation**:
```python
from collections import namedtuple
Operation = namedtuple('Operation', ['job_idx', 'op_idx_in_job'])
# Operation(0, 1) represents the 2nd operation of the 1st job
```

## Complete Example

### Simple 2-Job, 2-Machine Problem

```python
import numpy as np
from collections import namedtuple
from src.algorithms import ProblemInstance

Operation = namedtuple('Operation', ['job_idx', 'op_idx_in_job'])

# Basic parameters
num_jobs = 2
num_machines = 2  
num_operations_per_job = [2, 2]  # Each job has 2 operations

# Processing times
processing_times = [
    # Job 0
    np.array([
        [3, 5],      # Op 0: M0=3, M1=5
        [6, np.inf]  # Op 1: M0=6, M1=cannot
    ]),
    # Job 1  
    np.array([
        [4, 2],      # Op 0: M0=4, M1=2
        [np.inf, 7]  # Op 1: M0=cannot, M1=7
    ])
]

# Precedence constraints
predecessors_map = {
    Operation(0, 0): set(),                    # First op of job 0
    Operation(0, 1): {Operation(0, 0)},        # Second op needs first op
    Operation(1, 0): set(),                    # First op of job 1
    Operation(1, 1): {Operation(1, 0)}         # Second op needs first op
}

successors_map = {
    Operation(0, 0): {Operation(0, 1)},        # First op precedes second
    Operation(0, 1): set(),                    # Last op of job 0
    Operation(1, 0): {Operation(1, 1)},        # First op precedes second  
    Operation(1, 1): set()                     # Last op of job 1
}

# Create problem instance
problem = ProblemInstance(
    num_jobs=num_jobs,
    num_machines=num_machines,
    num_operations_per_job=num_operations_per_job,
    processing_times=processing_times,
    predecessors_map=predecessors_map,
    successors_map=successors_map
)
```

## Complex Precedence Patterns

### Pattern 1: Parallel Operations After Common Predecessor

```
Job workflow:
Op0 → Op1, Op2 (parallel) → Op3

Precedence encoding:
- Op0: no predecessors  
- Op1: {Op0}
- Op2: {Op0}  
- Op3: {Op1, Op2}
```

```python
# Example for job with parallel operations
predecessors_map = {
    Operation(0, 0): set(),                           # Start operation
    Operation(0, 1): {Operation(0, 0)},               # Parallel branch 1
    Operation(0, 2): {Operation(0, 0)},               # Parallel branch 2
    Operation(0, 3): {Operation(0, 1), Operation(0, 2)}  # Join operation
}
```

### Pattern 2: Complex Assembly Structure

```
Job workflow:
     Op1
    /    \
Op0       Op4
    \    /
     Op2, Op3

Precedence encoding:
- Op0: no predecessors
- Op1: {Op0}
- Op2: {Op0}  
- Op3: {Op0}
- Op4: {Op1, Op2, Op3}
```

## JSON File Format

For storing problem instances in files:

```json
{
  "metadata": {
    "name": "PMk01",
    "description": "Extended benchmark based on Mk01",
    "source": "Brandimarte extended for POFJSP"
  },
  "problem": {
    "num_jobs": 10,
    "num_machines": 6,
    "num_operations_per_job": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "processing_times": [
      [
        [3, 1, 3, null, 2, null],
        [2, null, 1, 4, null, null],
        [null, null, 3, null, 3, 1],
        [2, 1, null, null, null, 1],
        [1, 3, null, 3, null, null],
        [null, null, 1, 2, 6, null]
      ]
    ],
    "predecessors_map": {
      "(0,0)": [],
      "(0,1)": ["(0,0)"],
      "(0,2)": ["(0,0)"],
      "(0,3)": ["(0,0)"],
      "(0,4)": ["(0,2)", "(0,3)"],
      "(0,5)": ["(0,4)", "(0,1)"]
    }
  }
}
```

**Notes**:
- Use `null` for infeasible machine assignments (converted to `np.inf`)
- Operation keys use format `"(job_idx,op_idx_in_job)"`
- Predecessor lists contain operation keys

## Validation Rules

### 1. Consistency Checks

```python
def validate_problem_instance(problem):
    """Validate problem instance for consistency."""
    
    # Check processing times dimensions
    assert len(problem.processing_times) == problem.num_jobs
    for j, pt_matrix in enumerate(problem.processing_times):
        assert pt_matrix.shape == (problem.num_operations_per_job[j], problem.num_machines)
    
    # Check operation consistency
    all_operations = set()
    for j in range(problem.num_jobs):
        for o in range(problem.num_operations_per_job[j]):
            all_operations.add(Operation(j, o))
    
    # Validate precedence maps
    assert set(problem.predecessors_map.keys()) == all_operations
    assert set(problem.successors_map.keys()) == all_operations
    
    # Check precedence consistency
    for op, predecessors in problem.predecessors_map.items():
        for pred_op in predecessors:
            assert op in problem.successors_map[pred_op]
```

### 2. Feasibility Checks

```python
def check_feasibility(problem):
    """Check if problem instance is feasible."""
    
    # Each operation must be processable on at least one machine
    for j in range(problem.num_jobs):
        for o in range(problem.num_operations_per_job[j]):
            feasible_machines = np.sum(~np.isinf(problem.processing_times[j][o, :]))
            assert feasible_machines > 0, f"Operation ({j},{o}) has no feasible machines"
    
    # Check for cycles in precedence graph
    assert is_dag(problem), "Precedence constraints contain cycles"

def is_dag(problem):
    """Check if precedence graph is a directed acyclic graph."""
    # Implementation of topological sort or DFS cycle detection
    # Returns True if DAG, False if cycles exist
    pass
```

## Extended Benchmark Instances

### PMk01-PMk10 Precedence Patterns

| Instance | Precedence Pattern Description |
|----------|-------------------------------|
| PMk01-02 | `∅,{1},{1},{1},{2,3},{4,5}` - Simple parallel after split |
| PMk03    | `∅,{1},{1},{1},{2},{3},{4},{5},{7},{6,8,9}` - Complex assembly |
| PMk04-05 | `∅,{1},{1},{1},{2,3},{4},{5},{6,7},{8}` - Multi-stage pipeline |
| PMk06    | `∅,{1},{1},{1},{2},{3},{4},{5},{7},{8},{6,9,10},{11},{12},{13},{14}` - Long assembly line |
| PMk07    | `∅,{1},{1},{1},{2,3,4}` - High-width parallel section |
| PMk08-10 | `∅,{1},{1},{1},{2},{3},{4},{5},{7},{8},{6,9,10},{11},{12},{13}` - Extended pipeline |

### Loading Benchmark Instances

```python
from src.problem_loader import load_benchmark_instance

# Load predefined benchmark
problem = load_benchmark_instance('PMk01')

# Load from JSON file
problem = load_problem_from_json('data/instances/custom_problem.json')
```

## Common Patterns in Garment Manufacturing

### 1. Fabric Preparation → Parallel Sewing → Assembly
```
Cut → (Collar, Sleeves, Body) → Assembly → Finishing
```

### 2. Multi-Component Assembly  
```
Component1 → \
Component2 →  Assembly → Quality Check → Packaging
Component3 → /
```

### 3. Size Variation Processing
```
Cutting → Size-specific operations → Common finishing
```

These patterns can be encoded using the precedence constraint format described above.

## Performance Considerations

- **Memory**: O(n × max_operations × m) for processing times
- **Precedence Storage**: O(total_operations²) worst case  
- **Validation**: O(total_operations + edges) for DAG check

For large instances (>1000 operations), consider sparse representations for precedence constraints. 