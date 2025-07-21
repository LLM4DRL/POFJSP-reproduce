# 🧬 POFJSP Algorithm Implementation Guide

This document provides detailed technical information about the algorithm implementations in this repository.

## 🏆 **IAOA+GNS Algorithm**

### **Overview**
The **Improved Arithmetic Optimization Algorithm with Grade Neighborhood Search (IAOA+GNS)** is the best-performing algorithm for POFJSP, combining:
- Arithmetic Optimization Algorithm (AOA) for global search
- Grade Neighborhood Search (GNS) for local intensification
- Adaptive parameter control and population management

### **Key Features**
- **Hierarchical Search**: Two-level optimization (AOA + GNS)
- **Adaptive MOA**: Math Operator Accelerated parameter adapts during search
- **Bottleneck Analysis**: Identifies critical machines and jobs for targeted improvement
- **2D Clustering Crossover**: Preserves good solution structures

### **Implementation Details**
```python
# Location: src/algorithms/iaoa_gns.py
class IAOAGNSAlgorithm:
    def __init__(self, pop_size=80, max_iterations=60):
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.moa_a = 2.0  # MOA parameter
        self.moa_u = 0.4  # MOA parameter
    
    def solve(self, problem, verbose=False):
        # Initialize population
        # Run IAOA iterations with GNS
        # Return best solution
```

### **Parameter Guidelines**
- **Population Size**: 80 (optimal for most problems)
- **Max Iterations**: 60 (balance between quality and time)
- **MOA Parameters**: Default values work well across problem sizes

---

## 🤖 **Deep Reinforcement Learning**

### **Architecture**
The RL system uses **Graph Neural Networks + Proximal Policy Optimization** with:
- **Graph Representation**: Jobs and machines as nodes, operations as edges
- **Hierarchical Actions**: First select job, then select machine
- **Dynamic Action Masking**: Only valid actions are available

### **Network Components**
```python
# Location: src/rl/models/
class GraphCNN(nn.Module):
    # Graph neural network for state representation
    
class JobActor(nn.Module):
    # Selects which job to process next
    
class MachineActor(nn.Module):  
    # Selects machine for chosen job

class Critic(nn.Module):
    # Value function estimation
```

### **Training Configuration**
```python
# Default training parameters
TrainingConfig(
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    batch_size=256,
    curriculum_stages=3,  # Progressive difficulty
    max_jobs=100,         # Scale up to 100x100
    max_machines=100
)
```

### **Curriculum Learning**
1. **Stage 1**: Small problems (8×6 to 15×10)
2. **Stage 2**: Medium problems (15×10 to 30×20)  
3. **Stage 3**: Large problems (30×20 to 50×35)

---

## 🧬 **Traditional Algorithms**

### **Genetic Algorithm**

**Representation**: Operations sequence + machine assignments
```python
# Chromosome: [(job_id, op_id, machine_id), ...]
chromosome = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
```

**Key Operators**:
- **Selection**: Tournament selection (size 3)
- **Crossover**: Order-based crossover respecting precedence
- **Mutation**: Sequence swap + machine reassignment
- **Elitism**: Keep 10% best solutions

### **Simulated Annealing**

**Neighborhood Operators**:
1. **Sequence Swap**: Swap two operations (precedence-aware)
2. **Machine Change**: Reassign operation to different machine
3. **Sequence Shift**: Move operation to new position

**Cooling Schedule**:
```python
# Exponential cooling with reheating
temperature = initial_temp * (cooling_rate ** iteration)
# Reheat when stagnated for too long
if stagnation_count > max_stagnation:
    temperature *= reheat_factor
```

### **Tabu Search**

**Tabu List Management**:
- **Size**: 50 moves (adjustable)
- **Move Types**: Operation swaps, machine changes
- **Aspiration**: Accept tabu moves if they improve best solution

**Diversification Strategy**:
- **Frequency Matrix**: Track move frequency for penalty
- **Intensification**: Restart from best solution when stagnated
- **Long-term Memory**: Bias against frequently used assignments

---

## 🔧 **Implementation Architecture**

### **Unified Interface**
All algorithms follow the same pattern:
```python
def main():
    # Create test problem
    # Run algorithm with small parameters
    # Display results
    
if __name__ == "__main__":
    main()
```

### **Problem Instance Format**
```python
class ProblemInstance:
    num_jobs: int
    num_machines: int
    num_operations_per_job: List[int]
    processing_times: List[np.ndarray]  # [job][op, machine]
    predecessors_map: Dict[str, List[str]]  # "(job,op)": ["(j,o)", ...]
    successors_map: Dict[str, List[str]]
```

### **Solution Representation**
```python
class Solution:
    operation_sequence: List[Tuple[int, int]]  # [(job, op), ...]
    machine_assignment: List[int]              # [machine_id, ...]
    makespan: float                            # Objective value
```

---

## 📊 **Performance Characteristics**

### **Scalability Analysis**

| Problem Size | IAOA+GNS | RL Training | GA | SA | Tabu |
|-------------|----------|-------------|----|----|------|
| **Small (10×10)** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡ |
| **Medium (30×50)** | ⚡⚡⚡⚡⚡ | ⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡ |
| **Large (100×100)** | ⚡⚡⚡⚡ | ⚡ | ⚡⚡ | ⚡⚡⚡ | ⚡⚡ |

### **Memory Requirements**
- **IAOA+GNS**: O(pop_size × operations) = ~10MB for 100×100
- **RL**: O(network_params + replay_buffer) = ~500MB during training
- **Traditional**: O(operations²) for precedence = ~100MB for 100×100

### **Convergence Behavior**
- **IAOA+GNS**: Rapid early improvement, then local refinement
- **RL**: Slow initial learning, then rapid improvement with curriculum
- **GA**: Steady improvement with occasional jumps
- **SA**: Initial random walk, then hill climbing with occasional escapes
- **Tabu**: Aggressive exploration with memory-guided intensification

---

## 🚀 **Usage Examples**

### **Quick Testing**
```bash
# Test all algorithms on same problem
python src/algorithms/iaoa_gns.py
python src/algorithms/genetic_algorithm.py
python src/algorithms/simulated_annealing.py
python src/algorithms/tabu_search.py

# Full comparison
python main.py --compare-algorithms --dataset dummy --verbose
```

### **Production Usage**
```bash
# Best performance for production
python main.py --algorithm iaoa_gns

# Research and adaptation
python main.py --algorithm rl --train --fast-mode

# Quick approximation
python main.py --algorithm sa
```

### **Parameter Tuning**
```bash
# Custom parameters (not yet implemented - use direct script execution)
python scripts/algorithms/run_iaoa_gns.py --pop-size 100 --max-iter 100
```

---

## 🔬 **Research Extensions**

### **Algorithm Hybridization**
- **IAOA+RL**: Use RL for initial population of IAOA
- **Multi-objective**: Extend to minimize makespan + energy + cost
- **Online Learning**: Adapt to changing job arrivals

### **Problem Variants**
- **Stochastic Processing**: Uncertain processing times
- **Resource Constraints**: Limited tools, operators
- **Multi-factory**: Distributed manufacturing

### **Performance Optimization**
- **Parallelization**: Multi-threaded population evaluation
- **GPU Acceleration**: CUDA implementations for large problems
- **Approximation**: Fast heuristics for real-time scheduling

---

This guide provides the technical foundation for understanding and extending the algorithm implementations. For usage examples, see the main README.