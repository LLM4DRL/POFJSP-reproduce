# 📚 POFJSP Technical Documentation

This directory contains essential technical documentation for the POFJSP implementation.

## 📋 **Documentation Index**

### **Core References**
- [`problem_format.md`](problem_format.md) - POFJSP problem instance format specification
- [`CLAUDE.md`](CLAUDE.md) - Development guidance for AI assistants

### **Algorithm Documentation**
- [`algorithm_guide.md`](algorithm_guide.md) - Detailed algorithm implementation guides
- [`rl_architecture.md`](rl_architecture.md) - Deep RL system architecture

### **Research Papers**
- [`Optimal scheduling of partially ordered flexible job shop based on hierarchical neighborhood strategy.pdf`]() - Original IAOA+GNS paper

---

## 🎯 **Quick References**

### **Problem Format**
See [`problem_format.md`](problem_format.md) for complete specification of:
- Processing time matrices
- Precedence constraint encoding
- JSON format for file storage
- Validation and feasibility checks

### **Algorithm Implementation**
Each algorithm is implemented in `src/algorithms/` with:
- Standalone main function for testing
- Unified interface through control center
- Comprehensive parameter configuration

### **Control Center Usage**
The unified control center provides all functionality:
```bash
python main.py --algorithm {iaoa_gns,ga,sa,tabu,rl}  # Run algorithms
python main.py --format                              # Code formatting
python main.py --health-check                        # Repository analysis
python main.py --status                              # System status
```

---

## 🔬 **Research Context**

This implementation focuses on the **Partially Ordered Flexible Job Shop Problem (POFJSP)**, which extends traditional job shop scheduling by:

1. **Flexibility**: Operations can be processed on multiple machines with different processing times
2. **Partial Ordering**: Jobs have precedence constraints that form directed acyclic graphs (not just sequences)
3. **Multi-objective**: Optimizing makespan while respecting all constraints

### **Key Algorithms Implemented**
- **IAOA+GNS**: Hierarchical metaheuristic with grade neighborhood search
- **Deep RL**: Graph Neural Network with Proximal Policy Optimization
- **Traditional**: Genetic Algorithm, Simulated Annealing, Tabu Search

---

## 📊 **Performance Benchmarks**

Current performance on 30×50 problems (135 operations):

| Algorithm | Makespan | Time (s) | Success Rate |
|-----------|----------|----------|--------------|
| IAOA+GNS  | **11.0** | **10.6** | 100% |
| RL (GNN)  | 12.9     | 51.5     | 85% |
| GA        | 13.2     | 25.3     | 95% |
| SA        | 14.1     | 8.7      | 90% |
| Tabu      | 14.5     | 18.9     | 88% |

---

## 🛠️ **Development Notes**

### **Code Organization**
- All algorithms have `main()` functions for standalone testing
- Control center provides unified interface
- Repository health monitoring integrated
- Automated code formatting and quality checks

### **Testing Strategy**
```bash
# Individual algorithm tests
python src/algorithms/iaoa_gns.py
python src/algorithms/genetic_algorithm.py

# Integration tests  
python main.py --compare-algorithms --dataset dummy

# Quality checks
python main.py --health-check
```

### **Adding New Algorithms**
1. Implement in `src/algorithms/new_algorithm.py` with `main()` function
2. Add to control center algorithm registry
3. Create unit tests
4. Update documentation

---

This documentation is kept concise to avoid duplication with the main README. For usage examples and getting started, see the main [`README.md`](../README.md) file.