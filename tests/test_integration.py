#!/usr/bin/env python3
"""
Integration Test for POFJSP Codebase

Tests core functionality to ensure the refactored code is working correctly.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_problem_instance():
    """Test problem instance creation and validation."""
    print("Testing Problem Instance...")
    
    try:
        import numpy as np
        from problems.problem_instance import ProblemInstance, Operation
        
        # Create a simple test problem
        problem = ProblemInstance(
            num_jobs=2,
            num_machines=2,
            num_operations_per_job=[2, 2],
            processing_times=[
                np.array([[10, 20], [15, 25]]),
                np.array([[12, 18], [22, 16]])
            ],
            predecessors_map={},
            successors_map={}
        )
        
        print(f"✓ Problem created: {problem}")
        
        # Test validation
        valid_machines = problem.get_valid_machines(Operation(0, 0))
        print(f"✓ Valid machines for operation (0,0): {valid_machines}")
        
        # Test processing time lookup
        proc_time = problem.get_processing_time(Operation(0, 0), 0)
        print(f"✓ Processing time for operation (0,0) on machine 0: {proc_time}")
        
        return True
        
    except Exception as e:
        print(f"✗ Problem Instance test failed: {e}")
        traceback.print_exc()
        return False


def test_validation_system():
    """Test input validation and error handling."""
    print("\nTesting Validation System...")
    
    try:
        from validation import validate_inputs, Validators
        from exceptions import ValidationError
        
        @validate_inputs(x=Validators.positive_int, y=Validators.positive_float)
        def test_function(x, y):
            return x * y
        
        # Test valid input
        result = test_function(5, 2.5)
        print(f"✓ Valid input test passed: {result}")
        
        # Test invalid input
        try:
            test_function(-1, 2.5)
            print("✗ Validation failed to catch invalid input")
            return False
        except ValidationError:
            print("✓ Validation correctly caught invalid input")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation System test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\nTesting Performance Monitoring...")
    
    try:
        from performance.monitor import PerformanceTracker, performance_tracker
        import time
        
        # Test performance tracker
        tracker = PerformanceTracker()
        tracker.start_tracking("test_algorithm")
        
        # Simulate some work
        time.sleep(0.1)
        tracker.record_makespan(100.0, 1)
        tracker.record_algorithm_metric("test_metric", 42)
        
        metrics = tracker.stop_tracking()
        print(f"✓ Performance tracking completed")
        print(f"  Execution time: {metrics.execution_time:.3f}s")
        print(f"  Makespan progression: {metrics.makespan_progression}")
        
        # Test context manager
        with performance_tracker("test_operation") as ctx_tracker:
            time.sleep(0.05)
            ctx_tracker.record_makespan(95.0, 1)
        
        print("✓ Context manager test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance Monitoring test failed: {e}")
        traceback.print_exc()
        return False


def test_training_config():
    """Test training configuration system."""
    print("\nTesting Training Configuration...")
    
    try:
        from training.config import TrainingConfig, get_training_config, config_manager
        
        # Test basic configuration
        config = get_training_config('debug')
        print(f"✓ Debug config loaded: {config.total_timesteps} timesteps")
        
        # Test configuration validation
        config._validate_config()
        print("✓ Configuration validation passed")
        
        # Test curriculum stages
        stages = config_manager.create_curriculum_stages(config)
        print(f"✓ Created {len(stages)} curriculum stages")
        
        for i, stage in enumerate(stages):
            print(f"  Stage {i+1}: {stage.name} - {stage.job_range} jobs")
        
        return True
        
    except Exception as e:
        print(f"✗ Training Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_curriculum_learning():
    """Test curriculum learning system."""
    print("\nTesting Curriculum Learning...")
    
    try:
        from training.curriculum import CurriculumManager, ProblemInstanceGenerator
        from training.config import get_training_config, config_manager
        
        # Test curriculum manager
        config = get_training_config('debug')
        stages = config_manager.create_curriculum_stages(config)
        curriculum = CurriculumManager(config, stages)
        
        print(f"✓ Curriculum manager created with {len(stages)} stages")
        print(f"  Current stage: {curriculum.get_current_stage().name}")
        
        # Test problem generation
        generator = ProblemInstanceGenerator(seed=42)
        instances = generator.generate_curriculum_instances(stages[0], 2)
        
        print(f"✓ Generated {len(instances)} problem instances")
        for i, instance in enumerate(instances):
            print(f"  Instance {i+1}: {instance.num_jobs} jobs, {instance.num_machines} machines")
        
        # Test curriculum progression
        curriculum.record_performance(0.5, 100.0, 50.0)
        should_advance, reason = curriculum.should_advance_stage()
        print(f"✓ Curriculum progression test: should_advance={should_advance} ({reason})")
        
        return True
        
    except Exception as e:
        print(f"✗ Curriculum Learning test failed: {e}")
        traceback.print_exc()
        return False


def test_memory_management():
    """Test memory management and file cleanup."""
    print("\nTesting Memory Management...")
    
    try:
        # Test large file cleanup script
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.cleanup_large_files import get_file_size_mb, find_large_files
        
        # Test file size calculation
        test_file = Path(__file__)
        if test_file.exists():
            size_mb = get_file_size_mb(test_file)
            print(f"✓ File size calculation: {test_file.name} = {size_mb:.3f}MB")
        
        # Test finding large files in current directory
        large_files = find_large_files(Path('.'), min_size_mb=0.1)  # Very small threshold for testing
        print(f"✓ Found {len(large_files)} files larger than 0.1MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory Management test failed: {e}")
        traceback.print_exc()
        return False


def test_algorithm_basic():
    """Test basic algorithm functionality (without full dependencies)."""
    print("\nTesting Basic Algorithm Components...")
    
    try:
        from algorithms.decoder import decode_solution, SimpleOperation
        from problems.problem_instance import ProblemInstance, Solution, Operation
        import numpy as np
        
        # Create a minimal problem
        problem = ProblemInstance(
            num_jobs=2,
            num_machines=2,
            num_operations_per_job=[1, 1],
            processing_times=[
                np.array([[10, 20]]),
                np.array([[15, 25]])
            ],
            predecessors_map={},
            successors_map={}
        )
        
        # Create a simple solution
        operations = [Operation(0, 0), Operation(1, 0)]
        machines = [0, 1]
        solution = Solution(operations, machines)
        
        print(f"✓ Created problem and solution")
        
        # Test solution validation
        is_valid = solution.validate(problem)
        print(f"✓ Solution validation: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"✗ Algorithm Components test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("POFJSP Codebase Integration Tests")
    print("=" * 60)
    
    tests = [
        test_problem_instance,
        test_validation_system,
        test_performance_monitoring,
        test_training_config,
        test_curriculum_learning,
        test_memory_management,
        test_algorithm_basic
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The codebase is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)