import numpy as np
from collections import namedtuple

# --- Data Structures ---
Operation = namedtuple('Operation', ['job_idx', 'op_idx_in_job'])

class ProblemInstance:
    def __init__(self, num_jobs, num_machines, num_operations_per_job, processing_times, predecessors_map, successors_map):
        """
        Initializes the problem instance.
        Args:
            num_jobs (int): Total number of jobs.
            num_machines (int): Total number of machines.
            num_operations_per_job (list): List where num_operations_per_job[j] is the number of operations for job j.
            processing_times (list): A list of 2D NumPy arrays.
                                     processing_times[j][o, m] is the time for op o of job j on machine m.
                                     Use np.inf if op cannot be processed on machine.
            predecessors_map (dict): A dictionary where predecessors_map[Operation(j,o)] is a set of Operation tuples.
            successors_map (dict): A dictionary where successors_map[Operation(j,o)] is a set of Operation tuples.
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations_per_job = num_operations_per_job
        self.processing_times = processing_times # List of np.arrays
        self.predecessors_map = predecessors_map # {(job_idx, op_idx_in_job): set of (job_idx, op_idx_in_job)}
        self.successors_map = successors_map

        self.total_operations = sum(num_operations_per_job)
        self.all_operations = []
        for j in range(num_jobs):
            for o in range(num_operations_per_job[j]):
                self.all_operations.append(Operation(j, o))

    @classmethod
    def from_json(cls, json_path):
        """Load problem instance from JSON file."""
        import json
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both legacy format (keys at root) and new format (keys under 'problem')
        if 'problem' in data:
            problem_data = data['problem']
        else:
            problem_data = data
            
        num_jobs = problem_data['num_jobs']
        num_machines = problem_data['num_machines']
        num_operations_per_job = problem_data['num_operations_per_job']
        
        # Convert processing times to numpy arrays
        processing_times = []
        for job_proc_times in problem_data['processing_times']:
            arr = np.array(job_proc_times)
            arr[arr == float('inf')] = np.inf
            processing_times.append(arr)
        
        # Convert predecessors map
        predecessors_map = {}
        successors_map = {}
        
        for op_key_str, pred_list in problem_data['predecessors_map'].items():
            # Parse operation key
            job_idx, op_idx = map(int, op_key_str.strip('()').split(','))
            op = Operation(job_idx, op_idx)
            
            # Parse predecessors
            predecessors = set()
            for pred_str in pred_list:
                pred_job, pred_op = map(int, pred_str.strip('()').split(','))
                predecessors.add(Operation(pred_job, pred_op))
            
            predecessors_map[op] = predecessors
        
        # Build successors map from predecessors map
        successors_map = {op: set() for op in predecessors_map}
        for op, preds in predecessors_map.items():
            for pred in preds:
                if pred not in successors_map:
                    successors_map[pred] = set()
                successors_map[pred].add(op)
        
        return cls(num_jobs, num_machines, num_operations_per_job, 
                  processing_times, predecessors_map, successors_map)
    
    @classmethod
    def from_fjsp_file(cls, file_path):
        """Load problem instance from FJSP format file."""
        # Placeholder for FJSP format support
        raise NotImplementedError("FJSP format loading not implemented yet")

class Solution:
    def __init__(self, operation_sequence, machine_assignment):
        """
        Represents a solution (chromosome).
        Args:
            operation_sequence (list): A list of Operation tuples.
            machine_assignment (list): A list of machine indices corresponding to operation_sequence.
        """
        self.operation_sequence = operation_sequence # List of Operation(job_idx, op_idx_in_job)
        self.machine_assignment = machine_assignment # List of machine_idx
        self.makespan = float('inf')
        self.schedule_details = {} # Populated by decoder: {Operation: {'start_time', 'end_time', 'machine'}}
        self.machine_schedules = [] # Will be properly initialized in decode_solution

    def __lt__(self, other): # For sorting solutions by makespan
        return self.makespan < other.makespan