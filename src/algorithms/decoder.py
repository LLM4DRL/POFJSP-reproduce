import numpy as np
import heapq

def decode_solution(solution, problem, verbose=False):
    """
    Decodes a solution to calculate makespan and schedule details.
    This uses an insertion-based strategy respecting precedence constraints.
    """
    # Create a simple Operation class for consistent handling
    class SimpleOperation:
        def __init__(self, job_idx, op_idx_in_job):
            self.job_idx = job_idx
            self.op_idx_in_job = op_idx_in_job
        
        def __hash__(self):
            return hash((self.job_idx, self.op_idx_in_job))
        
        def __eq__(self, other):
            return (self.job_idx, self.op_idx_in_job) == (other.job_idx, other.op_idx_in_job)
        
        def __repr__(self):
            return f"Operation({self.job_idx}, {self.op_idx_in_job})"
    
    # Convert operation_sequence to SimpleOperation objects
    operation_objects = []
    for op in solution.operation_sequence:
        if hasattr(op, 'job_idx') and hasattr(op, 'op_idx_in_job'):
            operation_objects.append(SimpleOperation(op.job_idx, op.op_idx_in_job))
        else:
            # Handle tuple format (job_id, op_id)
            operation_objects.append(SimpleOperation(op[0], op[1]))
    
    # schedule_details: {SimpleOperation: {'start_time', 'end_time', 'machine'}}
    schedule_details = {}
    # machine_schedules: list of lists, machine_schedules[m] = sorted list of (start_time, end_time)
    machine_schedules = [[] for _ in range(problem.num_machines)]

    # Keep track of completion times of operations for precedence constraints
    operation_completion_times = {}
    scheduled_operations = set()

    # Create a graph of operation dependencies
    in_degree = {op: 0 for op in operation_objects}
    op_to_predecessors = {}
    
    for op in operation_objects:
        op_key = (op.job_idx, op.op_idx_in_job)
        if op_key in problem.predecessors_map:
            op_to_predecessors[op] = problem.predecessors_map[op_key]
            in_degree[op] = len(problem.predecessors_map[op_key])
        else:
            op_to_predecessors[op] = []
    
    # Find operations with no predecessors
    ready_operations = [op for op in operation_objects if in_degree[op] == 0]
    
    # Process operations in topological order
    processed_ops = 0
    
    while ready_operations and processed_ops < len(operation_objects):
        current_op = ready_operations.pop(0)
        
        if current_op in scheduled_operations:
            continue
        
        # Find its index in the operation sequence
        op_idx = operation_objects.index(current_op)
        assigned_machine = solution.machine_assignment[op_idx]
        proc_time = problem.processing_times[current_op.job_idx][current_op.op_idx_in_job, assigned_machine]

        if proc_time == np.inf:
            if verbose:
                print(f"ERROR: Invalid processing time for op={current_op}, machine={assigned_machine}")
            solution.makespan = float('inf')
            solution.schedule_details = {}
            solution.machine_schedules = [[] for _ in range(problem.num_machines)]
            return float('inf'), {}, [[] for _ in range(problem.num_machines)]

        # Determine earliest start time based on predecessors
        earliest_start_due_to_predecessors = 0
        if current_op in op_to_predecessors:
            for pred_op_key in op_to_predecessors[current_op]:
                pred_job, pred_op = pred_op_key
                # Find the corresponding SimpleOperation
                pred_op_obj = None
                for op_obj in operation_objects:
                    if op_obj.job_idx == pred_job and op_obj.op_idx_in_job == pred_op:
                        pred_op_obj = op_obj
                        break
                if pred_op_obj and pred_op_obj in operation_completion_times:
                    earliest_start_due_to_predecessors = max(
                        earliest_start_due_to_predecessors,
                        operation_completion_times[pred_op_obj]
                    )
        
        # Sort machine schedule by start times to find gaps
        machine_schedules[assigned_machine].sort()

        # Try to insert in existing gaps
        last_finish_time_on_machine = 0
        inserted = False
        for j in range(len(machine_schedules[assigned_machine])):
            gap_start = last_finish_time_on_machine
            gap_end = machine_schedules[assigned_machine][j][0]
            
            possible_start_in_gap = max(earliest_start_due_to_predecessors, gap_start)
            if possible_start_in_gap + proc_time <= gap_end:
                op_start_time = possible_start_in_gap
                inserted = True
                break
            last_finish_time_on_machine = machine_schedules[assigned_machine][j][1]

        if not inserted:
            # If no suitable gap, schedule after the last operation on the machine
            op_start_time = max(earliest_start_due_to_predecessors, last_finish_time_on_machine)

        op_end_time = op_start_time + proc_time
        
        # Update schedules
        machine_schedules[assigned_machine].append((op_start_time, op_end_time))
        machine_schedules[assigned_machine].sort()

        schedule_details[current_op] = {
            'start_time': op_start_time,
            'end_time': op_end_time,
            'machine': assigned_machine
        }
        operation_completion_times[current_op] = op_end_time
        scheduled_operations.add(current_op)
        
        # Update successors' in_degree and add to ready queue
        op_key = (current_op.job_idx, current_op.op_idx_in_job)
        if op_key in problem.successors_map:
            for succ_op_key in problem.successors_map[op_key]:
                succ_job, succ_op = succ_op_key
                # Find the corresponding SimpleOperation
                succ_op_obj = None
                for op_obj in operation_objects:
                    if op_obj.job_idx == succ_job and op_obj.op_idx_in_job == succ_op:
                        succ_op_obj = op_obj
                        break
                if succ_op_obj and succ_op_obj in in_degree:
                    in_degree[succ_op_obj] -= 1
                    if in_degree[succ_op_obj] == 0:
                        ready_operations.append(succ_op_obj)
        
        processed_ops += 1
    
    # Check if all operations were processed
    if processed_ops < len(operation_objects):
        if verbose:
            print(f"WARNING: Not all operations were processed.")
    
    makespan = 0
    if operation_completion_times:
        makespan = max(operation_completion_times.values())
    
    # For GNS, it's useful to have machine schedules also store op info
    final_machine_schedules_detailed = [[] for _ in range(problem.num_machines)]
    for op, details in schedule_details.items():
        final_machine_schedules_detailed[details['machine']].append(
            (details['start_time'], details['end_time'], op)
        )
    for m_idx in range(problem.num_machines):
        final_machine_schedules_detailed[m_idx].sort()

    solution.makespan = makespan
    solution.schedule_details = schedule_details
    solution.machine_schedules = final_machine_schedules_detailed
    return makespan, schedule_details, final_machine_schedules_detailed