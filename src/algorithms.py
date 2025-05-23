import numpy as np
import random
from sklearn.cluster import KMeans
from collections import defaultdict, namedtuple
import copy
import heapq # For managing machine idle times efficiently

# --- Configuration & Constants ---
# These would typically be part of your problem instance or algorithm parameters
# MOA parameters from paper
MOA_MIN = 0.2
MOA_MAX = 1.0

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

# --- 1. Decoding and Fitness Evaluation ---
def decode_solution(solution, problem):
    """
    Decodes a solution to calculate makespan and schedule details.
    This uses an insertion-based strategy respecting precedence constraints.
    """
    # schedule_details: {Operation(j,o): {'start_time', 'end_time', 'machine'}}
    schedule_details = {}
    # machine_schedules: list of lists, machine_schedules[m] = sorted list of (start_time, end_time)
    machine_schedules = [[] for _ in range(problem.num_machines)] # Stores (start_time, end_time) tuples

    # Keep track of completion times of operations for precedence constraints
    operation_completion_times = {} # {Operation(j,o): end_time}

    for i, op in enumerate(solution.operation_sequence):
        assigned_machine = solution.machine_assignment[i]
        proc_time = problem.processing_times[op.job_idx][op.op_idx_in_job, assigned_machine]

        if proc_time == np.inf: # Should not happen with valid machine assignment
            solution.makespan = float('inf')
            solution.schedule_details = {}
            solution.machine_schedules = [[] for _ in range(problem.num_machines)]
            return float('inf'), {}, [[] for _ in range(problem.num_machines)]


        # 1. Determine earliest start time based on predecessors
        earliest_start_due_to_predecessors = 0
        if op in problem.predecessors_map:
            for pred_op in problem.predecessors_map[op]:
                if pred_op not in operation_completion_times:
                    # This indicates an issue with the operation sequence or a bug
                    # For robustness, assume this means the predecessor isn't scheduled yet,
                    # which shouldn't happen if OS respects SOME topological sort.
                    # Or, it means the problem is ill-defined / OS is invalid.
                    # For now, let's assume valid OS are generated.
                    # If a predecessor is not found, it implies it hasn't been scheduled,
                    # which might be an error in sequence generation or a very complex scenario.
                    # This part needs careful handling based on how OS is generated.
                    # If OS is just a permutation, this check is vital.
                    pass # This implies pred_op must be in operation_completion_times
                else:
                     earliest_start_due_to_predecessors = max(
                        earliest_start_due_to_predecessors,
                        operation_completion_times[pred_op]
                    )
        
        # 2. Find an available time slot on the assigned machine
        # machine_schedules[assigned_machine] is a sorted list of (start, end) tuples
        
        # Strategy: Iterate through gaps or append at the end
        op_start_time = -1
        
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
        heapq.heappush(machine_schedules[assigned_machine], (op_start_time, op_end_time)) # Keep sorted by start time
        # For actual insertion into a list and keeping it sorted:
        # bisect.insort(machine_schedules[assigned_machine], (op_start_time, op_end_time))

        schedule_details[op] = {
            'start_time': op_start_time,
            'end_time': op_end_time,
            'machine': assigned_machine
        }
        operation_completion_times[op] = op_end_time

    makespan = 0
    if operation_completion_times:
        makespan = max(operation_completion_times.values())
    
    # For GNS, it's useful to have machine schedules also store op info
    final_machine_schedules_detailed = [[] for _ in range(problem.num_machines)]
    for op, details in schedule_details.items():
        final_machine_schedules_detailed[details['machine']].append(
            (details['start_time'], details['end_time'], op.job_idx, op.op_idx_in_job)
        )
    for m_idx in range(problem.num_machines):
        final_machine_schedules_detailed[m_idx].sort()

    solution.makespan = makespan
    solution.schedule_details = schedule_details
    solution.machine_schedules = final_machine_schedules_detailed # Store detailed for GNS
    return makespan, schedule_details, final_machine_schedules_detailed


# --- 2. Solution Representation and Initialization ---
def get_topological_sort_operations(problem):
    """Generates one valid topological sort of all operations."""
    # This is a simplified approach; real POFJSP might need more sophisticated handling
    # if the OS part of the chromosome is not already guaranteed to be topologically valid.
    # For now, we assume problem.all_operations can be processed sequentially if
    # the decoder handles precedence. A better way is to generate a valid sequence.
    
    # Kahn's algorithm for topological sort
    in_degree = {op: 0 for op in problem.all_operations}
    adj = {op: [] for op in problem.all_operations}

    for op, preds in problem.predecessors_map.items():
        in_degree[op] = len(preds)
        for pred_op in preds:
            if pred_op in adj: # Ensure pred_op is a valid operation
                 adj[pred_op].append(op)
            # else:
                # This might indicate an issue with predecessor definitions
                # print(f"Warning: Predecessor {pred_op} for {op} not in all_operations.")


    queue = [op for op in problem.all_operations if in_degree[op] == 0]
    random.shuffle(queue) # Add randomness
    
    topo_sorted_ops = []
    while queue:
        u = queue.pop(0)
        topo_sorted_ops.append(u)
        
        # Sort successors for deterministic behavior if multiple choices, or shuffle for randomness
        # current_successors = sorted(list(adj[u])) # For determinism
        current_successors = list(adj[u])
        random.shuffle(current_successors) # For randomness

        for v_op in current_successors:
            in_degree[v_op] -= 1
            if in_degree[v_op] == 0:
                queue.append(v_op)
    
    if len(topo_sorted_ops) != problem.total_operations:
        # print(f"Warning: Topological sort found {len(topo_sorted_ops)} ops, expected {problem.total_operations}. Cycle?")
        # Fallback: return a random shuffle, decoder will handle precedence issues (potentially poorly)
        # This indicates a cycle or issue in precedence definition.
        # A robust implementation would handle this more gracefully or raise an error.
        # For now, if topo sort fails, use a random permutation of all operations.
        # This is a critical point: if problem has cycles, topo sort fails.
        # The paper assumes process plans are DAGs.
        if not topo_sorted_ops: # if cycle detected and no ops could be sorted
            temp_ops = list(problem.all_operations)
            random.shuffle(temp_ops)
            return temp_ops
        return topo_sorted_ops


    return topo_sorted_ops


def generate_random_machine_assignment(op_sequence, problem):
    machine_assignment = []
    for op in op_sequence:
        job_idx, op_idx_in_job = op.job_idx, op.op_idx_in_job
        # Get machines that can process this operation
        valid_machines = [m for m in range(problem.num_machines) 
                          if problem.processing_times[job_idx][op_idx_in_job, m] != np.inf]
        if not valid_machines:
            # This should not happen if the problem is well-defined
            # print(f"Error: No valid machine for operation {op}")
            # Fallback: assign a random machine, decoder will give inf makespan
            machine_assignment.append(random.randint(0, problem.num_machines - 1))
        else:
            machine_assignment.append(random.choice(valid_machines))
    return machine_assignment

def initialize_population(pop_size, problem):
    population = []
    # The paper uses "Forward method" and "Backward method"
    # Forward: random OS (respecting precedence somewhat), random MS
    # Backward: reverse of forward OS, random MS
    # For simplicity here, we'll generate OS using topological sort + shuffle
    # and random MS.
    
    # Half forward, half backward (simplified)
    num_forward = pop_size // 2
    num_backward = pop_size - num_forward

    for _ in range(num_forward):
        # Generate a topologically plausible operation sequence
        op_sequence = get_topological_sort_operations(problem)
        if len(op_sequence) != problem.total_operations: # Fallback if topo sort fails
            op_sequence = list(problem.all_operations)
            random.shuffle(op_sequence)

        machine_assignment = generate_random_machine_assignment(op_sequence, problem)
        sol = Solution(op_sequence, machine_assignment)
        decode_solution(sol, problem) # Calculate initial makespan
        population.append(sol)

    for _ in range(num_backward):
        op_sequence_forward = get_topological_sort_operations(problem)
        if len(op_sequence_forward) != problem.total_operations: # Fallback
            op_sequence_forward = list(problem.all_operations)
            random.shuffle(op_sequence_forward)
        
        op_sequence_backward = op_sequence_forward[::-1] # Reverse
        machine_assignment = generate_random_machine_assignment(op_sequence_backward, problem)
        sol = Solution(op_sequence_backward, machine_assignment)
        decode_solution(sol, problem)
        population.append(sol)
        
    return population

# --- 3. Exploration Phase Operators ---
def calculate_dvpc(solution, best_solution_schedule_details, problem):
    """Degree of Variance of Process Completion (DVPC)"""
    dvpc_val = 0
    if not best_solution_schedule_details: # If best solution hasn't been decoded
        return random.random() * 100 # or some other default large value

    for op, details in solution.schedule_details.items():
        if op in best_solution_schedule_details:
            dvpc_val += abs(details['end_time'] - best_solution_schedule_details[op]['end_time'])
        # else:
            # op in current solution but not in best (e.g. different num ops if problem varies)
            # This simple DVPC assumes same set of operations.
            # dvpc_val += details['end_time'] # Penalize if op not in best
    return dvpc_val

def calculate_woc(job_idx, schedule_details_for_job, problem):
    """Work Order Compactness (WOC)"""
    # WOC = sum(MT_ijk) / ET_ipi (actual total processing time / actual completion time of job)
    # schedule_details_for_job: {op_idx_in_job: {'start_time', 'end_time', 'machine', 'proc_time'}}
    
    total_processing_time_for_job = 0
    job_completion_time = 0
    
    ops_in_this_job = [op for op in schedule_details_for_job.keys() if op.job_idx == job_idx]

    if not ops_in_this_job:
        return 0 # Or handle as error / edge case

    for op in ops_in_this_job:
        details = schedule_details_for_job[op]
        assigned_machine = details['machine']
        proc_time = problem.processing_times[op.job_idx][op.op_idx_in_job, assigned_machine]
        total_processing_time_for_job += proc_time
        job_completion_time = max(job_completion_time, details['end_time'])

    if job_completion_time == 0: # Avoid division by zero if job has no ops or zero proc time
        return 0
    
    return total_processing_time_for_job / job_completion_time


def two_d_clustering_crossover(parent1_sol, parent2_sol, population, best_solution_overall, problem):
    # This is a simplified version. The paper's 2D clustering is more involved.
    # It clusters the entire population based on (fitness, DVPC) into k=4 clusters,
    # then selects parents from different clusters.
    # Here, we'll just do a POX-like crossover for the given parents.
    
    # POX Crossover (Precedence Preservative Order-Based Crossover)
    # Simplified: Identify "elite genes" (e.g., a job) from parent1 and keep them.
    # Fill the rest from parent2 while maintaining relative order of parent2's ops.

    child1_op_seq = [None] * problem.total_operations
    child1_ma_seq = [None] * problem.total_operations
    
    # For POFJSP, "elite genes" could be a sub-sequence of operations for a critical job.
    # The paper mentions "WOC" to identify elite genes (jobs with high compactness).
    # Let's try to keep one randomly chosen job's operations from parent1.
    
    # Step 1: Choose a job to inherit from parent1
    job_to_inherit_idx = random.randint(0, problem.num_jobs - 1)
    
    # Get operations of this job and their positions from parent1
    parent1_job_ops = [] # list of (op_object, original_index_in_parent1_op_seq)
    for i, op in enumerate(parent1_sol.operation_sequence):
        if op.job_idx == job_to_inherit_idx:
            parent1_job_ops.append((op, i))
            
    # Place these inherited operations into child1 at their original positions
    for op, original_idx in parent1_job_ops:
        if original_idx < len(child1_op_seq): # Boundary check
            child1_op_seq[original_idx] = op
            child1_ma_seq[original_idx] = parent1_sol.machine_assignment[original_idx]
            
    # Step 2: Fill remaining slots from parent2
    parent2_ops_to_fill = []
    for op in parent2_sol.operation_sequence:
        # Check if this op (or its equivalent if ops are just (job,op_idx)) is already in child1 from parent1
        # This requires careful handling if op objects are not unique or if we are tracking by (job,op_idx)
        # For simplicity, assume op objects are comparable.
        # A more robust check: not any(c_op is not None and c_op.job_idx == op.job_idx and c_op.op_idx_in_job == op.op_idx_in_job for c_op in child1_op_seq)
        is_op_already_placed = False
        for placed_op in child1_op_seq:
            if placed_op is not None and placed_op.job_idx == op.job_idx and placed_op.op_idx_in_job == op.op_idx_in_job:
                is_op_already_placed = True
                break
        if not is_op_already_placed:
             parent2_ops_to_fill.append(op)


    # Fill Nones in child1_op_seq with ops from parent2_ops_to_fill
    fill_idx = 0
    for i in range(len(child1_op_seq)):
        if child1_op_seq[i] is None:
            if fill_idx < len(parent2_ops_to_fill):
                op_to_add = parent2_ops_to_fill[fill_idx]
                child1_op_seq[i] = op_to_add
                # Find machine for op_to_add in parent2
                parent2_op_idx = -1
                for k_op_idx, p2_op in enumerate(parent2_sol.operation_sequence):
                    if p2_op.job_idx == op_to_add.job_idx and p2_op.op_idx_in_job == op_to_add.op_idx_in_job:
                        parent2_op_idx = k_op_idx
                        break
                if parent2_op_idx != -1:
                     child1_ma_seq[i] = parent2_sol.machine_assignment[parent2_op_idx]
                else: # Fallback: random machine
                     child1_ma_seq[i] = generate_random_machine_assignment([op_to_add], problem)[0]
                fill_idx += 1
            else:
                # Should not happen if lengths are consistent and logic is correct
                # print("Error in POX crossover: Not enough ops from parent2 to fill child.")
                # Fallback: fill with a random valid operation if any are missing, or re-initialize
                # This part needs robust handling for missing operations.
                # For now, let's ensure all_operations are present.
                # A quick fix if child1_op_seq has Nones: re-initialize that part.
                # This indicates a potential flaw in the simple POX above.
                # A true POX needs careful handling of all operations.
                pass


    # Ensure child1_op_seq is complete and valid (all ops present once)
    # This simplified POX might lead to missing or duplicate ops if not careful.
    # A full POX implementation is more complex.
    # For now, let's assume the above creates a permutation. If not, it needs fixing.
    # Quick check for Nones:
    if any(op is None for op in child1_op_seq) or any(ma is None for ma in child1_ma_seq):
        # print("Warning: Crossover resulted in incomplete child. Returning copy of parent1.")
        return copy.deepcopy(parent1_sol) # Fallback

    # Ensure all operations are present exactly once
    op_counts = defaultdict(int)
    for op in child1_op_seq:
        op_counts[op] +=1
    if not all(count == 1 for count in op_counts.values()) or len(op_counts) != problem.total_operations:
        # print("Warning: Crossover resulted in invalid op sequence (duplicates/missing). Returning copy of parent1.")
        # This is a common issue with naive crossover. A repair mechanism or better crossover is needed.
        # For this example, we'll return a copy of a parent as a fallback.
        return copy.deepcopy(parent1_sol)


    child_sol = Solution(child1_op_seq, child1_ma_seq)
    decode_solution(child_sol, problem)
    return child_sol


def effective_parallel_mutation(solution, problem, mutation_rate=0.2):
    mutated_sol = copy.deepcopy(solution)
    
    # Machine Mutation (20% of operations as per paper, implied)
    num_ops_to_mutate_machine = int(problem.total_operations * mutation_rate)
    if num_ops_to_mutate_machine == 0 and problem.total_operations > 0 : num_ops_to_mutate_machine = 1

    for _ in range(num_ops_to_mutate_machine):
        if not mutated_sol.operation_sequence: continue
        op_idx_in_seq = random.randint(0, len(mutated_sol.operation_sequence) - 1)
        op_to_mutate = mutated_sol.operation_sequence[op_idx_in_seq]
        
        job_idx, op_idx_in_job = op_to_mutate.job_idx, op_to_mutate.op_idx_in_job
        
        # Find machines that can process this op faster than current
        current_machine = mutated_sol.machine_assignment[op_idx_in_seq]
        current_proc_time = problem.processing_times[job_idx][op_idx_in_job, current_machine]
        
        better_machines = []
        for m_idx in range(problem.num_machines):
            if m_idx == current_machine: continue
            new_proc_time = problem.processing_times[job_idx][op_idx_in_job, m_idx]
            if new_proc_time < current_proc_time and new_proc_time != np.inf:
                better_machines.append(m_idx)
        
        if better_machines:
            mutated_sol.machine_assignment[op_idx_in_seq] = random.choice(better_machines)
        # else: if no strictly better, could try any other valid machine
        elif problem.num_machines > 1:
            valid_machines = [m for m in range(problem.num_machines) 
                              if problem.processing_times[job_idx][op_idx_in_job, m] != np.inf and m != current_machine]
            if valid_machines:
                 mutated_sol.machine_assignment[op_idx_in_seq] = random.choice(valid_machines)


    # Operation Mutation (swap operations on the same machine)
    # The paper says "exchange position of any multiple operations processed on the same machine"
    # This is a bit vague. Let's pick a machine, then swap two ops on it in the OS.
    if problem.num_machines > 0:
        machine_to_mutate_ops_on = random.randint(0, problem.num_machines - 1)
        
        ops_on_this_machine_indices_in_seq = []
        for i, op_assigned_machine in enumerate(mutated_sol.machine_assignment):
            if op_assigned_machine == machine_to_mutate_ops_on:
                ops_on_this_machine_indices_in_seq.append(i)
        
        if len(ops_on_this_machine_indices_in_seq) >= 2:
            idx1_in_list, idx2_in_list = random.sample(range(len(ops_on_this_machine_indices_in_seq)), 2)
            
            # Get actual indices in the full operation_sequence
            seq_idx1 = ops_on_this_machine_indices_in_seq[idx1_in_list]
            seq_idx2 = ops_on_this_machine_indices_in_seq[idx2_in_list]
            
            # Swap in operation_sequence
            op1 = mutated_sol.operation_sequence[seq_idx1]
            op2 = mutated_sol.operation_sequence[seq_idx2]
            mutated_sol.operation_sequence[seq_idx1], mutated_sol.operation_sequence[seq_idx2] = op2, op1
            
            # Machine assignments remain the same as they are on the same machine.
            # MA part also needs to be swapped if we were swapping ops from different machines.
            # Here, we are swapping positions in OS for ops on the SAME machine.
            # The MA for these positions also needs to be swapped if they were different,
            # but since they are on the same machine, their MA values are the same.
            # However, the MA list corresponds to the OS list, so if OS items swap, MA items must also swap.
            ma1 = mutated_sol.machine_assignment[seq_idx1]
            ma2 = mutated_sol.machine_assignment[seq_idx2]
            mutated_sol.machine_assignment[seq_idx1], mutated_sol.machine_assignment[seq_idx2] = ma2, ma1


    decode_solution(mutated_sol, problem)
    return mutated_sol

# --- 4. Development Phase Operators (GNS) ---
def get_bottlenecks(population, problem): # Pass population of Solution objects
    # Bottleneck job: job contributing to the makespan of the best solution
    # Bottleneck machine: machine with the highest utilization or longest processing time
    
    if not population:
        return None, -1 # No job, invalid machine index

    best_sol_in_pop = min(population, key=lambda s: s.makespan) # Assumes makespan is populated
    
    bottleneck_job_idx = -1
    max_job_finish_time = -1

    if not best_sol_in_pop.schedule_details: # if schedule not decoded
        decode_solution(best_sol_in_pop, problem)

    # Find bottleneck job (job that finishes last)
    actual_job_finish_times = [0] * problem.num_jobs
    for op, details in best_sol_in_pop.schedule_details.items():
        actual_job_finish_times[op.job_idx] = max(actual_job_finish_times[op.job_idx], details['end_time'])
    
    if any(actual_job_finish_times):
        bottleneck_job_idx = np.argmax(actual_job_finish_times)

    # Find bottleneck machine (machine that finishes last)
    machine_finish_times = [0] * problem.num_machines
    for m_idx in range(problem.num_machines):
        if best_sol_in_pop.machine_schedules[m_idx]: # If machine has ops
            # machine_schedules[m] is list of (start, end, job, op)
            machine_finish_times[m_idx] = max(op_details[1] for op_details in best_sol_in_pop.machine_schedules[m_idx])
            
    bottleneck_machine_idx = -1
    if any(machine_finish_times):
        bottleneck_machine_idx = np.argmax(machine_finish_times)
        
    return bottleneck_job_idx, bottleneck_machine_idx


def get_operation_priority(op, problem): # op is Operation(job_idx, op_idx_in_job)
    # G1 (High): Multiple successors
    # G2 (Medium): One successor
    # G3 (Low): No successors (last op of a job)
    num_successors = 0
    if op in problem.successors_map:
        num_successors = len(problem.successors_map[op])

    if num_successors > 1: return "G1"
    if num_successors == 1: return "G2"
    return "G3" # num_successors == 0

def grade_neighborhood_search(solution, bottleneck_type, bottleneck_id, problem):
    # bottleneck_type: "job" or "machine"
    # bottleneck_id: job_idx or machine_idx
    
    gns_sol = copy.deepcopy(solution)
    if not gns_sol.schedule_details: # Ensure schedule is decoded
        decode_solution(gns_sol, problem)

    operations_to_consider = []
    if bottleneck_type == "job":
        for i, op in enumerate(gns_sol.operation_sequence):
            if op.job_idx == bottleneck_id:
                operations_to_consider.append({'op_obj': op, 'seq_idx': i, 
                                               'details': gns_sol.schedule_details.get(op, None)})
    elif bottleneck_type == "machine":
        for i, op_assigned_machine in enumerate(gns_sol.machine_assignment):
            if op_assigned_machine == bottleneck_id:
                op_obj = gns_sol.operation_sequence[i]
                operations_to_consider.append({'op_obj': op_obj, 'seq_idx': i,
                                               'details': gns_sol.schedule_details.get(op_obj, None)})
    
    if not operations_to_consider:
        return gns_sol

    # Prioritize operations
    for item in operations_to_consider:
        if item['details']: # Only if op was scheduled
            item['priority_val'] = {"G1": 1, "G2": 2, "G3": 3}[get_operation_priority(item['op_obj'], problem)]
        else: # Should not happen if solution is decoded
            item['priority_val'] = 3 

    # Sort by priority (G1 highest, so lower val is better)
    operations_to_consider.sort(key=lambda x: x['priority_val'])
    
    # Select 10% of operations using roulette wheel (simplified: select top 10% by priority)
    num_to_select = max(1, int(len(operations_to_consider) * 0.1))
    selected_ops_for_gns = operations_to_consider[:num_to_select]

    for op_info in selected_ops_for_gns:
        op_obj = op_info['op_obj']
        seq_idx = op_info['seq_idx']
        op_details = op_info['details']

        if not op_details: continue # Skip if op somehow has no details

        current_machine = op_details['machine']
        current_start_time = op_details['start_time']
        current_proc_time = problem.processing_times[op_obj.job_idx][op_obj.op_idx_in_job, current_machine]

        # Determine predecessor completion time
        pred_completion_time = 0
        if op_obj in problem.predecessors_map:
            for pred_op in problem.predecessors_map[op_obj]:
                if pred_op in gns_sol.schedule_details:
                     pred_completion_time = max(pred_completion_time, gns_sol.schedule_details[pred_op]['end_time'])
        
        # GNS1: Interval > proc_time / 2 -> swap with another op on same machine
        # This is complex to implement correctly as it requires modifying OS and re-decoding.
        # A simpler interpretation: try to move this op earlier on its machine if a gap exists.
        # The paper's GNS1: "randomly exchange positions with other operations processed after
        # the completion time of its predecessor operation on the corresponding machine"
        
        # GNS2: Interval small -> move to faster machine
        # GNS3: No idle time on machine -> move to less loaded machine

        # Simplified GNS: Try to improve by changing machine
        # Try GNS2: move to a faster machine
        found_faster_machine = False
        for m_idx in range(problem.num_machines):
            if m_idx == current_machine: continue
            new_proc_time = problem.processing_times[op_obj.job_idx][op_obj.op_idx_in_job, m_idx]
            if new_proc_time < current_proc_time and new_proc_time != np.inf:
                gns_sol.machine_assignment[seq_idx] = m_idx
                found_faster_machine = True
                break # Take the first faster one found
        
        if not found_faster_machine:
            # Try GNS3: move to a less loaded machine (heuristic: machine that finishes earliest)
            machine_finish_times = [0] * problem.num_machines
            for m_idx_calc in range(problem.num_machines):
                if gns_sol.machine_schedules[m_idx_calc]:
                    machine_finish_times[m_idx_calc] = max(op_sched[1] for op_sched in gns_sol.machine_schedules[m_idx_calc])
            
            sorted_machines_by_load = np.argsort(machine_finish_times)
            for less_loaded_m_idx in sorted_machines_by_load:
                if less_loaded_m_idx == current_machine: continue
                if problem.processing_times[op_obj.job_idx][op_obj.op_idx_in_job, less_loaded_m_idx] != np.inf:
                    gns_sol.machine_assignment[seq_idx] = less_loaded_m_idx
                    break # Take the first valid less loaded one

    decode_solution(gns_sol, problem)
    return gns_sol

# --- 5. Main IAOA+GNS Algorithm ---
def iaoa_gns_algorithm(problem, pop_size, max_iterations):
    # Initialize population
    population = initialize_population(pop_size, problem)
    
    # Find initial best solution
    best_solution_overall = min(population, key=lambda s: s.makespan)
    
    for t in range(max_iterations):
        # Update MOA
        moa = MOA_MIN + t * ((MOA_MAX - MOA_MIN) / max_iterations)
        
        new_population = []
        
        # For 2D clustering crossover, we need DVPC which depends on best solution's schedule
        if not best_solution_overall.schedule_details: # Ensure it's decoded
            decode_solution(best_solution_overall, problem)

        # Prepare data for 2D clustering if used across population
        # For simplicity, the crossover implemented takes two parents directly.
        # A full 2D clustering would:
        # 1. Calculate (fitness, DVPC) for all in population.
        # 2. Normalize these values.
        # 3. Cluster using KMeans (k=4).
        # 4. Select parents from different clusters.

        for i in range(pop_size):
            current_sol = population[i]
            
            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            
            offspring_sol = None

            if r1 > moa: # Exploration Phase
                if r2 > 0.5: # Two-dimensional clustering crossover
                    # Select another parent (e.g., random, or from different cluster if implemented)
                    parent2_idx = random.randint(0, pop_size - 1)
                    while parent2_idx == i: parent2_idx = random.randint(0, pop_size - 1)
                    parent2 = population[parent2_idx]
                    offspring_sol = two_d_clustering_crossover(current_sol, parent2, population, best_solution_overall, problem)
                else: # Effective parallel mutation
                    offspring_sol = effective_parallel_mutation(current_sol, problem)
            else: # Development Phase (GNS)
                # Identify bottlenecks based on current best_solution_overall or current_sol
                # Paper implies GNS is on the current solution being processed.
                # Let's use current_sol's bottlenecks for its GNS.
                # For GNS, we need the schedule of the solution being improved.
                if not current_sol.schedule_details: decode_solution(current_sol, problem)

                # Determine bottleneck for current_sol
                # This is a simplification; paper might imply bottleneck of global best.
                # Let's find bottleneck for the current_sol
                current_sol_bottleneck_job_idx, current_sol_bottleneck_machine_idx = -1, -1
                
                # Simplified bottleneck finding for current_sol
                if current_sol.schedule_details:
                    job_finish_times = [0] * problem.num_jobs
                    for op_cs, details_cs in current_sol.schedule_details.items():
                        job_finish_times[op_cs.job_idx] = max(job_finish_times[op_cs.job_idx], details_cs['end_time'])
                    if any(job_finish_times): current_sol_bottleneck_job_idx = np.argmax(job_finish_times)

                    machine_finish_times = [0] * problem.num_machines
                    for m_idx_cs in range(problem.num_machines):
                        if current_sol.machine_schedules[m_idx_cs]:
                             machine_finish_times[m_idx_cs] = max(op_det[1] for op_det in current_sol.machine_schedules[m_idx_cs])
                    if any(machine_finish_times): current_sol_bottleneck_machine_idx = np.argmax(machine_finish_times)


                if r3 > 0.5: # Bottleneck job GNS
                    if current_sol_bottleneck_job_idx != -1:
                        offspring_sol = grade_neighborhood_search(current_sol, "job", current_sol_bottleneck_job_idx, problem)
                    else: # Fallback if no bottleneck job found
                        offspring_sol = copy.deepcopy(current_sol)
                else: # Bottleneck machine GNS
                    if current_sol_bottleneck_machine_idx != -1:
                        offspring_sol = grade_neighborhood_search(current_sol, "machine", current_sol_bottleneck_machine_idx, problem)
                    else: # Fallback
                        offspring_sol = copy.deepcopy(current_sol)
            
            # Elitism: Compare offspring with current_sol
            if offspring_sol.makespan < current_sol.makespan:
                new_population.append(offspring_sol)
            else:
                new_population.append(current_sol)

        population = new_population
        current_best_in_pop = min(population, key=lambda s: s.makespan)
        if current_best_in_pop.makespan < best_solution_overall.makespan:
            best_solution_overall = copy.deepcopy(current_best_in_pop) # Deepcopy to save state

        # print(f"Iteration {t+1}/{max_iterations}, Best Makespan: {best_solution_overall.makespan}")
            
    return best_solution_overall


# --- Example Usage ---
if __name__ == '__main__':
    # --- Define a sample problem instance (e.g., from paper's Mk01 or a small example) ---
    # This needs to be carefully defined based on your specific problem.
    # Example: 2 jobs, 2 machines
    # Job 0: Op0_0 (J0,O0), Op0_1 (J0,O1)
    # Job 1: Op1_0 (J1,O0), Op1_1 (J1,O1)
    
    # num_operations_per_job = [2, 2] # J0 has 2 ops, J1 has 2 ops
    # processing_times = [
    #     # Job 0: [[Op0_0_M0, Op0_0_M1], [Op0_1_M0, Op0_1_M1]]
    #     np.array([[3, 5], [6, np.inf]]),  # J0,O0: M0=3, M1=5; J0,O1: M0=6, M1=cannot
    #     # Job 1: [[Op1_0_M0, Op1_0_M1], [Op1_1_M0, Op1_1_M1]]
    #     np.array([[4, 2], [np.inf, 7]])   # J1,O0: M0=4, M1=2; J1,O1: M0=cannot, M1=7
    # ]

    # predecessors_map = {
    #     Operation(0,1): {Operation(0,0)}, # J0,O1 needs J0,O0
    #     Operation(1,1): {Operation(1,0)}, # J1,O1 needs J1,O0
    #     Operation(0,0): set(),
    #     Operation(1,0): set()
    # }
    # successors_map = { # Can be derived from predecessors
    #     Operation(0,0): {Operation(0,1)},
    #     Operation(1,0): {Operation(1,1)},
    #     Operation(0,1): set(),
    #     Operation(1,1): set()
    # }
    
    # problem_instance = ProblemInstance(
    #     num_jobs=2,
    #     num_machines=2,
    #     num_operations_per_job=num_operations_per_job,
    #     processing_times=processing_times,
    #     predecessors_map=predecessors_map,
    #     successors_map=successors_map
    # )

    # print("Problem Instance Defined.")
    # print(f"Total operations: {problem_instance.total_operations}")
    # print(f"All operations: {problem_instance.all_operations}")

    # # --- Algorithm Parameters ---
    # POP_SIZE = 20 # Small for quick test, paper uses 80-100
    # MAX_ITERATIONS = 50 # Small for quick test, paper uses 50-70

    # print(f"\nRunning IAOA+GNS with PopSize={POP_SIZE}, MaxIter={MAX_ITERATIONS}...")
    # best_found_solution = iaoa_gns_algorithm(problem_instance, POP_SIZE, MAX_ITERATIONS)
    
    # print("\n--- Best Solution Found ---")
    # if best_found_solution and best_found_solution.makespan != float('inf'):
    #     print(f"Makespan: {best_found_solution.makespan}")
    #     print("Operation Sequence (Job, Op):")
    #     for op in best_found_solution.operation_sequence:
    #         print(f"  ({op.job_idx}, {op.op_idx_in_job}) -> M{best_found_solution.machine_assignment[best_found_solution.operation_sequence.index(op)]}")
        
    #     print("\nSchedule Details (Operation: Start, End, Machine):")
    #     for op, details in sorted(best_found_solution.schedule_details.items(), key=lambda item: item[1]['start_time']):
    #         print(f"  Op({op.job_idx},{op.op_idx_in_job}): Start={details['start_time']}, End={details['end_time']}, Machine={details['machine']}")
        
    #     print("\nMachine Schedules:")
    #     for m_idx in range(problem_instance.num_machines):
    #         print(f"  Machine {m_idx}:")
    #         if best_found_solution.machine_schedules[m_idx]:
    #             for op_sched_detail in sorted(best_found_solution.machine_schedules[m_idx]): # Sort by start time
    #                 print(f"    Op({op_sched_detail[2]},{op_sched_detail[3]}) from {op_sched_detail[0]} to {op_sched_detail[1]}")
    #         else:
    #             print("    (idle)")
    # else:
    #     print("No valid solution found or algorithm did not complete as expected.")

    print("IAOA+GNS algorithm structure implemented.")
    print("To run, uncomment and define a 'problem_instance' in the 'if __name__ == \"__main__\":' block.")
    print("The example problem is commented out. You need to provide your own problem data.")

