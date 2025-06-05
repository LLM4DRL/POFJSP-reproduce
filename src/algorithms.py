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
    # Debug: Print operation sequence and machine assignment
    print(f"Decoding solution with {len(solution.operation_sequence)} operations")
    
    # schedule_details: {Operation(j,o): {'start_time', 'end_time', 'machine'}}
    schedule_details = {}
    # machine_schedules: list of lists, machine_schedules[m] = sorted list of (start_time, end_time)
    machine_schedules = [[] for _ in range(problem.num_machines)] # Stores (start_time, end_time) tuples

    # Keep track of completion times of operations for precedence constraints
    operation_completion_times = {} # {Operation(j,o): end_time}
    
    # NEW: Track which operations have been scheduled to avoid duplicates
    scheduled_operations = set()

    # IMPROVED APPROACH: Use a priority queue based on precedence constraints
    # Create a graph of operation dependencies
    in_degree = {op: 0 for op in solution.operation_sequence}
    for op in solution.operation_sequence:
        if op in problem.predecessors_map:
            in_degree[op] = len(problem.predecessors_map[op])
    
    # Find operations with no predecessors
    ready_operations = [op for op in solution.operation_sequence if in_degree[op] == 0]
    
    # Process operations in topological order
    processed_ops = 0
    
    while ready_operations and processed_ops < len(solution.operation_sequence):
        # Get the next operation
        current_op = ready_operations.pop(0)
        
        # NEW: Skip if this operation has already been scheduled
        if current_op in scheduled_operations:
            print(f"WARNING: Operation {current_op} was already scheduled. Skipping duplicate.")
            continue
        
        # Find its index in the operation sequence
        op_idx = solution.operation_sequence.index(current_op)
        assigned_machine = solution.machine_assignment[op_idx]
        proc_time = problem.processing_times[current_op.job_idx][current_op.op_idx_in_job, assigned_machine]

        if proc_time == np.inf: # Should not happen with valid machine assignment
            print(f"ERROR: Invalid processing time for op={current_op}, machine={assigned_machine}")
            solution.makespan = float('inf')
            solution.schedule_details = {}
            solution.machine_schedules = [[] for _ in range(problem.num_machines)]
            return float('inf'), {}, [[] for _ in range(problem.num_machines)]

        # Determine earliest start time based on predecessors
        earliest_start_due_to_predecessors = 0
        if current_op in problem.predecessors_map:
            for pred_op in problem.predecessors_map[current_op]:
                if pred_op not in operation_completion_times:
                    print(f"WARNING: Predecessor {pred_op} for {current_op} not scheduled yet. This might indicate a topological sort issue.")
                    # If predecessor not scheduled, we'll need to handle this case
                    # For now, let's be robust and continue, but this is not ideal
                else:
                    earliest_start_due_to_predecessors = max(
                        earliest_start_due_to_predecessors,
                        operation_completion_times[pred_op]
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
        
        # Debug: Check for negative or strange start times
        if op_start_time < 0:
            print(f"ERROR: Negative start time at index {op_idx}, op={current_op}, machine={assigned_machine}, start_time={op_start_time}")

        # Update schedules
        heapq.heappush(machine_schedules[assigned_machine], (op_start_time, op_end_time)) # Keep sorted by start time

        schedule_details[current_op] = {
            'start_time': op_start_time,
            'end_time': op_end_time,
            'machine': assigned_machine
        }
        operation_completion_times[current_op] = op_end_time
        
        # NEW: Mark this operation as scheduled
        scheduled_operations.add(current_op)
        
        # Update successors' in_degree and add to ready queue if all predecessors processed
        if current_op in problem.successors_map:
            for succ_op in problem.successors_map[current_op]:
                if succ_op in in_degree:
                    in_degree[succ_op] -= 1
                    if in_degree[succ_op] == 0:
                        ready_operations.append(succ_op)
        
        processed_ops += 1
    
    # Check if all operations were processed
    if processed_ops < len(solution.operation_sequence):
        print(f"WARNING: Not all operations were processed. This indicates a cycle in the precedence graph.")
        # Add remaining operations in the sequence even though it violates precedence
        for op in solution.operation_sequence:
            # NEW: Skip if this operation has already been scheduled
            if op in scheduled_operations:
                continue
                
            if op not in operation_completion_times:
                op_idx = solution.operation_sequence.index(op)
                assigned_machine = solution.machine_assignment[op_idx]
                proc_time = problem.processing_times[op.job_idx][op.op_idx_in_job, assigned_machine]
                
                # Schedule after the last operation on the machine
                machine_schedules[assigned_machine].sort()
                last_finish_time_on_machine = 0
                if machine_schedules[assigned_machine]:
                    last_finish_time_on_machine = machine_schedules[assigned_machine][-1][1]
                
                op_start_time = last_finish_time_on_machine
                op_end_time = op_start_time + proc_time
                
                heapq.heappush(machine_schedules[assigned_machine], (op_start_time, op_end_time))
                
                schedule_details[op] = {
                    'start_time': op_start_time,
                    'end_time': op_end_time,
                    'machine': assigned_machine
                }
                operation_completion_times[op] = op_end_time
                scheduled_operations.add(op)  # NEW: Mark as scheduled
                processed_ops += 1

    makespan = 0
    if operation_completion_times:
        makespan = max(operation_completion_times.values())
    
    # Debug: Check makespan
    print(f"Calculated makespan: {makespan}")
    
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
    """
    Generates random machine assignments for a sequence of operations.
    CRITICAL: This ensures operations of the same job are distributed across different machines.
    """
    machine_assignment = []
    
    # Track job distribution across machines
    job_machine_count = {}  # {job_idx: {machine_idx: count}}
    job_last_machine = {}   # {job_idx: last_machine_used}
    
    # Initialize job_machine_count
    for j in range(problem.num_jobs):
        job_machine_count[j] = {m: 0 for m in range(problem.num_machines)}
    
    for op in op_sequence:
        job_idx, op_idx_in_job = op.job_idx, op.op_idx_in_job
        
        # Get machines that can process this operation
        valid_machines = [m for m in range(problem.num_machines) 
                         if problem.processing_times[job_idx][op_idx_in_job, m] != np.inf]
        
        if not valid_machines:
            # Fallback if no valid machine (shouldn't happen with well-formed data)
            print(f"Warning: No valid machine found for operation {op}. Using machine 0.")
            machine_assignment.append(0)
            continue
        
        # If this is the first operation of the job, randomly choose a machine
        if job_idx not in job_last_machine or op_idx_in_job == 0:
            # Still pick the fastest machine for the first operation
            fastest_machine = min(valid_machines, key=lambda m: problem.processing_times[job_idx][op_idx_in_job, m])
            job_last_machine[job_idx] = fastest_machine
            job_machine_count[job_idx][fastest_machine] += 1
            machine_assignment.append(fastest_machine)
            continue
        
        # For subsequent operations, try to avoid using the same machine as previous operations
        
        # Calculate scores for each machine based on several criteria:
        # 1. Processing time (lower is better)
        # 2. Current usage by this job (lower is better)
        # 3. Not the same as last machine used for this job (bonus)
        machine_scores = {}
        for m in valid_machines:
            proc_time = problem.processing_times[job_idx][op_idx_in_job, m]
            
            # Calculate normalized processing time score (0-1, lower is better)
            min_time = min(problem.processing_times[job_idx][op_idx_in_job, vm] for vm in valid_machines)
            max_time = max(problem.processing_times[job_idx][op_idx_in_job, vm] for vm in valid_machines)
            time_range = max_time - min_time
            time_score = 0 if time_range == 0 else (proc_time - min_time) / time_range
            
            # Calculate usage score (how many operations of this job already on this machine)
            usage_count = job_machine_count[job_idx][m]
            usage_score = usage_count / (op_idx_in_job + 1)  # Normalize by number of ops processed so far
            
            # Penalty for using the same machine as the last operation of this job
            last_machine_penalty = 1.0 if m == job_last_machine.get(job_idx) else 0.0
            
            # Combine scores - lower is better
            machine_scores[m] = (0.3 * time_score) + (0.5 * usage_score) + (0.2 * last_machine_penalty)
        
        # Choose machine with lowest score (best option)
        best_machine = min(valid_machines, key=lambda m: machine_scores[m])
        
        # Force a different machine if there are valid alternatives and we've used this machine before
        if len(valid_machines) > 1 and job_machine_count[job_idx][best_machine] > 0:
            # Find the least used machine that is not the best machine
            alternative_machines = [m for m in valid_machines if m != best_machine]
            least_used = min(alternative_machines, 
                             key=lambda m: (job_machine_count[job_idx][m], problem.processing_times[job_idx][op_idx_in_job, m]))
            best_machine = least_used
        
        # Update tracking
        job_last_machine[job_idx] = best_machine
        job_machine_count[job_idx][best_machine] += 1
        machine_assignment.append(best_machine)
    
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
    # Debug: Starting crossover
    print(f"  Starting 2D clustering crossover. Parent1 makespan: {parent1_sol.makespan}, Parent2 makespan: {parent2_sol.makespan}")
    
    # This is a simplified version. The paper's 2D clustering is more involved.
    # It clusters the entire population based on (fitness, DVPC) into k=4 clusters,
    # then selects parents from different clusters.
    # Here, we'll implement a POX-like crossover that preserves precedence constraints.
    
    # IMPROVED METHOD: Use a topological sort as the basis for crossover
    # Step 1: Get a valid topological sort of operations to use as a template
    topo_sort = get_topological_sort_operations(problem)
    print(f"    Using topological sort as template to ensure precedence validity")
    
    # Step 2: Instead of inheriting a single job from parent1, inherit a random selection of jobs
    # This provides more diversity in the offspring
    num_jobs_to_inherit = max(1, int(problem.num_jobs * 0.3))  # Inherit about 30% of jobs
    jobs_to_inherit = random.sample(range(problem.num_jobs), num_jobs_to_inherit)
    print(f"    Inheriting jobs {jobs_to_inherit} from parent1")
    
    # Step 3: Create mapping from operations to their machines in each parent
    parent1_op_to_machine = {}
    for i, op in enumerate(parent1_sol.operation_sequence):
        parent1_op_to_machine[op] = parent1_sol.machine_assignment[i]
        
    parent2_op_to_machine = {}
    for i, op in enumerate(parent2_sol.operation_sequence):
        parent2_op_to_machine[op] = parent2_sol.machine_assignment[i]
    
    # Step 4: Create child's operation sequence following the topological order
    child_op_seq = topo_sort.copy()
    
    # Step 5: Create machine assignments with improved distribution
    job_machine_count = {}  # Track distribution of operations per job across machines
    for j in range(problem.num_jobs):
        job_machine_count[j] = {m: 0 for m in range(problem.num_machines)}
        
    job_last_machine = {}  # Track the last machine used for each job
    child_ma_seq = []
    
    for op in child_op_seq:
        job_idx, op_idx_in_job = op.job_idx, op.op_idx_in_job
        
        # Determine parent machine (if available)
        if op.job_idx in jobs_to_inherit and op in parent1_op_to_machine:
            parent_machine = parent1_op_to_machine[op]
        elif op in parent2_op_to_machine:
            parent_machine = parent2_op_to_machine[op]
        else:
            parent_machine = None
            
        # Get valid machines for this operation
        valid_machines = [m for m in range(problem.num_machines) 
                          if problem.processing_times[job_idx][op_idx_in_job, m] != np.inf]
        
        if not valid_machines:
            # No valid machines - use parent's machine or random
            if parent_machine is not None:
                chosen_machine = parent_machine
            else:
                chosen_machine = random.randint(0, problem.num_machines - 1)
                print(f"    WARNING: No valid machines for op ({job_idx},{op_idx_in_job})")
        else:
            # Determine which machines are already heavily used for this job
            current_machines = [m for m, count in job_machine_count[job_idx].items() if count > 0]
            
            # If all operations of this job so far are on a single machine, try to diversify
            if len(current_machines) == 1 and len(valid_machines) > 1 and job_machine_count[job_idx][current_machines[0]] > 0:
                # Force diversification - try to use a different machine than current_machines[0]
                other_machines = [m for m in valid_machines if m != current_machines[0]]
                
                if other_machines:
                    # Calculate machine scores based on processing time
                    machine_scores = []
                    for m in other_machines:
                        proc_time = problem.processing_times[job_idx][op_idx_in_job, m]
                        score = proc_time
                        
                        # Small bias for parent's machine if it's in other_machines
                        if parent_machine is not None and m == parent_machine:
                            score *= 0.9  # Reduce score by 10% to favor parent's choice
                            
                        machine_scores.append((score, m))
                        
                    # Sort by score (lower is better)
                    machine_scores.sort()
                    
                    if machine_scores:
                        chosen_machine = machine_scores[0][1]  # Best alternative
                    else:
                        chosen_machine = random.choice(other_machines)
                else:
                    # If no other options, use parent's machine or current machine
                    if parent_machine is not None and parent_machine in valid_machines:
                        chosen_machine = parent_machine
                    else:
                        chosen_machine = current_machines[0]
            else:
                # Normal case - score machines
                machine_scores = []
                for m in valid_machines:
                    # Base score is processing time
                    proc_time = problem.processing_times[job_idx][op_idx_in_job, m]
                    score = proc_time
                    
                    # Heavy penalty for machines that already have many ops from this job
                    score += job_machine_count[job_idx][m] * 20
                    
                    # Penalty for consecutive use of the same machine
                    if job_idx in job_last_machine and job_last_machine[job_idx] == m:
                        score += 30
                    
                    # Favor parent's machine with a bonus
                    if parent_machine is not None and m == parent_machine:
                        score *= 0.85  # 15% discount
                        
                    machine_scores.append((score, m))
                
                # Sort by score
                machine_scores.sort()
                
                # Choose with bias toward best machine
                if random.random() < 0.7 or len(machine_scores) == 1:
                    chosen_machine = machine_scores[0][1]  # Best scoring machine
                else:
                    # Otherwise select from top 40% of candidates
                    top_count = max(1, int(len(machine_scores) * 0.4))
                    chosen_machine = random.choice([m for _, m in machine_scores[:top_count]])
        
        # Add chosen machine to assignments
        child_ma_seq.append(chosen_machine)
        job_last_machine[job_idx] = chosen_machine
        job_machine_count[job_idx][chosen_machine] += 1
    
    # Check validity of child
    if len(child_op_seq) != problem.total_operations or len(child_ma_seq) != problem.total_operations:
        print(f"    WARNING: Invalid child solution (length mismatch). Using parent1 as fallback.")
        return copy.deepcopy(parent1_sol)
        
    # Make sure all operations are included (should be guaranteed by topo_sort)
    op_counts = defaultdict(int)
    for op in child_op_seq:
        op_counts[op] += 1
    
    if not all(count == 1 for count in op_counts.values()) or len(op_counts) != problem.total_operations:
        print(f"    WARNING: Invalid operation sequence (duplicates/missing). Using parent1 as fallback.")
        return copy.deepcopy(parent1_sol)
        
    print(f"    Successfully created child with {len(child_op_seq)} operations")
    child_sol = Solution(child_op_seq, child_ma_seq)
    decode_solution(child_sol, problem)
    print(f"    Child makespan: {child_sol.makespan}")
    return child_sol


def effective_parallel_mutation(solution, problem, mutation_rate=0.2):
    print(f"    Starting effective parallel mutation")
    
    # IMPROVED APPROACH: Ensure topological ordering is maintained
    # First, create a new solution that respects topological ordering
    topo_sort = get_topological_sort_operations(problem)
    
    # Create mappings from operations to machine assignments
    op_to_machine = {}
    for i, op in enumerate(solution.operation_sequence):
        op_to_machine[op] = solution.machine_assignment[i]
    
    # Create a new solution that follows topological ordering
    mutated_op_seq = topo_sort.copy()
    
    # Track job distribution across machines
    job_machine_count = {}  # {job_idx: {machine: count}}
    for j in range(problem.num_jobs):
        job_machine_count[j] = {m: 0 for m in range(problem.num_machines)}
    
    # First pass: count current distribution
    for op, machine in op_to_machine.items():
        job_machine_count[op.job_idx][machine] += 1
    
    # Now determine machine assignments with improved distribution
    job_last_machine = {}  # Keep track of last machine used for each job
    mutated_ma_seq = []
    
    # Calculate how many operations to mutate
    num_ops_to_mutate = max(1, int(len(mutated_op_seq) * mutation_rate))
    ops_to_mutate = set(random.sample(range(len(mutated_op_seq)), num_ops_to_mutate))
    
    # Find jobs that need diversification (all ops on same machine)
    jobs_needing_diversity = []
    for j in range(problem.num_jobs):
        machines_used = [m for m, count in job_machine_count[j].items() if count > 0]
        if len(machines_used) == 1 and sum(job_machine_count[j].values()) > 1:
            jobs_needing_diversity.append((j, machines_used[0]))  # (job_idx, current_machine)
    
    # Prioritize diversifying these jobs in mutation
    priority_ops = []
    if jobs_needing_diversity:
        for job_idx, current_machine in jobs_needing_diversity:
            # Find operations from this job in our sequence
            for i, op in enumerate(mutated_op_seq):
                if op.job_idx == job_idx:
                    priority_ops.append((i, op, current_machine))
                    if len(priority_ops) >= num_ops_to_mutate:
                        break
    
    # Add these priority ops to the mutation set
    for i, _, _ in priority_ops[:num_ops_to_mutate]:
        ops_to_mutate.add(i)
    
    for i, op in enumerate(mutated_op_seq):
        job_idx, op_idx_in_job = op.job_idx, op.op_idx_in_job
        
        # Get the current machine assignment
        current_machine = op_to_machine.get(op, None)
        
        # Determine if this operation should be mutated
        should_mutate = (i in ops_to_mutate) or (current_machine is None)
        
        if should_mutate:
            # Get valid machines for this operation
            valid_machines = [m for m in range(problem.num_machines) 
                              if problem.processing_times[job_idx][op_idx_in_job, m] != np.inf]
            
            if not valid_machines:
                # If no valid machines (shouldn't happen with proper problem definition)
                if current_machine is not None:
                    # Keep current assignment if it exists
                    mutated_ma_seq.append(current_machine)
                    job_machine_count[job_idx][current_machine] += 1
                else:
                    # Random fallback
                    random_machine = random.randint(0, problem.num_machines - 1)
                    mutated_ma_seq.append(random_machine)
                    job_machine_count[job_idx][random_machine] += 1
            else:
                # Find which machines are already heavily used for this job
                max_ops_on_machine = max(job_machine_count[job_idx][m] for m in valid_machines)
                most_used_machines = [m for m in valid_machines 
                                     if job_machine_count[job_idx][m] == max_ops_on_machine 
                                     and max_ops_on_machine > 0]
                
                # Score valid machines
                machine_scores = []
                for m in valid_machines:
                    # Base score is processing time
                    proc_time = problem.processing_times[job_idx][op_idx_in_job, m]
                    score = proc_time
                    
                    # Heavy penalty for machines that already have many ops from this job
                    if m in most_used_machines:
                        score += 100
                    
                    # Penalty based on how many ops from this job are already on this machine
                    score += job_machine_count[job_idx][m] * 30
                    
                    # Penalty if this is the last used machine for this job
                    if job_idx in job_last_machine and job_last_machine[job_idx] == m:
                        score += 50
                    
                    machine_scores.append((score, m))
                
                # Sort by score
                machine_scores.sort()
                
                # FORCED DIVERSIFICATION:
                # If this job already has all operations on one machine, force diversity
                current_machines = [m for m, count in job_machine_count[job_idx].items() if count > 0]
                
                if len(current_machines) == 1 and len(valid_machines) > 1:
                    # Force choice of a different machine if possible
                    other_machines = [m for m in valid_machines if m not in current_machines]
                    if other_machines:
                        # Find best alternative machine
                        alt_scores = [(s, m) for s, m in machine_scores if m in other_machines]
                        if alt_scores:
                            chosen_machine = alt_scores[0][1]
                        else:
                            chosen_machine = random.choice(other_machines)
                    else:
                        # If no alternative, use best scoring machine
                        chosen_machine = machine_scores[0][1]
                else:
                    # Choose from top candidates with bias toward best
                    top_count = max(1, min(3, len(machine_scores)))
                    
                    if random.random() < 0.7:  # 70% chance to pick best
                        chosen_machine = machine_scores[0][1]
                    else:
                        # Otherwise random from top options
                        chosen_machine = machine_scores[random.randint(0, top_count-1)][1]
                
                # Update assignments
                mutated_ma_seq.append(chosen_machine)
                job_last_machine[job_idx] = chosen_machine
                job_machine_count[job_idx][chosen_machine] += 1
        else:
            # Keep the current machine assignment
            mutated_ma_seq.append(current_machine)
            job_machine_count[job_idx][current_machine] += 1
            job_last_machine[job_idx] = current_machine
    
    # Create the mutated solution
    mutated_sol = Solution(mutated_op_seq, mutated_ma_seq)
    decode_solution(mutated_sol, problem)
    
    # Ensure we didn't break anything
    if mutated_sol.makespan == float('inf'):
        print("    WARNING: Mutation resulted in invalid solution with inf makespan. Using original.")
        return copy.deepcopy(solution)
        
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
    
    print(f"    Starting GNS for {bottleneck_type} {bottleneck_id}")
    
    # IMPROVED APPROACH: Ensure topological ordering is maintained
    # First, create a new solution that respects topological ordering
    topo_sort = get_topological_sort_operations(problem)
    
    # Create mappings from operations to machine assignments
    op_to_machine = {}
    for i, op in enumerate(solution.operation_sequence):
        op_to_machine[op] = solution.machine_assignment[i]
    
    # Create a new solution that follows topological ordering
    gns_op_seq = topo_sort.copy()
    gns_ma_seq = [op_to_machine.get(op, generate_random_machine_assignment([op], problem)[0]) for op in gns_op_seq]
    
    gns_sol = Solution(gns_op_seq, gns_ma_seq)
    
    # First, decode the solution to get the schedule
    decode_solution(gns_sol, problem)
    
    # Find operations to consider for GNS
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
        print(f"    No operations found for {bottleneck_type} {bottleneck_id}")
        return solution  # Return original solution if no operations found

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
    
    print(f"    Selected {len(selected_ops_for_gns)} operations for GNS")

    for op_info in selected_ops_for_gns:
        op_obj = op_info['op_obj']
        seq_idx = op_info['seq_idx']
        op_details = op_info['details']

        if not op_details: 
            print(f"    Skipping op {op_obj} - no details")
            continue # Skip if op somehow has no details

        current_machine = op_details['machine']
        current_start_time = op_details['start_time']
        current_proc_time = problem.processing_times[op_obj.job_idx][op_obj.op_idx_in_job, current_machine]

        # Determine predecessor completion time
        pred_completion_time = 0
        if op_obj in problem.predecessors_map:
            for pred_op in problem.predecessors_map[op_obj]:
                if pred_op in gns_sol.schedule_details:
                     pred_completion_time = max(pred_completion_time, gns_sol.schedule_details[pred_op]['end_time'])
        
        # Try GNS2: move to a faster machine
        found_faster_machine = False
        for m_idx in range(problem.num_machines):
            if m_idx == current_machine: continue
            new_proc_time = problem.processing_times[op_obj.job_idx][op_obj.op_idx_in_job, m_idx]
            if new_proc_time < current_proc_time and new_proc_time != np.inf:
                gns_sol.machine_assignment[seq_idx] = m_idx
                found_faster_machine = True
                print(f"    Moving op {op_obj} to faster machine {m_idx}")
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
                    print(f"    Moving op {op_obj} to less loaded machine {less_loaded_m_idx}")
                    break # Take the first valid less loaded one

    decode_solution(gns_sol, problem)
    print(f"    GNS complete, new makespan: {gns_sol.makespan}")
    return gns_sol

# --- 5. Main IAOA+GNS Algorithm ---
def iaoa_gns_algorithm(problem, pop_size, max_iterations):
    # Debug: Print algorithm parameters
    print(f"Starting IAOA+GNS algorithm with pop_size={pop_size}, max_iterations={max_iterations}")
    print(f"Problem: {problem.num_jobs} jobs, {problem.num_machines} machines, {problem.total_operations} operations")
    
    # Initialize population
    print("Initializing population...")
    population = initialize_population(pop_size, problem)
    
    # Find initial best solution
    best_solution_overall = min(population, key=lambda s: s.makespan)
    print(f"Initial best solution makespan: {best_solution_overall.makespan}")
    
    for t in range(max_iterations):
        # Update MOA
        moa = MOA_MIN + t * ((MOA_MAX - MOA_MIN) / max_iterations)
        
        new_population = []
        
        # For 2D clustering crossover, we need DVPC which depends on best solution's schedule
        if not best_solution_overall.schedule_details: # Ensure it's decoded
            print("Decoding best solution (needed for DVPC calculation)")
            decode_solution(best_solution_overall, problem)

        # Debug: Print iteration status
        print(f"Iteration {t+1}/{max_iterations}, MOA={moa:.4f}, Current best makespan: {best_solution_overall.makespan}")

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
                    
                    # Debug: Print crossover info
                    print(f"  Solution {i}: Using 2D clustering crossover")
                    offspring_sol = two_d_clustering_crossover(current_sol, parent2, population, best_solution_overall, problem)
                else: # Effective parallel mutation
                    # Debug: Print mutation info
                    print(f"  Solution {i}: Using effective parallel mutation")
                    offspring_sol = effective_parallel_mutation(current_sol, problem)
            else: # Development Phase (GNS)
                # Identify bottlenecks based on current best_solution_overall or current_sol
                # Paper implies GNS is on the current solution being processed.
                # Let's use current_sol's bottlenecks for its GNS.
                # For GNS, we need the schedule of the solution being improved.
                if not current_sol.schedule_details: 
                    # Debug: Print decoding for GNS
                    print(f"  Solution {i}: Decoding solution for GNS")
                    decode_solution(current_sol, problem)

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

                    # Debug: Print bottleneck info
                    print(f"  Solution {i}: Bottleneck job={current_sol_bottleneck_job_idx}, machine={current_sol_bottleneck_machine_idx}")

                if r3 > 0.5: # Bottleneck job GNS
                    if current_sol_bottleneck_job_idx != -1:
                        # Debug: Print GNS job info
                        print(f"  Solution {i}: Using job GNS on job {current_sol_bottleneck_job_idx}")
                        offspring_sol = grade_neighborhood_search(current_sol, "job", current_sol_bottleneck_job_idx, problem)
                    else: # Fallback if no bottleneck job found
                        print(f"  Solution {i}: No bottleneck job found, using copy")
                        offspring_sol = copy.deepcopy(current_sol)
                else: # Bottleneck machine GNS
                    if current_sol_bottleneck_machine_idx != -1:
                        # Debug: Print GNS machine info
                        print(f"  Solution {i}: Using machine GNS on machine {current_sol_bottleneck_machine_idx}")
                        offspring_sol = grade_neighborhood_search(current_sol, "machine", current_sol_bottleneck_machine_idx, problem)
                    else: # Fallback
                        print(f"  Solution {i}: No bottleneck machine found, using copy")
                        offspring_sol = copy.deepcopy(current_sol)
            
            # Elitism: Compare offspring with current_sol
            if offspring_sol.makespan < current_sol.makespan:
                print(f"  Solution {i}: Improvement! {current_sol.makespan} -> {offspring_sol.makespan}")
                new_population.append(offspring_sol)
            else:
                print(f"  Solution {i}: No improvement. {current_sol.makespan} -> {offspring_sol.makespan}")
                new_population.append(current_sol)

        population = new_population
        current_best_in_pop = min(population, key=lambda s: s.makespan)
        if current_best_in_pop.makespan < best_solution_overall.makespan:
            best_solution_overall = copy.deepcopy(current_best_in_pop) # Deepcopy to save state
            print(f"  New overall best solution found! Makespan: {best_solution_overall.makespan}")

    print(f"Algorithm completed. Final best makespan: {best_solution_overall.makespan}")
    return best_solution_overall


# --- Example Usage ---
if __name__ == '__main__':
    # --- Define a sample problem instance (e.g., from paper's Mk01 or a small example) ---
    # This needs to be carefully defined based on your specific problem.
    # Example: 2 jobs, 2 machines
    # Job 0: Op0_0 (J0,O0), Op0_1 (J0,O1)
    # Job 1: Op1_0 (J1,O0), Op1_1 (J1,O1)
    
    print("--- DEBUGGING SAMPLE RUN ---")
    
    num_operations_per_job = [2, 2] # J0 has 2 ops, J1 has 2 ops
    processing_times = [
        # Job 0: [[Op0_0_M0, Op0_0_M1], [Op0_1_M0, Op0_1_M1]]
        np.array([[3, 5], [6, np.inf]]),  # J0,O0: M0=3, M1=5; J0,O1: M0=6, M1=cannot
        # Job 1: [[Op1_0_M0, Op1_0_M1], [Op1_1_M0, Op1_1_M1]]
        np.array([[4, 2], [np.inf, 7]])   # J1,O0: M0=4, M1=2; J1,O1: M0=cannot, M1=7
    ]

    predecessors_map = {
        Operation(0,1): {Operation(0,0)}, # J0,O1 needs J0,O0
        Operation(1,1): {Operation(1,0)}, # J1,O1 needs J1,O0
        Operation(0,0): set(),
        Operation(1,0): set()
    }
    successors_map = { # Can be derived from predecessors
        Operation(0,0): {Operation(0,1)},
        Operation(1,0): {Operation(1,1)},
        Operation(0,1): set(),
        Operation(1,1): set()
    }
    
    problem_instance = ProblemInstance(
        num_jobs=2,
        num_machines=2,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )

    print("Problem Instance Defined.")
    print(f"Total operations: {problem_instance.total_operations}")
    print(f"All operations: {problem_instance.all_operations}")

    # --- Algorithm Parameters ---
    POP_SIZE = 10 # Small for quick test, paper uses 80-100
    MAX_ITERATIONS = 5 # Small for quick test, paper uses 50-70

    print(f"\nRunning IAOA+GNS with PopSize={POP_SIZE}, MaxIter={MAX_ITERATIONS}...")
    
    # Debug: Test the topological sort
    print("\nTesting topological sort:")
    topo_sort = get_topological_sort_operations(problem_instance)
    print(f"Topological sort result: {topo_sort}")
    
    # Debug: Test the machine assignment
    print("\nTesting machine assignment:")
    machine_assignment = generate_random_machine_assignment(topo_sort, problem_instance)
    print(f"Machine assignment: {machine_assignment}")
    
    # Debug: Test solution decoding
    print("\nTesting solution decoding:")
    test_solution = Solution(topo_sort, machine_assignment)
    makespan, schedule_details, machine_schedules = decode_solution(test_solution, problem_instance)
    print(f"Decoded makespan: {makespan}")
    
    # Run the full algorithm
    best_found_solution = iaoa_gns_algorithm(problem_instance, POP_SIZE, MAX_ITERATIONS)
    
    print("\n--- Best Solution Found ---")
    if best_found_solution and best_found_solution.makespan != float('inf'):
        print(f"Makespan: {best_found_solution.makespan}")
        print("Operation Sequence (Job, Op):")
        for op in best_found_solution.operation_sequence:
            print(f"  ({op.job_idx}, {op.op_idx_in_job}) -> M{best_found_solution.machine_assignment[best_found_solution.operation_sequence.index(op)]}")
        
        print("\nSchedule Details (Operation: Start, End, Machine):")
        for op, details in sorted(best_found_solution.schedule_details.items(), key=lambda item: item[1]['start_time']):
            print(f"  Op({op.job_idx},{op.op_idx_in_job}): Start={details['start_time']}, End={details['end_time']}, Machine={details['machine']}")
        
        print("\nMachine Schedules:")
        for m_idx in range(problem_instance.num_machines):
            print(f"  Machine {m_idx}:")
            if best_found_solution.machine_schedules[m_idx]:
                for op_sched_detail in sorted(best_found_solution.machine_schedules[m_idx]): # Sort by start time
                    print(f"    Op({op_sched_detail[2]},{op_sched_detail[3]}) from {op_sched_detail[0]} to {op_sched_detail[1]}")
            else:
                print("    (idle)")
    else:
        print("No valid solution found or algorithm did not complete as expected.")

    print("\nDebugging test completed.")

