import numpy as np
import random
import copy
from sklearn.cluster import KMeans
from src.problems.problem_instance import Operation, ProblemInstance, Solution
from src.algorithms.decoder import decode_solution

class IAOAGNSAlgorithm:
    """
    IAOA+GNS Algorithm Implementation for POFJSP
    
    This class implements the Improved Adaptive Optimization Algorithm with
    Grade Neighborhood Search for Partially Ordered Flexible Job Shop Scheduling.
    """
    
    # --- Configuration & Constants ---
    MOA_MIN = 0.2
    MOA_MAX = 1.0
    
    def __init__(self, pop_size=80, max_iterations=60):
        """
        Initialize IAOA+GNS algorithm parameters.
        
        Args:
            pop_size: Population size (default: 80 from paper)
            max_iterations: Maximum iterations (default: 60 from paper)
        """
        self.pop_size = pop_size
        self.max_iterations = max_iterations
    
    def solve(self, problem, verbose=False):
        """
        Solve the POFJSP instance using IAOA+GNS.
        
        Args:
            problem: ProblemInstance object
            verbose: Enable verbose logging
            
        Returns:
            Solution object with the best found solution
        """
        return self._iaoa_gns_algorithm(problem, self.pop_size, self.max_iterations, verbose)
    
    def _iaoa_gns_algorithm(self, problem, pop_size, max_iterations, verbose=False):
        """
        Main IAOA+GNS Algorithm implementation.
        """
        # Debug: Print algorithm parameters
        if verbose:
            print(f"Starting IAOA+GNS algorithm with pop_size={pop_size}, max_iterations={max_iterations}")
            print(f"Problem: {problem.num_jobs} jobs, {problem.num_machines} machines, {problem.total_operations} operations")
        
        # Initialize population
        if verbose:
            print("Initializing population...")
        population = self._initialize_population(pop_size, problem)
        
        # Find initial best solution
        best_solution_overall = min(population, key=lambda s: s.makespan)
        if verbose:
            print(f"Initial best solution makespan: {best_solution_overall.makespan}")
        
        for t in range(max_iterations):
            # Update MOA
            moa = self.MOA_MIN + t * ((self.MOA_MAX - self.MOA_MIN) / max_iterations)
            
            new_population = []
            
            # For 2D clustering crossover, we need DVPC which depends on best solution's schedule
            if not best_solution_overall.schedule_details: # Ensure it's decoded
                if verbose:
                    print("Decoding best solution (needed for DVPC calculation)")
                decode_solution(best_solution_overall, problem)

            # Debug: Print iteration status
            if verbose:
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
                        if verbose:
                            print(f"  Solution {i}: Using 2D clustering crossover")
                        offspring_sol = self._two_d_clustering_crossover(current_sol, parent2, population, best_solution_overall, problem, verbose)
                    else: # Effective parallel mutation
                        # Debug: Print mutation info
                        if verbose:
                            print(f"  Solution {i}: Using effective parallel mutation")
                        offspring_sol = self._effective_parallel_mutation(current_sol, problem, verbose=verbose)
                else: # Development Phase (GNS)
                    # Identify bottlenecks based on current best_solution_overall or current_sol
                    # Paper implies GNS is on the current solution being processed.
                    # Let's use current_sol's bottlenecks for its GNS.
                    # For GNS, we need the schedule of the solution being improved.
                    if not current_sol.schedule_details: 
                        # Debug: Print decoding for GNS
                        if verbose:
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
                        if verbose:
                            print(f"  Solution {i}: Bottleneck job={current_sol_bottleneck_job_idx}, machine={current_sol_bottleneck_machine_idx}")

                    if r3 > 0.5: # Bottleneck job GNS
                        if current_sol_bottleneck_job_idx != -1:
                            # Debug: Print GNS job info
                            if verbose:
                                print(f"  Solution {i}: Using job GNS on job {current_sol_bottleneck_job_idx}")
                            offspring_sol = self._grade_neighborhood_search(current_sol, "job", current_sol_bottleneck_job_idx, problem, verbose)
                        else: # Fallback if no bottleneck job found
                            if verbose:
                                print(f"  Solution {i}: No bottleneck job found, using copy")
                            offspring_sol = copy.deepcopy(current_sol)
                    else: # Bottleneck machine GNS
                        if current_sol_bottleneck_machine_idx != -1:
                            # Debug: Print GNS machine info
                            if verbose:
                                print(f"  Solution {i}: Using machine GNS on machine {current_sol_bottleneck_machine_idx}")
                            offspring_sol = self._grade_neighborhood_search(current_sol, "machine", current_sol_bottleneck_machine_idx, problem, verbose)
                        else: # Fallback
                            if verbose:
                                print(f"  Solution {i}: No bottleneck machine found, using copy")
                            offspring_sol = copy.deepcopy(current_sol)
                
                # Elitism: Compare offspring with current_sol
                if offspring_sol.makespan < current_sol.makespan:
                    if verbose:
                        print(f"  Solution {i}: Improvement! {current_sol.makespan} -> {offspring_sol.makespan}")
                    new_population.append(offspring_sol)
                else:
                    if verbose:
                        print(f"  Solution {i}: No improvement. {current_sol.makespan} -> {offspring_sol.makespan}")
                    new_population.append(current_sol)

            population = new_population
            current_best_in_pop = min(population, key=lambda s: s.makespan)
            if current_best_in_pop.makespan < best_solution_overall.makespan:
                best_solution_overall = copy.deepcopy(current_best_in_pop) # Deepcopy to save state
                if verbose:
                    print(f"  New overall best solution found! Makespan: {best_solution_overall.makespan}")

        if verbose:
            print(f"Algorithm completed. Final best makespan: {best_solution_overall.makespan}")
        return best_solution_overall
    
    # --- 2. Solution Representation and Initialization ---
    def _get_topological_sort_operations(self, problem):
        """Generates one valid topological sort of all operations."""
        # Kahn's algorithm for topological sort
        in_degree = {op: 0 for op in problem.all_operations}
        adj = {op: [] for op in problem.all_operations}

        for op, preds in problem.predecessors_map.items():
            in_degree[op] = len(preds)
            for pred_op in preds:
                if pred_op in adj: # Ensure pred_op is a valid operation
                     adj[pred_op].append(op)

        queue = [op for op in problem.all_operations if in_degree[op] == 0]
        random.shuffle(queue) # Add randomness
        
        topo_sorted_ops = []
        while queue:
            u = queue.pop(0)
            topo_sorted_ops.append(u)
            
            # Sort successors for deterministic behavior if multiple choices, or shuffle for randomness
            current_successors = list(adj[u])
            random.shuffle(current_successors) # For randomness

            for v_op in current_successors:
                in_degree[v_op] -= 1
                if in_degree[v_op] == 0:
                    queue.append(v_op)
        
        if len(topo_sorted_ops) != problem.total_operations:
            # Fallback: return a random shuffle, decoder will handle precedence issues (potentially poorly)
            if not topo_sorted_ops: # if cycle detected and no ops could be sorted
                temp_ops = list(problem.all_operations)
                random.shuffle(temp_ops)
                return temp_ops
            return topo_sorted_ops

        return topo_sorted_ops

    def _generate_random_machine_assignment(self, op_sequence, problem):
        """Generates random machine assignments for a sequence of operations."""
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

    def _initialize_population(self, pop_size, problem):
        """Initialize population with forward and backward methods."""
        population = []
        
        # Half forward, half backward (simplified)
        num_forward = pop_size // 2
        num_backward = pop_size - num_forward

        for _ in range(num_forward):
            # Generate a topologically plausible operation sequence
            op_sequence = self._get_topological_sort_operations(problem)
            if len(op_sequence) != problem.total_operations: # Fallback if topo sort fails
                op_sequence = list(problem.all_operations)
                random.shuffle(op_sequence)

            machine_assignment = self._generate_random_machine_assignment(op_sequence, problem)
            sol = Solution(op_sequence, machine_assignment)
            decode_solution(sol, problem) # Calculate initial makespan
            population.append(sol)

        for _ in range(num_backward):
            op_sequence_forward = self._get_topological_sort_operations(problem)
            if len(op_sequence_forward) != problem.total_operations: # Fallback
                op_sequence_forward = list(problem.all_operations)
                random.shuffle(op_sequence_forward)
            
            op_sequence_backward = op_sequence_forward[::-1] # Reverse
            machine_assignment = self._generate_random_machine_assignment(op_sequence_backward, problem)
            sol = Solution(op_sequence_backward, machine_assignment)
            decode_solution(sol, problem)
            population.append(sol)
            
        return population

    # --- 3. Exploration Phase Operators ---
    def _calculate_dvpc(self, solution, best_solution_schedule_details, problem):
        """Degree of Variance of Process Completion (DVPC)"""
        dvpc_val = 0
        if not best_solution_schedule_details: # If best solution hasn't been decoded
            return random.random() * 100 # or some other default large value

        for op, details in solution.schedule_details.items():
            if op in best_solution_schedule_details:
                dvpc_val += abs(details['end_time'] - best_solution_schedule_details[op]['end_time'])
        return dvpc_val

    def _calculate_woc(self, job_idx, schedule_details_for_job, problem):
        """Work Order Compactness (WOC)"""
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

    def _two_d_clustering_crossover(self, parent1_sol, parent2_sol, population, best_solution_overall, problem, verbose=False):
        """
        Two-dimensional clustering crossover operator.
        Combines job-level machine assignments based on WOC and DVPC metrics.
        """
        # Debug: Starting crossover
        if verbose:
            print("    Starting 2D clustering crossover")
        
        # Create a copy of parent1 as the base for the offspring
        offspring_op_seq = parent1_sol.operation_sequence.copy()
        offspring_ma_seq = parent1_sol.machine_assignment.copy()
        
        # Calculate DVPC for both parents
        parent1_dvpc = self._calculate_dvpc(parent1_sol, best_solution_overall.schedule_details, problem)
        parent2_dvpc = self._calculate_dvpc(parent2_sol, best_solution_overall.schedule_details, problem)
        
        # Create operation-to-machine mapping for both parents
        p1_op_to_machine = {op: parent1_sol.machine_assignment[i] for i, op in enumerate(parent1_sol.operation_sequence)}
        p2_op_to_machine = {op: parent2_sol.machine_assignment[i] for i, op in enumerate(parent2_sol.operation_sequence)}
        
        # For each job, decide whether to use parent1 or parent2's machine assignment
        for job_idx in range(problem.num_jobs):
            # Get operations for this job
            job_ops = [op for op in problem.all_operations if op.job_idx == job_idx]
            
            # Skip if no operations for this job
            if not job_ops:
                continue
            
            # Calculate WOC for this job from both parents
            p1_job_schedule = {op: parent1_sol.schedule_details.get(op, None) for op in job_ops if op in parent1_sol.schedule_details}
            p2_job_schedule = {op: parent2_sol.schedule_details.get(op, None) for op in job_ops if op in parent2_sol.schedule_details}
            
            # If either parent doesn't have schedule details for this job, use the other parent
            if not p1_job_schedule:
                if p2_job_schedule:
                    # Use parent2's machine assignment for this job
                    for op in job_ops:
                        if op in p2_op_to_machine:
                            idx = offspring_op_seq.index(op)
                            offspring_ma_seq[idx] = p2_op_to_machine[op]
                continue
            
            if not p2_job_schedule:
                # Already using parent1's machine assignment
                continue
            
            # Calculate WOC for both parents
            p1_woc = self._calculate_woc(job_idx, p1_job_schedule, problem)
            p2_woc = self._calculate_woc(job_idx, p2_job_schedule, problem)
            
            # Compare WOC and DVPC to decide which parent to use for this job
            if p1_woc <= p2_woc and parent1_dvpc <= parent2_dvpc:
                # Use parent1 (already the default)
                pass
            elif p1_woc > p2_woc and parent1_dvpc > parent2_dvpc:
                # Use parent2
                for op in job_ops:
                    if op in p2_op_to_machine:
                        idx = offspring_op_seq.index(op)
                        offspring_ma_seq[idx] = p2_op_to_machine[op]
            else:
                # Conflicting indicators, use the better WOC
                if p2_woc < p1_woc:
                    for op in job_ops:
                        if op in p2_op_to_machine:
                            idx = offspring_op_seq.index(op)
                            offspring_ma_seq[idx] = p2_op_to_machine[op]
        
        # Create and decode the offspring solution
        offspring_sol = Solution(offspring_op_seq, offspring_ma_seq)
        decode_solution(offspring_sol, problem, verbose)
        
        if verbose:
            print(f"    Crossover complete, offspring makespan: {offspring_sol.makespan}")
        
        return offspring_sol

    def _effective_parallel_mutation(self, solution, problem, mutation_rate=0.2, verbose=False):
        """
        Effective parallel mutation operator.
        
        This mutation has two components:
        1. OS mutation: Swap operations while preserving precedence constraints
        2. MA mutation: Change machine assignments for selected operations
        """
        # Debug: Starting mutation
        if verbose:
            print("    Starting effective parallel mutation")
        
        # Create a copy of the solution
        offspring_op_seq = solution.operation_sequence.copy()
        offspring_ma_seq = solution.machine_assignment.copy()
        
        # 1. OS Mutation (Operation Sequence)
        # We'll use a precedence-preserving swap mutation
        
        # First, get a valid topological ordering to ensure we maintain precedence
        topo_sort = self._get_topological_sort_operations(problem)
        
        # Create a mapping from operations to their positions in the topological sort
        topo_positions = {op: i for i, op in enumerate(topo_sort)}
        
        # Determine number of swaps based on mutation rate
        num_swaps = max(1, int(len(offspring_op_seq) * mutation_rate * 0.5))
        
        if verbose:
            print(f"    Performing {num_swaps} operation sequence swaps")
        
        for _ in range(num_swaps):
            # Select a random position
            pos1 = random.randint(0, len(offspring_op_seq) - 1)
            op1 = offspring_op_seq[pos1]
            
            # Find a valid swap partner that preserves precedence
            # We'll try up to 10 times to find a valid swap
            valid_swap_found = False
            for _ in range(10):  # Try 10 times
                pos2 = random.randint(0, len(offspring_op_seq) - 1)
                if pos1 == pos2:
                    continue
                    
                op2 = offspring_op_seq[pos2]
                
                # Check if swapping these operations would violate precedence
                # If op1 must come before op2 in topo_sort, we can't swap
                # If op2 must come before op1 in topo_sort, we can't swap
                if topo_positions[op1] < topo_positions[op2] and pos1 > pos2:
                    # Can't swap: op1 should be before op2 but is after
                    continue
                if topo_positions[op2] < topo_positions[op1] and pos2 > pos1:
                    # Can't swap: op2 should be before op1 but is after
                    continue
                    
                # Swap is valid
                offspring_op_seq[pos1], offspring_op_seq[pos2] = offspring_op_seq[pos2], offspring_op_seq[pos1]
                offspring_ma_seq[pos1], offspring_ma_seq[pos2] = offspring_ma_seq[pos2], offspring_ma_seq[pos1]
                valid_swap_found = True
                break
                
            if not valid_swap_found and verbose:
                print(f"    Could not find valid swap for position {pos1}")
        
        # 2. MA Mutation (Machine Assignment)
        # Change machine assignments for some operations
        
        # Determine number of machine changes
        num_changes = max(1, int(len(offspring_ma_seq) * mutation_rate))
        
        if verbose:
            print(f"    Performing {num_changes} machine assignment changes")
        
        for _ in range(num_changes):
            # Select a random operation
            pos = random.randint(0, len(offspring_op_seq) - 1)
            op = offspring_op_seq[pos]
            current_machine = offspring_ma_seq[pos]
            
            # Get valid alternative machines
            valid_machines = []
            for m in range(problem.num_machines):
                if m != current_machine and problem.processing_times[op.job_idx][op.op_idx_in_job, m] != np.inf:
                    valid_machines.append(m)
            
            if valid_machines:
                # Choose a new machine with preference for faster ones
                machine_times = [(problem.processing_times[op.job_idx][op.op_idx_in_job, m], m) for m in valid_machines]
                machine_times.sort()  # Sort by processing time
                
                # Select with bias toward faster machines
                if random.random() < 0.7:  # 70% chance to pick the fastest
                    new_machine = machine_times[0][1]
                else:
                    new_machine = random.choice(valid_machines)
                    
                offspring_ma_seq[pos] = new_machine
                if verbose:
                    print(f"    Changed machine for op {op} from {current_machine} to {new_machine}")
            elif verbose:
                print(f"    No valid alternative machines for op {op}")
        
        # Create and evaluate the offspring
        offspring_sol = Solution(offspring_op_seq, offspring_ma_seq)
        decode_solution(offspring_sol, problem, verbose)
        
        if verbose:
            print(f"    Mutation complete, offspring makespan: {offspring_sol.makespan}")
        
        return offspring_sol

    # --- 4. Development Phase Operators (GNS) ---
    def _get_bottlenecks(self, population, problem): # Pass population of Solution objects
        """Identify bottleneck job and machine from the best solution in population."""
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

    def _get_operation_priority(self, op, problem): # op is Operation(job_idx, op_idx_in_job)
        """Determine operation priority for GNS based on successor count."""
        # G1 (High): Multiple successors
        # G2 (Medium): One successor
        # G3 (Low): No successors (last op of a job)
        num_successors = 0
        if op in problem.successors_map:
            num_successors = len(problem.successors_map[op])

        if num_successors > 1: return "G1"
        if num_successors == 1: return "G2"
        return "G3" # num_successors == 0

    def _grade_neighborhood_search(self, solution, bottleneck_type, bottleneck_id, problem, verbose=False):
        """
        Grade Neighborhood Search (GNS) for local improvement.
        
        Args:
            bottleneck_type: "job" or "machine"
            bottleneck_id: job_idx or machine_idx
        """
        if verbose:
            print(f"    Starting GNS for {bottleneck_type} {bottleneck_id}")
        
        # IMPROVED APPROACH: Ensure topological ordering is maintained
        # First, create a new solution that respects topological ordering
        topo_sort = self._get_topological_sort_operations(problem)
        
        # Create mappings from operations to machine assignments
        op_to_machine = {}
        for i, op in enumerate(solution.operation_sequence):
            op_to_machine[op] = solution.machine_assignment[i]
        
        # Create a new solution that follows topological ordering
        gns_op_seq = topo_sort.copy()
        gns_ma_seq = [op_to_machine.get(op, self._generate_random_machine_assignment([op], problem)[0]) for op in gns_op_seq]
        
        gns_sol = Solution(gns_op_seq, gns_ma_seq)
        
        # First, decode the solution to get the schedule
        decode_solution(gns_sol, problem, verbose)
        
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
            if verbose:
                print(f"    No operations found for {bottleneck_type} {bottleneck_id}")
            return solution  # Return original solution if no operations found

        # Prioritize operations
        for item in operations_to_consider:
            if item['details']: # Only if op was scheduled
                item['priority_val'] = {"G1": 1, "G2": 2, "G3": 3}[self._get_operation_priority(item['op_obj'], problem)]
            else: # Should not happen if solution is decoded
                item['priority_val'] = 3 

        # Sort by priority (G1 highest, so lower val is better)
        operations_to_consider.sort(key=lambda x: x['priority_val'])
        
        # Select 10% of operations using roulette wheel (simplified: select top 10% by priority)
        num_to_select = max(1, int(len(operations_to_consider) * 0.1))
        selected_ops_for_gns = operations_to_consider[:num_to_select]
        
        if verbose:
            print(f"    Selected {len(selected_ops_for_gns)} operations for GNS")

        for op_info in selected_ops_for_gns:
            op_obj = op_info['op_obj']
            seq_idx = op_info['seq_idx']
            op_details = op_info['details']

            if not op_details: 
                if verbose:
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
                    if verbose:
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
                        if verbose:
                            print(f"    Moving op {op_obj} to less loaded machine {less_loaded_m_idx}")
                        break # Take the first valid less loaded one

        decode_solution(gns_sol, problem, verbose)
        if verbose:
            print(f"    GNS complete, new makespan: {gns_sol.makespan}")
        return gns_sol


def main():
    """Main function for testing IAOA+GNS algorithm."""
    import sys
    import os
    from pathlib import Path
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from problems.problem_instance import ProblemInstance, Operation
    import numpy as np
    
    # Create a simple test problem
    print("Creating sample POFJSP problem...")
    num_operations_per_job = [2, 2]  # 2 jobs, 2 operations each
    processing_times = [
        np.array([[3, 5], [6, np.inf]]),  # Job 0: Op0 on M0=3,M1=5; Op1 on M0=6,M1=inf
        np.array([[4, 2], [np.inf, 7]])   # Job 1: Op0 on M0=4,M1=2; Op1 on M0=inf,M1=7
    ]
    
    # Simple precedence: op0 -> op1 for each job
    predecessors_map = {
        Operation(0, 1): {Operation(0, 0)},
        Operation(1, 1): {Operation(1, 0)}
    }
    successors_map = {
        Operation(0, 0): {Operation(0, 1)},
        Operation(1, 0): {Operation(1, 1)}
    }
    
    problem = ProblemInstance(
        num_jobs=2,
        num_machines=2,
        num_operations_per_job=num_operations_per_job,
        processing_times=processing_times,
        predecessors_map=predecessors_map,
        successors_map=successors_map
    )
    
    # Run IAOA+GNS algorithm
    print("\nRunning IAOA+GNS algorithm...")
    algorithm = IAOAGNSAlgorithm(pop_size=20, max_iterations=10)  # Small params for testing
    solution = algorithm.solve(problem, verbose=True)
    
    if solution:
        print(f"\n[SUCCESS] Algorithm completed successfully!")
        print(f"Final makespan: {solution.makespan}")
        print(f"Operation sequence: {solution.operation_sequence}")
        print(f"Machine assignment: {solution.machine_assignment}")
    else:
        print("❌ Algorithm failed to find a solution")


if __name__ == "__main__":
    main()