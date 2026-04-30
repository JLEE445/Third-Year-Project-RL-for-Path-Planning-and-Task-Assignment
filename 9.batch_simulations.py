# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import random
import time
from Warehouses import return_warehouse 



# ============================================================
# GLOBAL SETTINGS
# ============================================================
actions = ['up', 'right', 'down', 'left']
num_actions = len(actions)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def get_next_location(i,j,action_index,size):
    if actions[action_index] == 'up' and i>0:
        return i-1, j
    elif actions[action_index] == 'right' and j < size - 1:
        return i, j+1
    elif actions[action_index] == 'down' and i < size-1:
        return i+1, j
    elif actions[action_index] == 'left' and j > 0:
        return i, j-1
    else:
        return i,j
# ============================================================
# CLASSES
# ============================================================

class Target:
    def __init__(self,size):
        self.active = False
        self.size = size
        self.best_actions = np.zeros((size, size), dtype=int)
        self.V_table = np.zeros((size,size))
    def initialize(self, location):
        self.active = True
        self.location = location
    def remove(self):
        self.active = False
        self.V_table = np.zeros((self.size, self.size))
        self.best_actions = np.zeros((self.size, self.size), dtype=int)

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

def prepare_environment(config):
    #load warehouse
    warehouse, size = return_warehouse(config["warehouse_name"])

    # clean warehouse 
    for i in range(size):
        for j in range(size):
            if warehouse[i][j] == 3:
                warehouse[i][j] = 1

    #compute valid_robot_positions
    #compute valid_target_positions
    valid_robot_positions = []
    valid_target_positions = []
    for i in range(size):
        for j in range(size):
            if warehouse[i,j] == 0: #empty cell
                valid_robot_positions.append((i,j))
            elif warehouse[i,j] == 1: #shelf cell
                valid_target_positions.append((i,j))

    return warehouse, size, valid_robot_positions, valid_target_positions


# ============================================================
# LEARNING (Q-LEARNING)
# ============================================================

def learn_target_policy(config, target_location, environment_data):
    epsilon = config["epsilon_start"]
    size = environment_data["size"]
    warehouse = environment_data["warehouse"]
    learning_rate = config["learning_rate"]
    discount_factor = config["discount_factor"]
    min_epsilon = config["min_epsilon"]
    epsilon_decay = config["epsilon_decay"]
    q_values = np.zeros((size, size, num_actions))
    # Define rewards (-1 for each step, 100 for reaching target, -100 for collision with wall)
    rewards = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i,j) == target_location:
                rewards[i][j] = 100
            elif warehouse[i,j] == 0: # empty cell
                rewards[i][j] = -1
            else: # wall or shelf
                rewards[i][j] = -100
    num_episodes = config["num_episodes"]
    num_steps = config["num_steps"]
    for episode in range(num_episodes):
        i,j = random.choice(environment_data["valid_robot_positions"])
        for step in range(num_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)  # Explore
            else:
                action = np.argmax(q_values[i][j])  # Exploit
            old_i, old_j = i, j
            i, j = get_next_location(i, j, action, size)
            reward = rewards[i][j]
            old_q = q_values[old_i][old_j][action]
            best_future = np.max(q_values[i][j])
            q_values[old_i][old_j][action] = (old_q
                                               + learning_rate * 
                                               (
                                                   reward + 
                                                   discount_factor * best_future
                                                     - old_q
                                                ))
            if(i,j) == target_location:
                break
        # Decay epsilon
        
        
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    
    best_actions = np.zeros((size, size), dtype=int)
    V_table = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            best_actions[i][j] = np.argmax(q_values[i][j])
            V_table[i][j] = np.max(q_values[i][j])


    return best_actions, V_table#, training_history


# ============================================================
# ASSIGNMENT STRATEGIES
# ============================================================

def assignment_strategy(targets, robot_locations, config):
    
    strategy = config["assignment_strategy"]

    if strategy == "greedy":
        n_r = len(robot_locations)
        n_t = len(targets)
        assignments = [None] * n_r

        if n_t == 0:
            return assignments

        values = np.array([
            [targets[t].V_table[robot_locations[r]] for t in range(n_t)]
            for r in range(n_r)
        ])

        robot_order = np.argsort(values.mean(axis=1))[::-1]   # highest first
        taken = set()

        for r in robot_order:
            for t in np.argsort(values[r])[::-1]:             # best target first
                if t not in taken:
                    assignments[r] = targets[t]
                    taken.add(t)
                    break

    elif strategy == "random":
        n_robots = len(robot_locations)
        n_targets = len(targets)

        assignments = [None] * n_robots

        # Randomly permute robot indices
        robot_indices = list(range(n_robots))
        random.shuffle(robot_indices)

        # Randomly choose targets (no duplicates)
        chosen_targets = random.sample(targets, min(n_robots, n_targets))

        # Assign targets to random robots
        for i in range(len(chosen_targets)):
            assignments[robot_indices[i]] = chosen_targets[i]

    elif strategy == "global_optimum":
        n_robots = len(robot_locations)
        n_targets = len(targets)

        assignments = [None] * n_robots

        if n_targets == 0:
            return assignments

        option_values = np.zeros((n_robots, n_targets))

        for r in range(n_robots):
            r_loc = robot_locations[r]
            for t in range(n_targets):
                option_values[r, t] = targets[t].V_table[r_loc]

        # Only assign as many robots as we have targets for
        robots_to_assign = min(n_robots, n_targets)

        best_total = -np.inf
        best_choice = None

        if robots_to_assign == 1:
            for t0 in range(n_targets):
                total = option_values[0, t0]
                if total > best_total:
                    best_total = total
                    best_choice = (t0,)

        elif robots_to_assign == 2:
            for t0 in range(n_targets):
                for t1 in range(n_targets):
                    if t1 == t0:
                        continue
                    total = option_values[0, t0] + option_values[1, t1]
                    if total > best_total:
                        best_total = total
                        best_choice = (t0, t1)

        elif robots_to_assign == 3:
            for t0 in range(n_targets):
                for t1 in range(n_targets):
                    if t1 == t0:
                        continue
                    for t2 in range(n_targets):
                        if t2 == t0 or t2 == t1:
                            continue
                        total = (
                            option_values[0, t0]
                            + option_values[1, t1]
                            + option_values[2, t2]
                        )
                        if total > best_total:
                            best_total = total
                            best_choice = (t0, t1, t2)

        else:
            raise ValueError("global_optimum currently only supports up to 3 robots")

        if best_choice is not None:
            for r, t_idx in enumerate(best_choice):
                assignments[r] = targets[t_idx]
    elif strategy == "fifo":
        n_robots = len(robot_locations)

        assignments = [None] * n_robots
        
        # Assign targets to random robots
        for i in range(min(n_robots, len(targets))):
            assignments[i] = targets[i]

    else:
        raise ValueError(f"Unknown assignment strategy: {strategy}")

    return assignments

# ============================================================
# SIMULATION LOOP (ONE RUN)
# ============================================================

def run_simulation(config):
    #runs ONE full simulation.
    # ---------------------------
    # Setup
    # ---------------------------
    set_seed(config["seed"])
    warehouse, size, valid_robot_positions, valid_target_positions = prepare_environment(config)
    environment_data = {
    "warehouse": warehouse,
    "size": size,
    "valid_robot_positions": valid_robot_positions,
    "valid_target_positions": valid_target_positions,
}
    n_robots = config["n_robots"]

    max_number_of_targets = config["max_number_of_targets"]
    target_list = [Target(size) for _ in range(max_number_of_targets)]

    initial_targets = min(3, max_number_of_targets, len(valid_target_positions))
    active_target_indices = random.sample(range(len(valid_target_positions)), initial_targets)

    for i in range(initial_targets):
        target_list[i].initialize(valid_target_positions[active_target_indices[i]])
    robot_indices = random.sample(range(len(valid_robot_positions)), n_robots)
    robot_current_locations = [
        valid_robot_positions[idx] for idx in robot_indices
    ]
    start_time = time.time()

    for target in target_list:
        if target.active:
            target.best_actions, target.V_table = learn_target_policy(config, target.location, environment_data)
    robot_assignments = assignment_strategy(
        [t for t in target_list if t.active],
        robot_current_locations,
        config
    )

    #simulation begins
    terminal_robots = [False] * n_robots
    paths = [[] for _ in range(n_robots)]
    target_paths = [[] for _ in range(max_number_of_targets)]
    mean_path_lengths = 0
    steps_per_robots = [[] for _ in range(n_robots)]
    step_count_per_robot = [0 for _ in range(n_robots)]
    max_steps = config["max_sim_steps"]
    step_counter = 0
    targets_found = 0
    total_targets_to_collect = config["total_targets_to_collect"]

    while (targets_found < total_targets_to_collect) and step_counter < max_steps:
        for i, target in enumerate(target_list):
            if target.active:
                target_paths[i].append(target.location)
            else:
                target_paths[i].append((0,0))
        
        for i, robot_loc in enumerate(robot_current_locations):
            paths[i].append(robot_loc)
        
        for idx in range(n_robots):
            if terminal_robots[idx]:
                continue
            target = robot_assignments[idx]
            if target is None:
                terminal_robots[idx] = True
                continue
            if robot_current_locations[idx] == target.location:
                terminal_robots[idx] = True
                targets_found += 1
                #print(f"robot {idx} found target {target.location} (total found: {targets_found}), (collisions: {collisions})")
                target.remove()
                robot_assignments = assignment_strategy(
                        [t for t in target_list if t.active],
                        robot_current_locations,
                        config
                    )
                terminal_robots = [False] * n_robots #forces premature termination
                robot_current_locations[idx] = paths[idx][-2]
                steps_per_robots[idx].append(step_count_per_robot[idx])
                step_count_per_robot[idx] = 0
            else:
                action = target.best_actions[robot_current_locations[idx]]
                old_pos = robot_current_locations[idx]
                new_pos = get_next_location(
                    old_pos[0], 
                    old_pos[1], 
                    action,
                    size
                )

                robot_current_locations[idx] = new_pos
                if new_pos != old_pos:
                    step_count_per_robot[idx] += 1
        
        for target in target_list:
            if not target.active and targets_found < total_targets_to_collect:
                if np.random.rand() < 0.2:
                    loc = random.choice(valid_target_positions)
                    target.initialize(loc)
                    target.best_actions, target.V_table = learn_target_policy(config, target.location, environment_data)
                    robot_assignments = assignment_strategy(
                        [t for t in target_list if t.active],
                        robot_current_locations,
                        config
                    )
                    terminal_robots = [False] * n_robots
                    #print(f"New target spawned at {loc}")
        
        active_targets = sum(t.active for t in target_list)
        #print(f"Step {step_counter}: active targets = {active_targets}")
        step_counter += 1
        

    #print("Simulation finished")
    #print("targets found:", targets_found)
    #print("time passed:", time.time() - start_time)
    end_time = time.time()
    sums = 0
    mean_distance_to_a_target = 0
    for i in range(n_robots):
        average_steps = [0 for _ in range(n_robots)]
        for j in range(len(steps_per_robots[i])):
            average_steps[i] += steps_per_robots[i][j]
            mean_distance_to_a_target += steps_per_robots[i][j]
        if len(steps_per_robots[i]) != 0:
            average_steps[i] = average_steps[i]/len(steps_per_robots[i])
        else:
            average_steps[i] = 0
        print("Robot ",i," average steps between targets: ", steps_per_robots[i], "average: ", average_steps[i])
        sums += average_steps[i]
    mean_path_lengths=(sums/n_robots)
    mean_distance_to_a_target = mean_distance_to_a_target/targets_found
    result = {
        # ----- CONFIG INFO -----
        "experiment_name": config["experiment_name"],
        "assignment_strategy": config["assignment_strategy"],
        "n_robots": config["n_robots"],
        "max_number_of_targets": config["max_number_of_targets"],
        "learning_rate": config["learning_rate"],
        "discount_factor": config["discount_factor"],
        "seed": config["seed"],

        # ----- METRICS -----
        
        "mean_path_length": mean_path_lengths,
        "mean_distance_to_a_target": mean_distance_to_a_target,
        "time_taken": end_time - start_time,
        "steps_taken": step_counter,
    }

    return result


# ============================================================
# BATCH EXPERIMENT RUNNER
# ============================================================

def run_batch(configs, output_file="results.csv"):
    #csv file
    results = []

    for i, config in enumerate(configs):
        print(f"Running experiment {i+1}/{len(configs)}")

        result = run_simulation(config)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"Saved results to {output_file}")

    return df


# ============================================================
# CONFIG GENERATION (EXPERIMENT SETUP)
# ============================================================

def generate_configs():
    

    configs = []

    #loop over parameters
    for strategy in ["random", "greedy", "fifo", "global_optimum"]:
        for n_robots in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for n_targets in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for seed in [1, 2, 3]:
                    if strategy == "global_optimum" and n_robots > 3:
                        continue # global optimum only implemented for 3 robots
                    num_episodes = 1000
                    config = {
                        "experiment_name": "main_experiment",
                        "warehouse_name": "D",
                        "assignment_strategy": strategy,
                        "n_robots": n_robots,
                        "max_number_of_targets": n_targets,
                        "total_targets_to_collect": 10,
                        "num_episodes": num_episodes,
                        "num_steps": 500,
                        "max_sim_steps": 1000,
                        "learning_rate": 0.9,
                        "discount_factor": 0.99,
                        "epsilon_start": 0.9,
                        "min_epsilon": 0.1,
                        "epsilon_decay": 1/num_episodes,
                        "seed": seed,
                    }

                    configs.append(config)

    return configs


# ============================================================
# ANALYSIS (OPTIONAL - CAN BE SEPARATE FILE)
# ============================================================




# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # ---------------------------
    # Generate experiment configs
    # ---------------------------
    configs = generate_configs()

    # ---------------------------
    # Run experiments
    # ---------------------------
    results_df = run_batch(configs, "results.csv")

    # ---------------------------
    # Analyse results
    # ---------------------------
    #analyse_results("results.csv")