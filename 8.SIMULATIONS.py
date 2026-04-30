import numpy as np
from Warehouses import return_warehouse
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import pygame
import time
#-----------------------------------------------------------------------------------------------------#
print("\n\nWelcome to the warehouse simulation!")
print("You can adjust parameters for the simulation or press enter to use defaults.\n\n")
warehouse, size = return_warehouse(input("Enter warehouse (A,B,C,D,E,F) (default D ): ") or 'D')
N_ROBOTS = int(input("Enter number of robots (default 3): ") or 3)
num_episodes = int(input("Enter number of episodes (default 5000): ") or 5000)
num_steps = int(input("Enter steps per episode (default 500): ") or 500)
epsilon_start = float(input("Enter Epsilon (default 0.9): ") or 0.9)
discount_factor = float(input("Enter discount factor (default 0.99): ") or 0.99)
learning_rate = float(input("Enter learning rate (default 0.9): ") or 0.9)
min_epsilon = float(input("Enter minimum epsilon (default 0.1): ") or 0.1)
epsilon_decay = float(input("Enter epsilon decay (default linear): ") or (1/num_episodes))
simulations = int(input("Enter number of simulations to run (default 10): ") or 10)
max_number_of_targets = int(input("Enter maximum number of targets at a time (default 4): ") or 4)
#-----------------------------------------------------------------------------------------------------#
for i in range(size):
    for j in range(size):
        if warehouse[i,j] == 3:
            warehouse[i,j] = 1
cmap = colors.ListedColormap(['white', 'red', 'black'])
fig, ax = plt.subplots(figsize = (10,10))
ax.imshow(warehouse[:,:], cmap=cmap)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig('7.Plain_Warehouse.png', bbox_inches = 'tight', pad_inches = 0)
plt.close(fig)
actions = ['up', 'right', 'down', 'left']
num_actions = len(actions)
#-----------------------------------------------------------------------------------------------------#
valid_robot_starting_squares = []
valid_target_starting_squares = []
for i in range(size):
    for j in range(size):
        if warehouse[i,j] == 0: #empty cell
            valid_robot_starting_squares.append((i,j))
        elif warehouse[i,j] == 1:
            valid_target_starting_squares.append((i,j))
#-----------------------------------------------------------------------------------------------------#
def get_next_location(i,j,action_index):
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
#-----------------------------------------------------------------------------------------------------#
class Target:
    def __init__(self):
        self.active = False
        self.best_actions = np.zeros((size, size), dtype=int)
        self.V_table = np.zeros((size,size))
    def initialize(self, location):
        self.active = True
        self.location = location
    def remove(self):
        self.active = False
        self.V_table = np.zeros ((size, size))
        self.best_actions = np.zeros((size, size), dtype=int)
#-----------------------------------------------------------------------------------------------------#
def learn(target_location):
    epsilon = epsilon_start
    q_values = np.zeros((size, size, num_actions))
    rewards = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i,j) == target_location:
                rewards[i,j] = 100
            elif warehouse[i,j] == 0:
                rewards[i,j] = -1
            else:
                rewards[i,j] = -100
    for episode in range(num_episodes):
        i,j = random.choice(valid_robot_starting_squares)
        for step in range(num_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(q_values[i,j,:])
            old_i, old_j = i,j
            i,j = get_next_location(i,j,action)
            reward = rewards[i,j]
            old_q = q_values[old_i, old_j, action]
            best_future = np.max(q_values[i,j,:])
            q_values[old_i, old_j, action] = (old_q +
                                              learning_rate *
                                              (
                                                  reward + 
                                                  discount_factor *
                                                  best_future -
                                                  old_q
                                              ))
            if(i,j) == target_location:
                break
        epsilon = max(min_epsilon, epsilon-epsilon_decay)
    best_actions = np.zeros((size, size), dtype=int)
    V_table = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            best_actions[i,j] = np.argmax(q_values[i,j,:])
            V_table[i,j] = np.max(q_values[i,j,:])
    return best_actions, V_table
#-----------------------------------------------------------------------------------------------------#
def assignment(target_list, robot_current_locations):
    n_robots = len(robot_current_locations)
    n_targets = len(target_list)#target_list is all ACTIVE targets
    assignments = [None]*n_robots #default wait
    if n_targets == 0:
        return assignments
    option_values = np.zeros((n_robots, n_targets)) #evaluate every combo
    for r in range(n_robots):
        r_loc = robot_current_locations[r]
        for t in range(n_targets):
            option_values[r,t] = target_list[t].V_table[r_loc]
    if n_robots <= 3:
        best_total = -np.inf
        best_choice = None
        if n_robots == 1:
            t0 = np.argmax(option_values[0])
            assignments[0] = target_list[t0]
            return assignments
        if n_robots == 2:
            for t0 in range(n_targets):
                for t1 in range(n_targets):
                    if t1 == t0:
                        continue
                    total = option_values[0,t0] + option_values[1,t1]
                    if total > best_total:
                        best_total = total
                        best_choice = (t0, t1)
            if best_choice is not None:
                assignments[0] = target_list[best_choice[0]]
                assignments[1] = target_list[best_choice[1]]
            return assignments
        if n_robots == 3:
            for t0 in range(n_targets):
                for t1 in range(n_targets):
                    if t1 == t0:
                        continue
                    for t2 in range(n_targets):
                        if t2 == t0 or t2 == t1:
                            continue
                        total = (
                            option_values[0,t0] + 
                            option_values[1,t1] +
                            option_values[2,t2]
                        )
                        if total > best_total:
                            best_total = total
                            best_choice = (t0, t1, t2)
            if best_choice is not None:
                assignments[0] = target_list[best_choice[0]]
                assignments[1] = target_list[best_choice[1]]
                assignments[2] = target_list[best_choice[2]]
            return assignments
    else:  # n_robots > 3
        row_average = []
        best_choices = []
        for r in range(n_robots):
            row_average.append(np.average(option_values[r,:]))
            best_choices.append(np.argmax(option_values[r,:]))
        robot_order = np.argsort(row_average)
        for r in robot_order:
            t = best_choices[r]
            while target_list[t] in assignments:
                option_values[r,t] = -np.inf
                t = np.argmax(option_values[r,:])
            assignments[r] = target_list[t]
        return assignments
#-----------------------------------------------------------------------------------------------------#
def run_experiment():
    target_list = [Target() for _ in range(max_number_of_targets)]

    active_target_indices = random.sample(range(len(valid_target_starting_squares)), max_number_of_targets) # initialize 3 at start always

    for i in range(max_number_of_targets):
        target_list[i].initialize(valid_target_starting_squares[active_target_indices[i]])

    robot_indices = random.sample(range(len(valid_robot_starting_squares)), N_ROBOTS)
    robot_current_locations = [
        valid_robot_starting_squares[idx] for idx in robot_indices
    ]
    start_time = time.time()

    for target in target_list:
        if target.active:
            target.best_actions, target.V_table = learn(target.location)
    robot_assignments = assignment(
        [t for t in target_list if t.active],
        robot_current_locations
    )
    #simulation results:
    mean_path_lengths_ps = 0 #storing path length in steps
    time_taken_ps = [] #storing time in seconds

    steps_per_robots  = [[] for _ in range(N_ROBOTS)]
    step_count_per_robots = [0 for _ in range(N_ROBOTS)]
    #simulation begins
    terminal_robots = [False] * N_ROBOTS
    paths = [[] for _ in range(N_ROBOTS)]
    target_paths = [[] for _ in range(max_number_of_targets)]

    max_steps = 1000
    step_counter = 0
    targets_found = 0
    total_targets_to_collect = 10
    collisions = 0

    while (not all(terminal_robots) or targets_found < total_targets_to_collect) and step_counter < max_steps:
        for i, target in enumerate(target_list):
            if target.active:
                target_paths[i].append(target.location)
            else:
                target_paths[i].append((0,0))
        
        for i, robot_loc in enumerate(robot_current_locations):
            paths[i].append(robot_loc)
        
        for idx in range(N_ROBOTS):
            if terminal_robots[idx]:
                continue
            target = robot_assignments[idx]
            if target is None:
                terminal_robots[idx] = True
                continue
            if robot_current_locations[idx] == target.location:
                terminal_robots[idx] = True
                targets_found += 1
                print(f"robot {idx} found target {target.location} (total found: {targets_found}), (collisions: {collisions})")
                target.remove()
                robot_assignments = assignment(
                        [t for t in target_list if t.active],
                        robot_current_locations
                    )
                terminal_robots = [False] * N_ROBOTS #forces premature termination
                robot_current_locations[idx] = paths[idx][-2]
                steps_per_robots[idx].append(step_count_per_robots[idx])
                step_count_per_robots[idx] = 0
            else:
                action = target.best_actions[robot_current_locations[idx]]
                robot_current_locations[idx] = get_next_location(
                    robot_current_locations[idx][0],
                    robot_current_locations[idx][1],
                    action
                )
        collisions += len(robot_current_locations) - len(set(robot_current_locations))

        for target in target_list:
            if not target.active and targets_found < total_targets_to_collect:
                if np.random.rand() < 1:
                    loc = random.choice(valid_target_starting_squares)
                    target.initialize(loc)
                    target.best_actions, target.V_table = learn(target.location)
                    robot_assignments = assignment(
                        [t for t in target_list if t.active],
                        robot_current_locations
                    )
                    terminal_robots = [False] * N_ROBOTS
                    print(f"New target spawned at {loc}")
        
        active_targets = sum(t.active for t in target_list)
        print(f"Step {step_counter}: active targets = {active_targets}")
        step_counter += 1
        for i in range (N_ROBOTS):
            step_count_per_robots[i] += 1
    end_time = time.time()
    print("Simulation finished")
    print("targets found:", targets_found)
    print("time passed:", end_time - start_time)
    sums = 0
    for i in range(N_ROBOTS):
        average_steps = [0 for _ in range(N_ROBOTS)]
        for j in range(len(steps_per_robots[i])):
            average_steps[i] += steps_per_robots[i][j]
        if len(steps_per_robots[i]) != 0:
            average_steps[i] = average_steps[i]/len(steps_per_robots[i])
        else:
            average_steps[i] = 0
        print("Robot ",i," average steps between targets: ", steps_per_robots[i], "average: ", average_steps[i])
        sums += average_steps[i]
    mean_path_lengths_ps=(sums/N_ROBOTS)
    print("Average per simulation: ", mean_path_lengths_ps)
    time_taken_ps.append(end_time - start_time)
    print("Time taken per simulation: ", time_taken_ps)

print("\n\nRunning simulations...")
for sim in range(simulations):
    print(f"\n\n--- Simulation {sim+1} ---")
    run_experiment()