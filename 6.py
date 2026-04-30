#Multi-agent Q-learning Simulation for a Warehouse Environment
#
#there are N_ROBOTS (3-6) (defined by user before start of simulation)
#at any given time there are N_TARGETS (varying between +- N_ROBOTS)
#(not defined by user, targets are initialised randomly during simulation)
#Simulation terminates once all targets have been reached
#^^throughout simulation, a maximum of 10 targets will appear
#
#
#This simulation encourages:
##correct task assignment to minimize length of robot journeys
##response to random changes in target locations - as if someone has updated the warehouse system with a new item for collection
##coordination between robots
#
#Initialization of the simulation:
##Three robots are initialzed
## - They are not Q-learning agents, a 'robot' is defined by its current location in the 'current_locations' array
## - The number of robots does not change
## - At the start of the simulation, random locations are assigned to robots from 'valid_robot_starting_squares' : a list of empty cell coordinates
## - Robots must all have unique starting locations
##Four targets are initialized
## - Targets are Q-learning agents, they learn the correct actions a robot must take to them
## - Each target is assigned a random location by 'valid_target_starting_squares' : a list of shelf cell coordinates
## - Targets must all have unique locations
## - When initialized, targets become active, when a robot locates the target (in the simulation, not in q-learning), the targets become inactive
## - When a target is initialized, it calls q-learning function to generate a policy table and a value table
## - Using the value tables, robots are assigned to targets based on the combination of assignments with the maximum average value across starting robot locations
#During the simulation:
## - Once initializing and initial assignment is complete, the simulation begins
## - Robots move towards assigned target following targets policy
## - When a robot's current location == a target's location, target is inactivated, triggering reassignment.
## - During reassignment, two functions are called:
#### - First: if N_TARGETS < 4 or N_TARGETS >= 2, there is a 10% chance with every new step in the simulation that a new target is activated, triggering reassignment
#### - Second: reassignment - every robot's current location is compared with target_values to reassign to optimal targets.
#### Note: it may be that agent C finds target first, makes it's way to another target, but it turns out agent B is closer, so it stops searching for its target, goes to other target and agent C goes to agent B's original target.
## - This repeats every step (if changes occur - must be triggered, otherwise wasting resources) UNTIL 10 targets have been activated
## - After this, no more targets are initialized ,and simulation continues until all targets are inactive
#Post simulation:
## - During simulation, seven arrays were appended at every step:
## - Current locations of robots
## - Current locations of targets (if inactive - target location = (0,0) and color = black (hidden in wall))
## - A pygame simulation is run, using these seven paths finish
## - Paths should all be same length, as each append is for each time step

import numpy as np
import pygame
import time
from Warehouses import return_warehouse
import random

# -----------------------
# Functions
# -----------------------

def get_next_location(i, j, action_index):
    if actions[action_index] == 'up' and i > 0:
        return i-1, j
    elif actions[action_index] == 'right' and j < size-1:
        return i, j+1
    elif actions[action_index] == 'down' and i < size-1:
        return i+1, j
    elif actions[action_index] == 'left' and j > 0:
        return i, j-1
    return i, j

def learn(target_location):
    epsilon = epsilon_start
    q_values = np.zeros((size, size, num_actions))
    rewards = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if (i, j) == target_location:
                rewards[i, j] = 100
            elif warehouse[i, j] == 0:
                rewards[i, j] = -1
            else:
                rewards[i, j] = -100

    for _ in range(num_episodes):
        i, j = random.choice(valid_robot_starting_cells)

        for _ in range(num_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(q_values[i, j])

            old_i, old_j = i, j
            i, j = get_next_location(i, j, action)

            reward = rewards[i, j]
            old_q = q_values[old_i, old_j, action]
            best_future = np.max(q_values[i, j])

            q_values[old_i, old_j, action] = old_q + learning_rate * (
                reward + discount_factor * best_future - old_q
            )

            if (i, j) == target_location:
                break

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return np.argmax(q_values, axis=2), np.max(q_values, axis=2)

def assignment(target_list, robot_current_locations):
    active_targets = [t for t in target_list if t.active]

    assignments = [None] * N_ROBOTS

    if len(active_targets) == 0:
        return assignments

    for r in range(N_ROBOTS):
        best_value = -np.inf
        best_target = None

        for t in active_targets:
            value = t.V_table[robot_current_locations[r]]
            if value > best_value:
                best_value = value
                best_target = t

        assignments[r] = best_target

    return assignments

# -----------------------
# Classes
# -----------------------

class Target:
    def __init__(self):
        self.active = False
        self.location = (0,0)
        self.best_actions = None
        self.V_table = None

    def initialize(self, location):
        self.active = True
        self.location = location
        self.best_actions, self.V_table = learn(location)

    def remove(self):
        self.active = False
        self.location = (0,0)
        self.best_actions = None
        self.V_table = None

# -----------------------
# Setup
# -----------------------

warehouse, size = return_warehouse(input("Enter warehouse (A-F): "))
actions = ['up','right','down','left']
num_actions = 4

num_episodes = int(input("Enter Number of Episodes (default 500): ") or 500)
num_steps = int(input("Enter Steps per Episode (default 300): ") or 300)
epsilon_start = float(input("Enter Epsilon (default 0.9): ") or 0.9)
discount_factor = float(input("Enter Discount Factor (default 0.99): ") or 0.99)
learning_rate = float(input("Enter Learning Rate (default 0.9): ") or 0.9)
min_epsilon = float(input("Enter Minimum Epsilon (default 0.1): ") or 0.1)
epsilon_decay = float(input("Enter Epsilon Decay (default 1/episodes): ") or (1 / num_episodes))

TOTAL_TARGETS = 10
N_ROBOTS = 3
N_TARGETS = 4

valid_robot_starting_cells = [(i,j) for i in range(size) for j in range(size) if warehouse[i,j]==0]
valid_target_cells = [(i,j) for i in range(size) for j in range(size) if warehouse[i,j]==1]

robot_locations = [[],
                   [],
                   [] ]
robot_current_locations = random.sample(valid_robot_starting_cells, N_ROBOTS)

for r in range(N_ROBOTS):
    robot_locations[r].append(robot_current_locations[r])

target_list = [Target() for _ in range(N_TARGETS)]
target_locations = [[] for _ in range(N_TARGETS)]

initial_targets = random.sample(valid_target_cells, N_TARGETS)
for i in range(N_TARGETS):
    target_list[i].initialize(initial_targets[i])
    target_locations[i].append(initial_targets[i])

target_location_history = list(initial_targets)

robot_assignments = assignment(target_list, robot_current_locations)

step_count = 0
start_time = time.perf_counter()

# -----------------------
# Simulation
# -----------------------

while (len(target_location_history) < TOTAL_TARGETS or any(t.active for t in target_list)):

    step_count += 1

    # Move robots
    for r in range(N_ROBOTS):

        target = robot_assignments[r]

        if target is None:
            robot_locations[r].append(robot_current_locations[r])
        else:

            action = target.best_actions[robot_current_locations[r][0],
                                        robot_current_locations[r][1]]
            new_loc = get_next_location(robot_current_locations[r][0],
                                        robot_current_locations[r][1],
                                        action)

            robot_current_locations[r] = new_loc
            robot_locations[r].append(new_loc)

            if new_loc == target.location:
                target.remove()

    # Update target paths
    for i in range(N_TARGETS):
        if target_list[i].active:
            target_locations[i].append(target_list[i].location)
        else:
            target_locations[i].append((0,0))

    # Count active
    active_targets = sum(t.active for t in target_list)

    # Possibly activate new
    if (2 <= active_targets < 4) and len(target_location_history) < TOTAL_TARGETS:
        if np.random.rand() < 0.1:
            loc = random.choice(valid_target_cells)
            while loc in target_location_history:
                loc = random.choice(valid_target_cells)

            for t in target_list:
                if not t.active:
                    t.initialize(loc)
                    target_location_history.append(loc)
                    break

    robot_assignments = assignment(target_list, robot_current_locations)

    if not any(t.active for t in target_list) and len(target_location_history) >= TOTAL_TARGETS:
        break

end_time = time.perf_counter()
print(f"Entire simulation complete in {end_time-start_time:.2f} seconds")

# -----------------------
# Pygame Animation (SLOWER)
# -----------------------

WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE // size
FPS = 2  # slower
EXTRA_DELAY = 0.4  # extra pause per step

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

max_steps = max(len(r) for r in robot_locations)

running = True
step = 0

while running:
    clock.tick(FPS)
    screen.fill((255,255,255))

    for r in range(N_ROBOTS):
        row,col = robot_locations[r][step]
        pygame.draw.circle(screen,(255,105,180),(col*CELL_SIZE+CELL_SIZE//2,row*CELL_SIZE+CELL_SIZE//2),CELL_SIZE//2-2)

    for t in range(N_TARGETS):
        row,col = target_locations[t][step]
        color = (0,255,0) if (row,col)!=(0,0) else (0,0,0)
        pygame.draw.circle(screen,color,(col*CELL_SIZE+CELL_SIZE//2,row*CELL_SIZE+CELL_SIZE//2),CELL_SIZE//2-4)

    pygame.display.flip()
    time.sleep(EXTRA_DELAY)

    step+=1
    if step>=max_steps:
        time.sleep(2)
        running=False

pygame.quit()