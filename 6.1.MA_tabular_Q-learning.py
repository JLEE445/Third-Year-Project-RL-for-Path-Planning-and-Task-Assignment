import numpy as np
from Warehouses import return_warehouse
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import pygame
import time

#load warehouse
warehouse, size = return_warehouse(input("Enter warehouse (A, B, C, D, E, F): "))

# treat old targets as regular squares
for i in range(size):
    for j in range(size):
        if warehouse[i, j] == 3:
            warehouse[i, j] = 1

actions = ['up', 'right', 'down', 'left']
num_actions = len(actions)

# Visualize empty warehouse
cmap = colors.ListedColormap(['white', 'red', 'black'])
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(warehouse[:, :], cmap=cmap)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
plt.savefig('6.Warehouse.png', bbox_inches='tight', pad_inches=0)
plt.close(fig)

# Number of agents
N_AGENTS = 3

# Valid starting locations
valid_robot_starting_squares = [(i, j) for i in range(size) for j in range(size) if warehouse[i, j] == 0]
valid_target_starting_squares = [(i, j) for i in range(size) for j in range(size) if warehouse[i, j] == 1]

#get next location
def get_next_location(i, j, action_index):
    if actions[action_index] == 'up' and i > 0:
        return i - 1, j
    elif actions[action_index] == 'right' and j < size - 1:
        return i, j + 1
    elif actions[action_index] == 'down' and i < size - 1:
        return i + 1, j
    elif actions[action_index] == 'left' and j > 0:
        return i, j - 1
    else:
        return i, j

#target class
class Target:
    def __init__(self):
        self.active = False
        self.best_actions = np.zeros((size, size), dtype=int)
        self.V_table = np.zeros((size, size))

    def initialize(self, location):
        self.active = True
        self.location = location

    def remove(self):
        self.active = False
        self.V_table = np.zeros((size, size))
        self.best_actions = np.zeros((size,size))

#training parameters
num_episodes = int(input("Enter Number of Episodes (default 5000): ") or 5000)
num_steps = int(input("Enter Steps per Episode (default 500): ") or 500)
epsilon_start = float(input("Enter Epsilon (default 0.9): ") or 0.9)
discount_factor = float(input("Enter Discount Factor (default 0.99): ") or 0.99)
learning_rate = float(input("Enter Learning Rate (default 0.9): ") or 0.9)
min_epsilon = float(input("Enter Minimum Epsilon (default 0.1): ") or 0.1)
epsilon_decay = float(input("Enter Epsilon Decay (default 1/episodes): ") or (1 / num_episodes))

#q-learning function
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

    for episode in range(num_episodes):
        i, j = random.choice(valid_robot_starting_squares)
        for step in range(num_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(q_values[i, j, :])

            old_i, old_j = i, j
            i, j = get_next_location(i, j, action)
            reward = rewards[i, j]

            old_q = q_values[old_i, old_j, action]
            best_future = np.max(q_values[i, j, :])
            q_values[old_i, old_j, action] = old_q + learning_rate * (
                reward + discount_factor * best_future - old_q
            )

            if (i, j) == target_location:
                break

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    best_actions = np.argmax(q_values, axis=2)
    V_table = np.max(q_values, axis=2)
    return best_actions, V_table

#assignment function
def assignment(target_list, robot_current_locations): #target_list = [t for t in target_list if t.active], robot_current_locations = robot_current_locations
    #initialise variables and constants
    n_agents = len(robot_current_locations) #always three
    n_targets = len(target_list) #number of active targets
    assignments = [None] * n_agents  # waiting by default unless otherwise

    if n_targets == 0:
        return assignments #break

    # collect values for all target robot(current_location) pairs, to make optimal assignement at that moment in time
    option_values = np.zeros((n_agents, n_targets))
    for r in range(len(robot_current_locations)):
        r_loc = robot_current_locations[r]
        for t in range(len(target_list)):
            target = target_list[t]
            option_values[r, t] = target.V_table[r_loc]

    # Greedy assignment: assign the highest value target to each robot without conflicts
    assigned_targets = set()
    for _ in range(min(n_agents, n_targets)):
        # Find the robot-target pair with the highest value
        r_idx, t_idx = np.unravel_index(np.argmax(option_values, axis=None), option_values.shape)
        assignments[r_idx] = target_list[t_idx]
        assigned_targets.add(t_idx)
        # Set row and column to -inf so they are not chosen again
        option_values[r_idx, :] = -np.inf
        option_values[:, t_idx] = -np.inf

    return assignments

#initialize targets
target_list = [Target() for _ in range(4)]
active_target_indices = random.sample(range(len(valid_target_starting_squares)), 3)
for i in range(3):
    target_list[i].initialize(valid_target_starting_squares[active_target_indices[i]])

# Initialize robots
robot_indices = random.sample(range(len(valid_robot_starting_squares)), N_AGENTS)
robot_current_locations = [valid_robot_starting_squares[idx] for idx in robot_indices]

# Train active targets
start_time = time.time()
for target in target_list:
    if target.active:
        target.best_actions, target.V_table = learn(target.location)

# Compute assignment
robot_assignments = assignment([t for t in target_list if t.active], robot_current_locations)

#simulation
terminal_robots = [False] * N_AGENTS
paths = [[] for _ in range(N_AGENTS)]
target_paths = [[] for _ in range(len(target_list))]

max_steps = 250
step_counter = 0
count2 = 0
count3 = 0
targets_to_collect = 10
targets_found = 0
while (not all(terminal_robots) or targets_found < targets_to_collect) and step_counter < max_steps:
    for i in range(len(target_list)):
        if not target_list[i].active:
            target_paths[i].append((0,0))
        else:
            target_paths[i].append(target_list[i].location)

    for i in range(N_AGENTS):
        paths[i].append(robot_current_locations[i])

    for idx in range(N_AGENTS):
        if terminal_robots[idx]:
            continue

        target = robot_assignments[idx]

        if target is None:
            # Robot waits -> mark as done
            terminal_robots[idx] = True
            continue

        if robot_current_locations[idx] == target.location:
            terminal_robots[idx] = True
            count3 += 1
            print("robot", idx, "found target", target.location)
            target.remove()
            targets_found += 1
            robot_current_locations[idx] = paths[idx][-2]
        else:
            action = target.best_actions[robot_current_locations[idx]]
            robot_current_locations[idx] = get_next_location(
                robot_current_locations[idx][0],
                robot_current_locations[idx][1],
                action
            )

    step_counter += 1
    count = 0
    for i in range(3):
        if target_list[i].active:
            count += 1
        else:
            if count3 < 10 and random.randint(0,10) < 2:
                terminal_robots = [False] *N_AGENTS
                loc = random.choice(valid_target_starting_squares)
                target_list[i].initialize(loc)
                target_list[i].best_actions, target_list[i].V_table = learn(target_list[i].location)
                # Compute assignment
                robot_assignments = assignment([t for t in target_list if t.active], robot_current_locations)
                count2 += 1

    print("number of active targets: " ,count)
end_time = time.time()
print("Time passed: ", end_time - start_time)
print(count3)

#matplotlib animation
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(warehouse[:, :], cmap=cmap)
ax.set_title(f"{N_AGENTS} Robots Path Animation")

robot_x = [[pos[1] for pos in path] for path in paths]
robot_y = [[pos[0] for pos in path] for path in paths]

for i in range(N_AGENTS):
    ax.scatter(robot_x[i][0], robot_y[i][0], marker='*', s=200)

targets_x = [t.location[1] for t in target_list if t.active]
targets_y = [t.location[0] for t in target_list if t.active]
ax.scatter(targets_x, targets_y, s=200)

lines = [ax.plot([], [], linewidth=2)[0] for _ in range(N_AGENTS)]

def animate(frame):
    for i in range(N_AGENTS):
        lines[i].set_data(robot_x[i][:frame + 1], robot_y[i][:frame + 1])
    return lines

num_frames = max(len(x) for x in robot_x)
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=300, repeat=False)
ani.save("6.robot_paths.gif", writer='pillow', fps=2)
plt.show()

#pygame animation
WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE // size
FPS = 3

ROBOT_COLORS = [
    (255,105,180),
    (255,165,0),
    (0,0,255),
    (0,255,255),
    (255,255,0),
    (128,0,255),
]

GREEN = (0,128,0)
BLACK = (0,0,0)

pygame.init()

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Robot Q-learning Simulation")

clock = pygame.time.Clock()

background_img = pygame.image.load("6.Warehouse.png")
background_img = pygame.transform.scale(background_img,(WINDOW_SIZE,WINDOW_SIZE))

num_steps = max(len(p) for p in paths)

running = True
step = 0

while running:

    clock.tick(FPS)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    screen.blit(background_img,(0,0))

    for i, path in enumerate(paths):

        if step < len(path):

            row, col = path[step]

            pygame.draw.circle(
                screen,
                ROBOT_COLORS[i % len(ROBOT_COLORS)],
                (col * CELL_SIZE + CELL_SIZE//2,
                 row * CELL_SIZE + CELL_SIZE//2),
                CELL_SIZE//2 - 2
            )

    for i, path in enumerate(target_paths):

        if step < len(path):

            row, col = path[step]
            if (row,col) != (0,0):
                pygame.draw.circle(
                    screen,
                    GREEN,
                    (col * CELL_SIZE + CELL_SIZE//2,
                    row * CELL_SIZE + CELL_SIZE//2),
                    CELL_SIZE//2 - 2
                )
            else:
                pygame.draw.circle(
                    screen,
                    BLACK,
                    (col * CELL_SIZE + CELL_SIZE//2,
                    row * CELL_SIZE + CELL_SIZE//2),
                    CELL_SIZE//2 - 2
                )


    pygame.display.flip()

    step += 1

    if step >= num_steps:

        time.sleep(2)

        running = False

pygame.quit()