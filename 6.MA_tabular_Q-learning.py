import numpy as np
from Warehouses import return_warehouse
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation


#load environment, including state space and action space
warehouse, size = return_warehouse(input("Enter warehouse (A, B, C, D, E, F): "))

for i in range(size):
    for j in range(size):
        if warehouse[i,j] == 3:
            warehouse[i,j] = 1

actions = ['up', 'right', 'down', 'left']
num_actions = len(actions)

#visualize empty warehouse
cmap = colors.ListedColormap(['white', 'red', 'black'])

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(warehouse[:,:], cmap = cmap)
# remove all figure padding
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')
# save image with no padding
plt.savefig('6.Warehouse.png', bbox_inches='tight', pad_inches=0)
plt.close(fig)
#define number of agents
N_AGENTS = int(input("Enter number of agents: "))

#valid starting locations:
valid_robot_starting_squares = [
    (i,j)
    for i in range(size)
    for j in range(size)
    if warehouse[i,j] == 0
]

valid_target_starting_squares = [
    (i,j)
    for i in range(size)
    for j in range(size)
    if warehouse[i,j] == 1
]

#get_next_location function
def get_next_location(i,j,action_index):
    if actions[action_index] == 'up' and i > 0:
        return i-1, j
    elif actions[action_index] == 'right' and j < size-1:
        return i, j+1
    elif actions[action_index] == 'down' and i < size-1:
        return i+1, j
    elif actions[action_index] == 'left' and j > 0:
        return i,j-1
    else:
        return i,j

#target class, for keeping track of targets if they appear and disappear throughout simulation
class Target:

    def __init__(self):
        self.active = False
        self.best_actions = np.zeros((size,size), dtype=int)
        self.V_table = np.zeros((size,size))
    
    def initialize(self, location):
        self.active = True
        self.location = location

    def remove(self):
        self.active = False
        self.V_table = np.zeros((size,size))
        self.best_actions = np.zeros((size,size), dtype=int)
#training parameters
#training parameters
num_episodes = int(input("Enter Number of Episodes (default 1000): ") or 1000)
num_steps = int(input("Enter Steps per Episode (default 500): ") or 500)
epsilon_start = float(input("Enter Epsilon (default 0.9): ") or 0.9)
discount_factor = float(input("Enter Discount Factor (default 0.99): ") or 0.99)
learning_rate = float(input("Enter Learning Rate (default 0.9): ") or 0.9)
min_epsilon = float(input("Enter Minimum Epsilon (default 0.1): ") or 0.1)
epsilon_decay = float(input("Enter Epsilon Decay (default 1/episodes): ") or (1 / num_episodes))

#q-learning function - calulated as one target per n_robots, not n_targets per one robot
def learn(target_location):
    epsilon = epsilon_start
    q_values = np.zeros((size, size, num_actions))
    rewards = np.zeros((size,size))

    # define reward table
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

            q_values[old_i, old_j, action] = old_q + learning_rate * (
                reward + discount_factor * best_future - old_q
            )

            if (i,j) == target_location:
                break
            #end of step
        #end of episode
        epsilon = max(min_epsilon, epsilon-epsilon_decay)
    #end of training, returning results
    best_actions = np.argmax(q_values, axis=2)
    V_table = np.max(q_values, axis=2)

    return best_actions, V_table

#optimal assignment
def assignment(target_list, robot_current_locations):  # number of agents constant
    
    n_targets = len(target_list)
    option_values = np.full((n_targets, n_targets, n_targets), -np.inf)

    for A in range(n_targets):
        for B in range(n_targets):
            for C in range(n_targets):

                # ensure all assignments are unique
                if len({A, B, C}) == N_AGENTS:

                    option_values[A,B,C] = (
                        target_list[A].V_table[robot_current_locations[0]]
                        + target_list[B].V_table[robot_current_locations[1]]
                        + target_list[C].V_table[robot_current_locations[2]]
                    ) / N_AGENTS

    # get best assignment indices
    best_indices = np.unravel_index(np.argmax(option_values), option_values.shape)
    best_indices = tuple(map(int, best_indices))

    return [
        target_list[best_indices[0]],
        target_list[best_indices[1]],
        target_list[best_indices[2]]
    ]
#initialize targets
target_list = [Target() for _ in range(4)]

target_indices = random.sample(
    range(len(valid_target_starting_squares)), 
    4
)

for i in range(4):
    target_list[i].initialize(
        valid_target_starting_squares[target_indices[i]]
    )

#initialize robots
robot_indices = random.sample(
    range(len(valid_robot_starting_squares)), 
    N_AGENTS
)

robot_current_locations = [
    valid_robot_starting_squares[idx]
    for idx in robot_indices
]

#train all targets
for target in target_list:
    target.best_actions, target.V_table = learn(target.location)

#compute optimal assignment
robot_assignments = assignment(target_list, robot_current_locations)

#simulation
terminal_robots = [False] * N_AGENTS
paths = [[] for _ in range(N_AGENTS)]

max_steps = 1000
step_counter = 0

while not all(terminal_robots) and step_counter < max_steps:

    for i in range(N_AGENTS):
        paths[i].append(robot_current_locations[i])

    for idx in range(N_AGENTS):

        if terminal_robots[idx]:
            continue

        target = robot_assignments[idx]

        if robot_current_locations[idx] == target.location:
            terminal_robots[idx] = True
        else:
            action = target.best_actions[robot_current_locations[idx]]
            robot_current_locations[idx] = get_next_location(
                robot_current_locations[idx][0],
                robot_current_locations[idx][1],
                action
            )

    step_counter += 1

# -----------------------
# render results
# -----------------------


fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(warehouse[:,:], cmap=cmap)
ax.set_title(f"{N_AGENTS} Robots Path Animation")

# convert paths for plotting
robot_x = [[pos[1] for pos in path] for path in paths]
robot_y = [[pos[0] for pos in path] for path in paths]

# plot starts
for i in range(N_AGENTS):
    ax.scatter(robot_x[i][0], robot_y[i][0], marker='*', s=200)

# plot targets
targets_x = [t.location[1] for t in target_list]
targets_y = [t.location[0] for t in target_list]
ax.scatter(targets_x, targets_y, s=200)

# create lines
lines = [
    ax.plot([], [], linewidth=2)[0]
    for _ in range(N_AGENTS)
]

def animate(frame):
    for i in range(N_AGENTS):
        lines[i].set_data(robot_x[i][:frame+1], robot_y[i][:frame+1])
    return lines

num_frames = max(len(x) for x in robot_x)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=num_frames,
    interval=300,
    repeat=False
)
# save the animation as a GIF
ani.save("6.robot_paths.gif", writer='pillow', fps=2)
plt.show()

# -----------------------
# Pygame Animation
# -----------------------
import pygame
import time

WINDOW_SIZE = 600
GRID_SIZE = size  # same as your warehouse
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
FPS = 5

# Robot colors
PINK = (255, 105, 180)     # hot pink
YELLOW = (255, 255, 0)     # bright yellow
BLUE = (0, 0, 255)         # classic blue

# Example list for multiple robots
ROBOT_COLORS = [PINK, YELLOW, BLUE]

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Robot Q-learning Simulation")
clock = pygame.time.Clock()

# Load warehouse image as background
background_img = pygame.image.load("6.Warehouse.png")
background_img = pygame.transform.scale(background_img, (WINDOW_SIZE, WINDOW_SIZE))

num_steps = max(len(p) for p in paths)
running = True
step = 0

while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(background_img, (0,0))

    for i, path in enumerate(paths):
        if step < len(path):
            row, col = path[step]
            pygame.draw.circle(
                screen,
                ROBOT_COLORS[i % len(ROBOT_COLORS)],
                (col*CELL_SIZE + CELL_SIZE//2, row*CELL_SIZE + CELL_SIZE//2),
                CELL_SIZE//2 - 2
            )

    pygame.display.flip()
    step += 1
    if step >= num_steps:
        time.sleep(2)
        running = False

pygame.quit()