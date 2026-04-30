# Basic Q-Learning in a 25x25 warehouse environment

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

import time


# Define colours
cmap = colors.ListedColormap(['white', 'red', 'black', 'green'])
EMPTY, SHELF, WALL, TARGET = 0,1,2,3

# Initialise environment (25x25)


warehouse = np.array([
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],#0
            [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],#1
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#2
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#3
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#4
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#5
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#6
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#7
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#8
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#9
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#10
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#11
            [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],#12
            [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],#13
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#14
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#15
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#16
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,3,0,0,1,1,0,0,2],#17
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#18
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#19
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#20
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#21
            [2,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,2],#22
            [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],#23
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]])#24


# Valid starting squares (not carrying)
valid_starting_squares = []
for i in range(25):
    for j in range(25):
        if warehouse[i, j] == 0:
            valid_starting_squares.append((i, j))

# Visualise warehouse
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(warehouse[:, :], cmap=cmap)
ax.title.set_text("Warehouse")
plt.savefig('1.Warehouse 25x25.png')


# Initialise actions
actions = ['up', 'right', 'down', 'left']
num_actions = len(actions)

# Define get_next_action
def get_next_action(i, j, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(q_values[i, j, :])

# Define get_next_location
def get_next_location(i, j, action_index):
    if actions[action_index] == 'up' and i > 0:
        return i-1, j
    elif actions[action_index] == 'right' and j < 24:
        return i, j+1
    elif actions[action_index] == 'down' and i < 24:
        return i+1, j
    elif actions[action_index] == 'left' and j > 0:
        return i, j-1
    else:
        return i, j
# Define parameters
epsilon = 0.9
discount_factor = 0.99
learning_rate = 0.9
epsilon_decay = 0.999

# Initialise reward table for both layers
rewards = np.zeros((25, 25))
for i in range(25):
    for j in range(25):
        cell = warehouse[i, j]
        if cell == EMPTY:
            rewards[i, j] = -1
        elif cell == SHELF:
            rewards[i, j] = -100
        elif cell == WALL:
            rewards[i, j] = -100
        else:
            rewards[i,j]= 100

# Initialise Q-table
q_values = np.zeros((25, 25, num_actions))

#----------------------------------------------------------------------------------------------------------------------#
#BEGIN LEARNING
#----------------------------------------------------------------------------------------------------------------------#
start_time = time.time()

for episode in range(1000):
    terminal = False
    i, j = random.choice(valid_starting_squares)
    print("Episode #: %s" % (episode,), end='\r')#takes additional time
    for step in range(500):
        if terminal == False:

            action = get_next_action(i, j, epsilon)
            old_i, old_j = i, j

            i, j = get_next_location(i, j, action)
            reward = rewards[i, j]
            if warehouse[i, j] == 3:
                terminal = True

            
            # -------------------------------------
            # Q update
            # -------------------------------------
            old_q = q_values[old_i, old_j, action]
            best_future = np.max(q_values[i, j])
            new_q = old_q + learning_rate * (reward + discount_factor * best_future - old_q)
            q_values[old_i, old_j, action] = new_q
        

    epsilon = max(0.5, epsilon * epsilon_decay)

print("Training complete!")
end_time = time.time()
time_elapsed = end_time - start_time
print("\n q-learning-basic time taken = ", time_elapsed)

best_action = np.zeros((25, 25))
for i in range(25):
    for j in range(25):
            best_action[i, j] = np.argmax(q_values[i, j, :])
with open('1.Basic_Q_Learning_Q_Values.txt', 'w') as f:
    f.write("Final Policy\n")
    for i in range(25):
        for j in range(25):
            f.write(f"{np.max(q_values[i, j, :])} ")
        f.write("\n")

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(warehouse[:, :], cmap=cmap)
ax.title.set_text('Greedy Policy (Carrying)')
for i in range(25):
    for j in range(25):
        dx = dy = 0
        if best_action[i, j] == 0:
            dy = -0.3
        elif best_action[i, j] == 1:
            dx = 0.3
        elif best_action[i, j] == 2:
            dy = 0.3
        elif best_action[i, j] == 3:
            dx = -0.3
        else:
            dy = 0
            dx = 0
        ax.arrow(j, i, dx, dy, color='black', head_width=0.2, head_length=0.1)
plt.savefig('1.Greedy Policy obtained from Basic Q-Learning.png')

path_i = []
path_j = []

i, j = random.choice(valid_starting_squares)
running = True
while (running):
    old_i, old_j = i, j
    if best_action[i, j] == 0:
        i = i-1
    elif best_action[i, j] == 1:
        j = j+1
    elif best_action[i, j] == 2:
        i = i+1
    elif best_action[i, j] == 3:
        j = j-1


    path_i.append(old_i)
    path_j.append(old_j)
    if warehouse[old_i, old_j] == 3:
        running = False

fig, ax = plt.subplots(figsize=(25, 25))
fig.patch.set_linewidth(0)
ax.imshow(warehouse[:,:], cmap=cmap)
ax.title.set_text('Pick-up')
ax.plot(path_j, path_i)
plt.savefig('1.Path to Target from Random Starting Square.png')