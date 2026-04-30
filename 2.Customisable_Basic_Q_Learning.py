#Same format as 1.Basic_Q_Learning_25x25.py but with more customisable parameters for the warehouse size, number of episodes, and learning rate., 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

import time

#optimal action table for Warehouse A:
ideal_actions_A = np.array([[4,4,4,4,4,4,4,4,4,4],
                           [4,1,1,1,5,2,3,3,3,4],
                           [4,0,4,4,5,2,4,4,0,4],
                           [4,7,4,4,5,2,4,4,7,4],
                           [4,2,4,4,5,2,4,4,2,4],
                           [4,2,4,4,5,2,4,4,2,4],
                           [4,2,4,4,1,1,4,4,2,4],
                           [4,2,4,4,6,0,4,4,2,4],
                           [4,1,1,1,6,0,3,3,3,4],
                           [4,4,4,4,4,4,4,4,4,4]])
#where 0,1,2,3 reperesent up, down, right, left
#4 represents don't care
#5 represents right or down
#6 represents right or up
#7 represents down or up
number_of_states_not_X = 40
#initialize environment:
from Warehouses import return_warehouse
warehouse, size  = return_warehouse(input("Enter Warehouse: ") or 'F')

#valid starting squares:
valid_starting_squares = []
for i in range(size):
    for j in range(size):
        if warehouse[i,j] == 0:
            valid_starting_squares.append((i,j))

#colours:           
cmap = colors.ListedColormap(['white', 'red', 'black', 'green'])

#for clarity:
EMPTY, SHELF, WALL, TARGET = 0,1,2,3

#visualise warehouse:
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(warehouse[:,:], cmap = cmap)
title = 'Warehouse Layout ' + str(size) + 'x' + str(size)
ax.title.set_text(title)
plt.savefig('2.Warehouse.png')

#initialise actions:
actions = ['up', 'right', 'down', 'left']
num_actions = len(actions)

#define get_next_action
def get_next_action(i,j,epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(q_values[i,j, :])

#define get_next_location
def get_next_location(i, j, action_index):
    if actions[action_index] == 'up' and i > 0:
        return i-1, j
    elif actions[action_index] == 'right' and j < size-1:
        return i, j+1
    elif actions[action_index] == 'down' and i < size-1:
        return i+1, j
    elif actions[action_index] == 'left' and j > 0:
        return i, j-1
    else:
        return i, j

#define parameters:
num_episodes = input("Enter Number of Episodes: ")
if num_episodes == '':
    num_episodes = 15000
else:
    num_episodes = int(num_episodes)
num_steps = input("Enter Number of Steps per Episode: ")
if num_steps == '':
    num_steps = 1100
else:
    num_steps = int(num_steps)
epsilon = input("Enter Epsilon: ")
if epsilon == '':
    epsilon = 1
else:
    epsilon = float(epsilon)
discount_factor = input("Enter Discount Factor: ")
if discount_factor == '':
    discount_factor = 0.9
else:
    discount_factor = float(discount_factor)
learning_rate = input("Enter Learning Rate: ")
if learning_rate == '':
    learning_rate = 0.2
else:
    learning_rate = float(learning_rate)
epsilon_decay = input("Enter Epsilon Decay: ")
if epsilon_decay == '':
    epsilon_decay = 1/num_episodes
else:
    epsilon_decay = float(epsilon_decay)
min_epsilon = input("Enter Minimum Epsilon: ")
if min_epsilon == '':
    min_epsilon = 0.1
else:
    min_epsilon = float(min_epsilon)

#initialize reward table:
rewards = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        cell = warehouse[i,j]
        if cell == EMPTY:
            rewards[i,j] = -1
        elif cell == SHELF:
            rewards[i,j] = -100
        elif cell == WALL:
            rewards[i,j] = -100
        else:
            rewards[i,j] = 100

#initialize Q-table

q_values = np.zeros((size, size, num_actions))

#begin learning:
################

start_time = time.time()
rewards_per_episode = np.zeros(num_episodes)
for episode in range(num_episodes):
    print("Episode #: %s" % (episode,), end='\r')
    terminal_state = False
    i,j = random.choice(valid_starting_squares)
    for step in range(num_steps):
        if terminal_state == False:

            #select action
            action = get_next_action(i,j,epsilon)
            old_i, old_j = i,j

            #get next state and update current positin
            i,j = get_next_location(i,j,action)
            #get rewards from new state
            reward = rewards[i,j]
            rewards_per_episode[episode] += reward

            if warehouse[i,j] == 3:
                terminal_state = True
            
            #update Q-table

            old_q = q_values[old_i, old_j, action]
            best_future = np.max(q_values[i,j])
            new_q = old_q + learning_rate * (reward + discount_factor * best_future - old_q)
            q_values[old_i, old_j, action] = new_q

    epsilon = max(min_epsilon, epsilon - epsilon_decay)

end_time = time.time()
print("Training Complete:")

#RESULTS SECTION

time_elapsed = end_time-start_time

print('\n Time taken: ', time_elapsed, ' s')

#cumulative reward per episode:
fp = open('rewards.txt', 'w')
for i in range(1, num_episodes):
    fp.write(str(rewards_per_episode[i]) + '\n')
fp.close()
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(rewards_per_episode)
ax.title.set_text('Rewards per Episode')
plt.savefig('2.Rewards_Per_Episode.png')

#best actions:
best_actions = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        best_actions[i,j] = np.argmax(q_values[i,j,:])

with open('2.Q_Values.txt', 'w') as f:
    f.write(str(q_values))

fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(warehouse[:,:], cmap = cmap)
ax.title.set_text('Learned Policy')
for i in range(size):
    for j in range(size):
        if warehouse[i,j] == 0:
            dx = dy = 0
            if best_actions[i,j] == 0:
                dy = -0.3
            elif best_actions[i,j] == 1:
                dx = 0.3
            elif best_actions[i,j] == 2:
                dy = 0.3
            elif best_actions[i,j] == 3:
                dx = -0.3
            ax.arrow(j, i, dx, dy, color='black', head_width=0.2, head_length=0.1)
plt.savefig('2.Learned_Policy.png')

path_i = []
path_j = []

i,j = random.choice(valid_starting_squares)
running = True
#while running:
#    old_i, old_j = i,j
#    if best_actions[i,j] == 0:
#        i = i-1
#    elif best_actions[i,j] == 1:
#        j = j+1
#    elif best_actions[i,j] == 2:
#        i = i+1
#    elif best_actions[i,j] == 3:
#        j = j-1
#    
#    path_i.append(old_i)
#    path_j.append(old_j)
#    if warehouse[old_i, old_j] == 3:
#        running = False

#fig, ax = plt.subplots(figsize=(10,10))
#fig.patch.set_linewidth(0)
#ax.imshow(warehouse[:,:], cmap = cmap)
#ax.title.set_text('Path taken from random starting point')
#ax.plot(path_j, path_i)
#plt.savefig('2.Path_Taken.png')

#comparing best action table with ideal action table:
"""
total = 0
for i in range(size):
    for j in range(size):
        if ideal_actions_A[i,j] != 4: #if not don't care
            if best_actions[i,j] == ideal_actions_A[i,j]:
                total += 1
            else:
                if ideal_actions_A[i,j] == 5: #if right or down
                    if best_actions[i,j] in [1,2]: #if right or down
                        total += 1
                elif ideal_actions_A[i,j] == 6: #if right or up
                    if best_actions[i,j] in [1,0]: #if right or up
                        total += 1
                elif ideal_actions_A[i,j] == 7: #if down or up
                    if best_actions[i,j] in [2,0]: #if down or up
                        total += 1
print("Number of correct actions: ", total, " out of ", number_of_states_not_X)
"""