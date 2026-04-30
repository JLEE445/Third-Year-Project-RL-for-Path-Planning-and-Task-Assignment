#Customisable DQN implementation for the warehouse environment, building on the previous Q-learning implementation.
#and using pytorch for the neural network.
#now includes reward clipping
import numpy as np
import matplotlib.pyplot as plt
from collections import deque #for experience replay
import random
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import colors
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
cmap = colors.ListedColormap(['white', 'red', 'black', 'green'])

#initialize environment:
from Warehouses import return_warehouse
warehouse, size = return_warehouse('A') 

#valid starting squares:
valid_starting_squares = []
for i in range(size):
    for j in range(size):
        if warehouse[i,j] == 0:
            valid_starting_squares.append((i,j))

#deep Q network:
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)
    
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

#replay memory:
class Replaymemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen = maxlen)
    
    def append(self, experience):
        self.memory.append(experience)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
    
#for clarity:
EMPTY, SHELF, WALL, TARGET = 0,1,2,3

#visualise warehouse:
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(warehouse[:,:], cmap = cmap)
title = 'Warehouse Layout ' + str(size) + 'x' + str(size)
ax.title.set_text(title)
plt.savefig('3.Warehouse.png')

#initialise actions:
actions = ['up', 'right', 'down', 'left']
num_actions = len(actions)

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

#initialise parameters:
counter = 0
num_episodes = input("Enter Number of Episodes: ")
if num_episodes == '':
    num_episodes = 500
else:
    num_episodes = int(num_episodes)
num_steps = input("Enter Number of Steps per Episode: ")
if num_steps == '':
    num_steps = 100
else:
    num_steps = int(num_steps)
epsilon = input("Enter Epsilon: ")
if epsilon == '':
    epsilon = 0.9
else:
    epsilon = float(epsilon)
discount_factor = input("Enter Discount Factor: ")
if discount_factor == '':
    discount_factor = 0.99
else:
    discount_factor = float(discount_factor)
learning_rate = input("Enter Learning Rate: ")
if learning_rate == '':
    learning_rate = 0.008
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
replay_memory_size = input("Enter Replay Memory Size: ")
if replay_memory_size == '':
    replay_memory_size = 10000
else:
    replay_memory_size = int(replay_memory_size)

#initialize networks:
policy_dqn = DQN(size*size, 128, num_actions)
target_dqn = DQN(size*size, 128, num_actions)

#initialize replay memory:
replay_memory = Replaymemory(replay_memory_size)

#initialize optimizer and loss function:
optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

#def state_to_dqn_input:
def state_to_dqn_input(state, num_states)->torch.Tensor:
    input_tensor = torch.zeros(num_states)
    input_tensor[state] = 1
    return input_tensor

#def optimze:
def optimize(mini_batch, policy_dqn, target_dqn):
    num_states = size*size
    current_q_list = []
    target_q_list = []
    counter = 0
    
    for state, action, new_state, reward, terminated in mini_batch:
        if terminated:
            target = torch.FloatTensor([reward])
            counter += 1
        else:
            with torch.no_grad():
                target = torch.FloatTensor(
                    reward + discount_factor * target_dqn(state_to_dqn_input(new_state, num_states)).max()
                )
        
        current_q = policy_dqn(state_to_dqn_input(state, num_states))
        current_q_list.append(current_q)

        target_q = target_dqn(state_to_dqn_input(state, num_states))
        target_q[action] = target
        target_q_list.append(target_q)
    
    loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return counter

#initialize rewards:
rewards = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        #change made: reward normalization - clipping rewards to between (1 and -1) to help with training stability:
        cell = warehouse[i,j]
        if cell == EMPTY:
            rewards[i,j] = -0.01
        elif cell == SHELF:
            rewards[i,j] = -1
        elif cell == WALL:
            rewards[i,j] = -1
        elif cell == TARGET:
            rewards[i,j] = 1

#begin training the neural network:
###################################

start_time = time.time()
rewards_per_episode = np.zeros(num_episodes)

target_dqn.load_state_dict(policy_dqn.state_dict())


for ep in range(num_episodes):
    visited_states = []
    state_i, state_j = random.choice(valid_starting_squares)
    state = state_i*size + state_j
    print("Episode #: %s" % (ep,), end='\r')
    terminated = False

    for step in range(num_steps):
        if not terminated:
            visited_states.append(state)
            if random.random() < epsilon:
                action = random.randint(0,3)
            else:
                with torch.no_grad():
                    qvals = policy_dqn(state_to_dqn_input(state, size*size))
                    action = qvals.argmax().item()

            new_state_i, new_state_j = get_next_location(state_i, state_j, action)
            new_state = new_state_i*size + new_state_j
            #logic for reward shaping, to encourage exploration:
            #if new_state in visited_states:
                #reward -10
                #reward+= rewards[new_state_i, new_state_j]
            #else:   
                #reward = rewards[new_state_i, new_state_j]
            reward = rewards[new_state_i, new_state_j]
            if warehouse[new_state_i, new_state_j] ==TARGET:
                terminated = True
            
            replay_memory.append((state, action, new_state, reward, terminated))
            state = new_state
            state_i, state_j = new_state_i, new_state_j
            rewards_per_episode[ep] += reward
    #end of single episode, now optimize network:
    if len(replay_memory) >= 32:
        mini_batch = replay_memory.sample(32)
        counter += optimize(mini_batch, policy_dqn, target_dqn)
    
    #decay epsilon:
    epsilon = max(min_epsilon, epsilon-epsilon_decay)
    #update target network every 10 episodes:
    if ep % 10 == 0:
        target_dqn.load_state_dict(policy_dqn.state_dict())
    
end_time = time.time()
print("Training Complete:")

#RESULTS SECTION

time_elapsed = end_time-start_time
torch.save(policy_dqn.state_dict(), '3.DQN_model.pt')

print('\n Time taken: ', time_elapsed, ' s')
#print(counter)
#cumulative reward per episode:
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(rewards_per_episode)
ax.title.set_text('Rewards per Episode2')
plt.savefig('3.Rewards_Per_Episode.png')
#for i in range(500):
#    print(rewards_per_episode[i])

#best_actions:
best_actions = np.zeros((size, size), dtype=int)
for i in range(size):
    for j in range(size):
        state = i*size + j
        with torch.no_grad():
            qvals = policy_dqn(state_to_dqn_input(state, size*size))
            best_actions[i,j] = qvals.argmax().item()

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(warehouse[:,:], cmap = cmap)
ax.title.set_text('Learned Policy')
for i in range(size):
    for j in range(size):
        dx = dy = 0
        if best_actions[i,j] == 0: #up
            dy = -0.3
        elif best_actions[i,j] == 1:#right
            dx = 0.3
        elif best_actions[i,j] == 2:#down
            dy = 0.3
        elif best_actions[i,j] == 3:#left
            dx = -0.3
        ax.arrow(j, i, dx, dy, color='black', head_width=0.2, head_length=0.1)
plt.savefig('3.Learned_Policy.png')

#comparing best action table with ideal action table:
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