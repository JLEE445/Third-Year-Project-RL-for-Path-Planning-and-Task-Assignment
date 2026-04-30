#DQN for multi-agent warehouse environment. Agents learn to navigate to the target while avoiding shelves and walls. Rewards are given for reaching the target and penalties for colliding with shelves/walls or taking too long. The code includes experience replay, epsilon-greedy action selection, and target network updates for stable training. Results include cumulative rewards per episode and visualizations of the learned paths.
import numpy as np
import matplotlib.pyplot as plt
from collections import deque #for experience replay
import random
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import colors
import time

cmap = colors.ListedColormap(['white', 'red', 'black', 'green', 'pink', 'purple'])

#initialize environment:
from Warehouses import return_warehouse
warehouse, size = return_warehouse('B') 
warehouse[7,2] = 3 #additional target
warehouse[1,1] = 4 #starting location for agent A
warehouse[1,8] = 5 #starting location for agent B

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
EMPTY, SHELF, WALL, TARGET, AGENT_A, AGENT_B = 0,1,2,3,4,5

#visualise warehouse:
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(warehouse[:,:], cmap = cmap)
title = 'Warehouse Layout ' + str(size) + 'x' + str(size)
ax.title.set_text(title)
plt.savefig('4.Warehouse.png')

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

num_episodes = input("Enter Number of Episodes: ")
if num_episodes == '':
    num_episodes = 500
else:
    num_episodes = int(num_episodes)
num_steps = input("Enter Number of Steps per Episode: ")
if num_steps == '':
    num_steps = 300
else:
    num_steps = int(num_steps)
epsilon = input("Enter Epsilon: ")
if epsilon == '':
    epsilon = 0.9
else:
    epsilon = float(epsilon)
discount_factor = input("Enter Discount Factor: ")
if discount_factor == '':
    discount_factor = 0.95
else:
    discount_factor = float(discount_factor)
learning_rate = input("Enter Learning Rate: ")
if learning_rate == '':
    learning_rate = 0.008
else:
    learning_rate = float(learning_rate)
epsilon_decay = input("Enter Epsilon Decay: ")
if epsilon_decay == '':
    epsilon_decay = 0.9995
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
policy_dqn_A = DQN(size*size, 128, num_actions)
target_dqn_A = DQN(size*size, 128, num_actions)
policy_dqn_B = DQN(size*size, 128, num_actions)
target_dqn_B = DQN(size*size, 128, num_actions)

#initialize replay memory:
replay_memory_A = Replaymemory(replay_memory_size)
replay_memory_B = Replaymemory(replay_memory_size)

#initialize optimizer and loss function:
optimizer_A = torch.optim.Adam(policy_dqn_A.parameters(), lr=learning_rate)
optimizer_B = torch.optim.Adam(policy_dqn_B.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

#def state_to_dqn_input:
def state_to_dqn_input(state, num_states)->torch.Tensor:
    input_tensor = torch.zeros(num_states)
    input_tensor[state] = 1
    return input_tensor

#def optimze:
def optimize(mini_batch, policy_dqn, target_dqn, optimizer):
    num_states = size*size
    current_q_list = []
    target_q_list = []
    
    for state, action, new_state, reward, terminated in mini_batch:
        if terminated:
            target = torch.FloatTensor([reward])
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

#initialize rewards:
rewards = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        #change made: reward normalization - clipping rewards to between (1 and -1) to help with training stability:
        cell = warehouse[i,j]
        if cell == EMPTY:
            rewards[i,j] = -0.05
        elif cell == SHELF:
            rewards[i,j] = -1
        elif cell == WALL:
            rewards[i,j] = -1
        elif cell == TARGET:
            rewards[i,j] = 1


#begin training the neural network:
###################################

start_time = time.time()
rewards_per_episode_A = np.zeros(num_episodes+1)
rewards_per_episode_B = np.zeros(num_episodes+1)
epsilon_per_episode = []

target_dqn_A.load_state_dict(policy_dqn_A.state_dict())
target_dqn_B.load_state_dict(policy_dqn_B.state_dict())

ep = 0
counter = 0
while ep < num_episodes:
    ep += 1
    #initilize state for A and B:
    #agent_A:
    state_i_A, state_j_A = 1,1
    state_A = state_i_A*size + state_j_A
    #agent_B:
    state_i_B, state_j_B = 1,8
    state_B = state_i_B*size + state_j_B

    print("Episode #: %s" % (ep,), end='\r')
    terminated_A, terminated_B = False, False
    for step in range(num_steps):
        if not terminated_A:
            if random.random() < epsilon:
                action_A = random.randint(0, num_actions-1)
            else:
                with torch.no_grad():
                    qvals_A = policy_dqn_A(state_to_dqn_input(state_A, size*size))
                    action_A = qvals_A.argmax().item()
            
            new_state_i_A, new_state_j_A = get_next_location(state_i_A, state_j_A, action_A)
            new_state_A = new_state_i_A*size + new_state_j_A
            reward_A = rewards[new_state_i_A, new_state_j_A]
            rewards_per_episode_A[ep] += reward_A
            if warehouse[new_state_i_A, new_state_j_A] == TARGET:
                terminated_A = True
            replay_memory_A.append((state_A, action_A, new_state_A, reward_A, terminated_A))
            state_i_A, state_j_A = new_state_i_A, new_state_j_A
            state_A = new_state_A
        if not terminated_B:
            if random.random() < epsilon:
                action_B = random.randint(0, num_actions-1)
            else:
                with torch.no_grad():
                    qvals_B = policy_dqn_B(state_to_dqn_input(state_B, size*size))
                    action_B = qvals_B.argmax().item()
            
            new_state_i_B, new_state_j_B = get_next_location(state_i_B, state_j_B, action_B)
            new_state_B = new_state_i_B*size + new_state_j_B
            reward_B = rewards[new_state_i_B, new_state_j_B]
            rewards_per_episode_B[ep] += reward_B
            if warehouse[new_state_i_B, new_state_j_B] == TARGET:
                terminated_B = True
            replay_memory_B.append((state_B, action_B, new_state_B, reward_B, terminated_B))
            state_i_B, state_j_B = new_state_i_B, new_state_j_B
            state_B = new_state_B
    #end of single episode, now optimize network:
    if len(replay_memory_A) >= 32:
        mini_batch = replay_memory_A.sample(32)
        optimize(mini_batch, policy_dqn_A, target_dqn_A, optimizer_A)
    if len(replay_memory_B) >= 32:
        mini_batch = replay_memory_B.sample(32)
        optimize(mini_batch, policy_dqn_B, target_dqn_B, optimizer_B)
    
    #decay epsilon:
    epsilon_per_episode.append(epsilon)
    epsilon = max(min_epsilon, epsilon*epsilon_decay)
    #update target network every 10 episodes:
    if ep % 10 == 0:
        target_dqn_A.load_state_dict(policy_dqn_A.state_dict())
        target_dqn_B.load_state_dict(policy_dqn_B.state_dict())
    
end_time = time.time()
print("Training Complete after %s episodes." % (ep))

#RESULTS SECTION

time_elapsed = end_time-start_time
torch.save(policy_dqn_A.state_dict(), '4.DQN_model_A.pt')
torch.save(policy_dqn_B.state_dict(), '4.DQN_model_B.pt')

print('\n Time taken: ', time_elapsed, ' s')
#print(counter)
#cumulative reward per episode:
fig, ax = plt.subplots(2)
sum_rewards_0 = np.zeros(num_episodes)
sum_rewards_1 = np.zeros(num_episodes)
for x in range(num_episodes):
    start = max(0, x-10)
    sum_rewards_0[x] = np.average(rewards_per_episode_A[start:x+1])
    sum_rewards_1[x] = np.average(rewards_per_episode_B[start:x+1])
ax[0].plot(sum_rewards_0[10:500], label='Agent A', color='pink')
ax[0].plot(sum_rewards_1[10:500], label='Agent B', color='purple')
ax[0].legend()
ax[1].plot(epsilon_per_episode)
ax[0].title.set_text('Rewards per Episode')
ax[1].title.set_text('Epsilon per Episode')
plt.savefig('4.1.Rewards_Per_Episode.png')

#best_actions:
best_actions_A = np.zeros((size, size), dtype=int)
for i in range(size):
    for j in range(size):
        state = i*size + j
        with torch.no_grad():
            qvals = policy_dqn_A(state_to_dqn_input(state, size*size))
            best_actions_A[i,j] = qvals.argmax().item()
best_actions_B = np.zeros((size, size), dtype=int)
for i in range(size):
    for j in range(size):
        state = i*size + j
        with torch.no_grad():
            qvals = policy_dqn_B(state_to_dqn_input(state, size*size))
            best_actions_B[i,j] = qvals.argmax().item()

path_i_A = []
path_j_A = []

i,j = 1,1
running = True
counter = 0
while running and counter < 100:
    old_i, old_j = i,j
    if best_actions_A[i,j] == 0:
        i = i-1
    elif best_actions_A[i,j] == 1:
        j = j+1
    elif best_actions_A[i,j] == 2:
        i = i+1
    elif best_actions_A[i,j] == 3:
        j = j-1
    
    path_i_A.append(old_i)
    path_j_A.append(old_j)
    if warehouse[old_i, old_j] == 3:
        running = False
    counter += 1

path_i_B = []
path_j_B = []

i,j = 1,8
running = True
counter = 0
while running and counter < 100:
    old_i, old_j = i,j
    if best_actions_B[i,j] == 0:
        i = i-1
    elif best_actions_B[i,j] == 1:
        j = j+1
    elif best_actions_B[i,j] == 2:
        i = i+1
    elif best_actions_B[i,j] == 3:
        j = j-1
    
    path_i_B.append(old_i)
    path_j_B.append(old_j)
    if warehouse[old_i, old_j] == 3:
        running = False
    counter += 1

fig, ax = plt.subplots(figsize=(10,10))
fig.patch.set_linewidth(0)
ax.imshow(warehouse[:,:], cmap = cmap)
ax.title.set_text('Path taken')
ax.plot(path_j_A, path_i_A, label='Agent A', color='pink', linewidth=2)
ax.plot(path_j_B, path_i_B, label='Agent B', color='purple', linewidth=2)
ax.legend()
plt.savefig('4.1.Path_Taken.png')