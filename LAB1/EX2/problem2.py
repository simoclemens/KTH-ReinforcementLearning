# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

# Define Fourier orders
orders = [[1, 1], [1, 2]]

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

n = len(env.observation_space.low)  # State space dimensionality
m = env.action_space.n               # Number of actions
p = len(orders)
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 1000  # Number of episodes to run for training
gamma = 0.3  # Value of gamma
lamda = 0.5  # Lamda value
alpha = 0.3  # Learning rate
epsilon = 0.5  # Greedy probability
momentum = 0.9  # Momentum parameter

# Reward
episode_reward_list = []  # Used to save episodes reward

# Initialize matrices
w = np.random.randn(m, p)
v = np.random.randn(m, p)
c = np.array(orders)

c_norms = np.linalg.norm(c, axis=1)

# FUNCTIONS
# Fourier's basis extraction
def fourier_basis(state):
    f_c = np.dot(c, state.T)
    fourier_eq = np.cos(np.pi * f_c)
    return fourier_eq

# Q values computation
def q_function(state):
    f_eq = fourier_basis(state)
    q_v = np.dot(w, f_eq.T)
    return q_v

# Z matrix update
def update_z(z, gamma, lamda, gradient, action):
    z = gamma*lamda*z
    z[action, :] = z[action, :] + gradient
    return z

# Weights update (SGD)
def update_w(w, z, v, alpha, delta, momentum):

    #weighted_alpha = np.full(2, alpha)

    v = momentum*v + alpha*delta*z
    w = w + m*v + alpha*delta*z

    return w

def epsilon_greedy_policy(Q_state, epsilon):

    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return np.random.choice(m)
    else:
        # Exploit: choose the action with the highest Q value
        return np.argmax(Q_state)


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''

    x = (s - low) / (high - low)
    return x

# Training process
for i in range(N_episodes):
    print("EPISODE:" + str(i))

    # Reset environment data
    done = False
    # Set total episode reward
    total_episode_reward = 0.
    # Reset eligibility matrix
    z = np.zeros((m, p))
    # Scale state variables
    state = scale_state_variables(env.reset()[0])
    # Compute Q values
    q_state = q_function(state)
    # Select action with epsilon greedy policy
    action = epsilon_greedy_policy(q_state, epsilon)
    move = 0

    while not done and move < 200:
        move += 1

        # Take the chosen action
        next_state, reward, done, _, _ = env.step(action)
        # Scale state variables
        next_state = scale_state_variables(next_state)
        # Compute Q values
        q_next_state = q_function(next_state)
        # Select next action with epsilon-greedy policy
        next_action = epsilon_greedy_policy(q_state, epsilon)

        # Compute temporal difference error
        delta = reward + gamma*q_next_state[next_action] - q_state[action]

        # Compute the gradient -> in our case it is represented by the Fourier's coefficient themselves
        gradient = fourier_basis(state)

        # Update z matrix
        z = update_z(z, gamma, lamda, gradient, action)

        # Clipping of z to avoid exploding gradient
        z = np.clip(z, -5, 5)

        # Update weights
        w = update_w(w, z, v, alpha, delta, momentum)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        action = next_action

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

results = {'W': w, "N": c}
with open("weights.pkl", 'wb') as file:
    pickle.dump(results, file)

"""
# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
"""
