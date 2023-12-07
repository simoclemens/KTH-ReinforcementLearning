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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#
from collections import deque

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, Agent

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length=15000):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            print("Buffer too small")
        indices = np.random.choice(len(self.buffer), n-1, replace=False)

        batch = [self.buffer[i] for i in indices]
        batch.append(self.buffer[-1])

        return zip(*batch)


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def getEpsilon(eps_max, eps_min, n_ep, N, mode="linear"):
    if mode == "linear":
        return max(eps_min, eps_max - (eps_max - eps_min) * (n_ep - 1) / (N * 0.95 - 1))
    elif mode == "exp":
        return max(eps_min, eps_max ** ((eps_min / eps_max) ** ((n_ep - 1) / (N * 0.95 - 1))))
    else:
        return 0


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

buffer_size = 10000
buffer = ExperienceReplayBuffer(maximum_length=buffer_size)

# Parameters
N_episodes = 275  # Number of episodes
discount_factor = 0.95  # Value of the discount factor
n_ep_running_average = 20  # Running average of 50 episodes
eps_max = 0.95  # epsilon
eps_min = 0.05
lr = 0.0001
gamma = 0.95
n_actions = env.action_space.n  # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
N = 16 # Training batch
C = int(buffer_size/N)  # Set target network to main network every C iterations.

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []  # this list contains the total reward per episode
episode_number_of_steps = []  # this list contains the number of steps per episode

# Agent initialization
# agent = RandomAgent(n_actions)
agent = Agent(n_actions, lr)

### TRAINING PROCESS
# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i, episode in enumerate(EPISODES):

    eps_k = getEpsilon(eps_max, eps_min, i, N_episodes)

    done = False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0

    while not done:
        # Take action with eps-greedy policy
        action = agent.forward(state, eps_k)
        # Compute next state and reward
        next_state, reward, done, _, _ = env.step(action)
        # Update episode reward
        total_episode_reward += reward

        # Store experience tuple in buffer
        exp = (state, action, reward, next_state, done)

        buffer.append(exp)

        if len(buffer) > 7500:
            # Sample a batch from the experience buffer
            states, actions, rewards, next_states, dones = buffer.sample_batch(N)

            # Compute target values
            y = np.zeros(len(states))
            for j in range(len(states)):
                if not dones[j]:
                    next_state_tensor = torch.tensor(np.array([next_states[j]]),
                                                     requires_grad=False,
                                                     dtype=torch.float32)
                    out = agent.target_network(next_state_tensor)
                    max_Q = torch.max(out)
                    y[j] = rewards[j] + gamma * max_Q
                else:
                    y[j] = rewards[j]

            # perform backwards propagation
            agent.backward(states, actions, y, N)

            # Update episode reward
            total_episode_reward += reward

        # set target network to main network
        if t % C == 0:
            agent.updateTargetNetwork()

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            episode, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

torch.save(agent.main_network.to("cpu"),"neural-network-1.pt")


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
