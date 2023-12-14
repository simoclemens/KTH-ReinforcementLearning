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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#
from collections import deque
import os

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, Agent

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
        indices = np.random.choice(len(self.buffer), n - 1, replace=False)

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


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

dim_state = len(env.observation_space.high)  # State dimensionality

# Parameters
N_episodes = 300  # Number of episodes

n_ep_running_average = 50  # Running average of 50 episodes
lr_a = 0.00005
lr_c = 0.0005
gamma = 0.99
buffer_size = 30000
N = 64  # Training batch

d = 2
mu = 0.15
sigma = 0.2

tau = 0.001

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Buffer initialization
buffer = ExperienceReplayBuffer(maximum_length=buffer_size)

# Agent initialization
# agent = RandomAgent(m)
agent = Agent(lr_a=lr_a, lr_c=lr_c)

while len(buffer) < buffer_size:
    done = False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0

    while not done:
        # Take a random action
        action = np.random.uniform(-1, 1, size=(2,))

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            done = True

        z = (state, action, reward, next_state, done)

        buffer.append(z)


# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:

    # Reset enviroment data
    done = False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    n_t = 0

    while not done:
        # Take a random action
        action = agent.forward(state)

        w_t = np.random.normal(mu, sigma, size=(2,))
        n_t = -mu*n_t + w_t

        action = action + n_t
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            done = True

        # Update episode reward
        total_episode_reward += reward

        z = (state, action, reward, next_state, done)

        buffer.append(z)


        # Sample a batch from the experience buffer
        states, actions, rewards, next_states, dones = buffer.sample_batch(N)

        # Compute target values
        y = np.zeros(len(states))
        for j in range(len(states)):
            if not dones[j]:
                next_state_tensor = torch.tensor(np.array([next_states[j]]),
                                                 requires_grad=False,
                                                 dtype=torch.float32).to("cuda")

                target_action = agent.act_target(next_state_tensor)
                Q_target = agent.crit_target(next_state_tensor, target_action)
                y[j] = rewards[j] + gamma * Q_target
            else:
                y[j] = rewards[j]

        # perform backwards propagation
        agent.backward_actor(states, actions, y, N)

        # Update episode reward
        total_episode_reward += reward

        # set target network to main network
        if t % d == 0:
            agent.backward_critic(states, N)
            agent.update_target(tau)

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

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
