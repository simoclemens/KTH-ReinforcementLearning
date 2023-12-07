# Authors: Simone Clemente (20001104-T332), Gustavo Mazzanti (19991119-T519)
# LAB01: Exercise 2


# Load packages
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
k = env.action_space.n  # number of actions
low, high = env.observation_space.low, env.observation_space.high  # min-max values for states

# vectors initialization
p = 2  # Fourier's order

# Fourier's coefficients used to generate the basis (all combinations)
ETA = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)])
l = np.shape(ETA)[0]

# Weights
W = np.zeros((k, l))

# Parameters
N_episodes = 600  # Number of training episodes
momentum = 0.2  # SGD momentum
epsilon = 0  # Randomization parameter
discount_factor = 1.0  # Value of gamma
eligibility_trace = 0.1
alpha_set = 0.1

counter_reduction = 0

alpha_reduction = True
scaling_basis = True

exercise_mean = False

# Reward
episode_reward_list = []  # Used to save episodes reward
mean_reward_episode_list = []  # Used to calcular mean reward for each episode
std_plus_reward_episode_list = []  # Used to calcular std reward for each episode
std_less_reward_episode_list = []


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# State variables scaling
def scale_state_variables(s, low=low, high=high):
    x = (s - low) / (high - low)
    return x


# Fourier's basis transformation
def basis_function(eta, state):
    bases = np.cos(np.pi * np.matmul(eta, state))
    return bases


# Compute Q_value
def Q_function(state):
    Q = np.dot(W, state)
    return Q


def epsilon_greedy_policy(Q, epsilon):
    """Function to choose action"""
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(np.arange(k))
    else:
        action = np.argmax(Q)
    return action


def scaling_basis_function(ETA):
    eta_normalitzation = np.linalg.norm(ETA, 2, 1)
    eta_normalitzation[eta_normalitzation == 0] = 1  # if ||eta_i||=0 then alpha_i=alpha
    alpha = np.divide(alpha_set, eta_normalitzation)
    return alpha


def update_z(z, discount_factor, eligibility_trace, gradient):
    z = discount_factor * eligibility_trace * z
    z[action, :] += gradient
    z = np.clip(z, -5, 5)
    return z

def update_w(W, z, momentum, velocity, delta_t, alpha):
    if scaling_basis:
        velocity = momentum * velocity + delta_t * np.matmul(z, np.diag(alpha))
    else:
        velocity = momentum * velocity + delta_t * alpha * z
    W = W + velocity
    return W


# Set alpha scaling
if scaling_basis:
    alpha = scaling_basis_function(ETA)
else:
    alpha = alpha_set


# TRAINING
for i in range(N_episodes):

    done = False
    state = scale_state_variables(env.reset()[0], low, high)
    total_episode_reward = 0.
    z = np.zeros((k, l))  # Create matrix z
    velocity = np.zeros((k, l))  # Create velocity vector
    while not done:

        # compute the Fourier's basis of the state
        basis = basis_function(ETA, state)  # Define the basis
        # compute the Q_value of the state
        Q = Q_function(basis)  # This vector results on [Q(action1,n bases),Q(action2,n bases),Q(action3,bases)]

        action = epsilon_greedy_policy(Q, epsilon)  # Choose Action
        next_state, reward, done, _, _ = env.step(action)
        next_state = scale_state_variables(next_state)

        # Update state for next iteration
        state = next_state
        next_basis = basis_function(ETA, next_state)
        Q_next = Q_function(next_basis)  # Update Q(state,action)
        action_next = np.argmax(Q_next)

        # Update episode reward
        total_episode_reward += reward

        # compute temporal distance error
        delta_t = reward + discount_factor * Q_next[action_next] - Q[action]

        # update z
        z = update_z(z, discount_factor, eligibility_trace, basis)
        # update W
        W = update_w(W, z, momentum, velocity, delta_t, alpha)

    if alpha_reduction and total_episode_reward > -200:
        alpha *= 0.6
        # alpha *= (0.9 - 0.05 * (-200 / total_episode_reward))

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

# Close environment
env.close()

results = {'W': W, "N": ETA}
with open("weights.pkl", 'wb') as file:
    pickle.dump(results, file)