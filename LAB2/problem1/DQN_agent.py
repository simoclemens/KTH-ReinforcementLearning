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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_layer = nn.Linear(input_size, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(64, output_size)
        print(self.hidden_layer)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        out = self.output_layer(x)

        return out


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, n_actions: int, lr):
        self.gpu = "cuda"
        self.cpu = "cpu"
        self.n_actions = n_actions
        self.last_action = None
        self.main_network = QNetwork(8, n_actions).to(self.gpu)
        self.target_network = QNetwork(8, n_actions)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=lr)

    def forward(self, state: np.ndarray, epsilon):
        ''' Performs a forward computation '''

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            state = torch.tensor(state).to(self.gpu)
            out = self.main_network(state).to(self.cpu)
            action = torch.argmax(out).item()
        return action

    def backward(self, states, actions, target_states, N):
        ''' Performs a backward pass on the network '''

        self.optimizer.zero_grad()
        # states, actions, rewards, next_states, dones = buffer.sample_batch(3)
        Q_values = self.main_network(torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32).to(self.gpu))
        Q_tensor = torch.zeros(N, requires_grad=False, dtype=torch.float32)

        # Get Q(s_i, a_i)
        for i in range(N):
            Q_tensor[i] = Q_values[i, actions[i]]

        Q_tensor = Q_tensor.to(self.gpu)

        target_values = torch.tensor(target_states, requires_grad=True, dtype=torch.float32).to(self.gpu)

        loss = nn.functional.mse_loss(Q_tensor, target_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 0.5)
        self.optimizer.step()

    def updateTargetNetwork(self):
        self.target_network.load_state_dict(self.main_network.state_dict())


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action
