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
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DDPG_soft_updates import soft_updates

class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''

    def __init__(self, lr_a, lr_c):
        self.gpu = "cuda"
        self.cpu = "cpu"
        self.act_main = ActNetwork(8).to(self.gpu)
        self.crit_main = CritNetwork(8).to(self.gpu)
        self.act_target = ActNetwork(8).to(self.gpu)
        self.crit_target = CritNetwork(8).to(self.gpu)
        self.optimizer_act = optim.Adam(self.act_main.parameters(), lr=lr_a)
        self.optimizer_crit = optim.Adam(self.crit_main.parameters(), lr=lr_c)

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        state = torch.tensor(state).to(self.gpu)
        action = self.act_main(state).to(self.cpu)
        return np.array(action.tolist())

    def backward_actor(self, states, actions, y, N):
        ''' Performs a backward pass on the network '''

        self.optimizer_act.zero_grad()
        # states, actions, rewards, next_states, dones = buffer.sample_batch(3)
        Q_values = self.crit_main(
            torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32).to(self.gpu),
            torch.tensor(np.array(actions), requires_grad=True, dtype=torch.float32).to(self.gpu)
        )

        target_values = torch.tensor(y, requires_grad=True, dtype=torch.float32).to(self.gpu)

        loss = nn.functional.mse_loss(Q_values.squeeze(), target_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.act_main.parameters(), 1)
        self.optimizer_act.step()

    def backward_critic(self, states, N):
        ''' Performs a backward pass on the network '''

        self.optimizer_crit.zero_grad()
        actions = self.act_main(torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32).to(self.gpu))

        Q_values = self.crit_main(
            torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32).to(self.gpu),
            actions.to(self.gpu)
        )

        zero_tensor = torch.zeros(N, requires_grad=False, dtype=torch.float32).to(self.gpu)

        loss = -nn.functional.l1_loss(Q_values.squeeze(), zero_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.act_main.parameters(), 1)
        self.optimizer_crit.step()

    def update_target(self, tau):
        self.act_target = soft_updates(self.act_main, self.act_target, tau)
        self.crit_target = soft_updates(self.crit_main, self.crit_target, tau)

class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


class ActNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_layer = nn.Linear(input_size, 400)
        self.hidden_layer = nn.Linear(400, 200)
        self.output_layer = nn.Linear(200, 2)

        self.activation = nn.ReLU()
        self.out_activation = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        out = self.out_activation(x)

        return out


class CritNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_layer = nn.Linear(input_size, 400)
        self.hidden_layer = nn.Linear(400 + 2, 200)
        self.output_layer = nn.Linear(200, 1)

        self.activation = nn.ReLU()
        self.out_activation = nn.Tanh()

    def forward(self, x, a):
        x = self.input_layer(x)
        x = torch.cat([x, a], dim=1)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        out = self.out_activation(x)

        return out
