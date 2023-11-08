import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    GOAL_REWARD = 30
    IMPOSSIBLE_REWARD = -1
    MINOTAUR_REWARD = -10

    def __init__(self, maze, weights=None, random_rewards=False, horizon=20):
        """ Constructor of the environment Maze.
        """
        # maze structure
        self.maze = maze

        # define possible actions
        self.actions = self.__actions()
        self.minotaur_actions=self.__minotaur_actions()

        # define possible states
        self.states, self.map, self.minotaur_position = self.__states()
        self.exit = self.minotaur_position
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)

        self.horizon = horizon

        # define transition probabilities matrix
        self.transition_probabilities = self.__transitions()

        self.positions = {self.minotaur_position: self.MINOTAUR_REWARD}

        # define rewards
        self.rewards = self.__rewards()
        self.dynamic_rewards = self.__dynamic_rewards()



    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __minotaur_actions(self):
        actions = dict()
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        end = False
        minotaur_position = None
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] != 1:
                    states[s] = (i, j)
                    map[(i, j)] = s
                    s += 1
                    if self.maze[i, j] == 2:
                        minotaur_position = (i, j)
        return states, map, minotaur_position

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """

        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row, col] == 1)
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state
        else:
            return self.map[(row, col)]

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s, a)
                transition_probabilities[next_s, s, a] = 1
        return transition_probabilities

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))
        # If the rewards are not described by a weight matrix
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s, a)
                # Reward for hitting a wall
                if s == next_s and a != self.STAY:
                    rewards[s, a] = self.IMPOSSIBLE_REWARD
                # Reward for reaching the exit
                elif s == next_s and self.maze[self.states[next_s]] == 2:
                    rewards[s, a] = self.GOAL_REWARD
                # Reward for taking a step to an empty cell that is not the exit
                else:
                    rewards[s, a] = self.STEP_REWARD

        return rewards

    def __dynamic_rewards(self):

        rewards = np.copy(self.rewards)
        dynamic_rewards = rewards[:, :, np.newaxis].repeat(self.horizon, axis=2)

        for t in range(self.horizon):
            new_positions = {}
            for position in self.positions:
                possible_positions = self.__possible_minotaur_positions(position)
                reward = self.positions[position]/len(possible_positions)
                for elem in possible_positions:
                    if elem in new_positions:
                        new_positions[elem] += reward
                    else:
                        new_positions[elem] = 10+reward
            for position in new_positions:
                sa_list = self.__states_actions(position)
                for elem in sa_list:
                    dynamic_rewards[elem[0],elem[1],t] = new_positions[position]
            self.positions = new_positions

        return dynamic_rewards

    def __possible_minotaur_positions(self, position):
        possible_positions = []
        for action in self.minotaur_actions:
            row = position[0] + self.minotaur_actions[action][0]
            col = position[1] + self.minotaur_actions[action][1]
            if (row != -1) and (row != self.maze.shape[0]) and (col != -1) and (col != self.maze.shape[1]):
                possible_positions.append((row, col))

        return possible_positions

    def __minotaur_path(self):
        path = []
        pos = self.minotaur_position
        for t in range(self.horizon):
            next_positions = self.__possible_minotaur_positions(pos)
            n = len(next_positions)
            next_move = random.randint(0, n - 1)
            pos = next_positions[next_move]
            path.append(pos)

        return path

    def __states_actions(self, position):
        states_actions = []
        for i, action in enumerate(self.actions):
            row = position[0] + self.actions[action][0]
            col = position[1] + self.actions[action][1]
            if (row != -1) and (row != self.maze.shape[0]) and (col != -1) and (col != self.maze.shape[1]):
                if (row, col) in self.map:
                    states_actions.append((self.map[(row, col)], i))
        return states_actions

    def dynamic_programming(self, horizon):
        """ Solves the shortest path problem using dynamic programming
            :input Maze env           : The maze environment in which we seek to
                                        find the shortest path.
            :input int horizon        : The time T up to which we solve the problem.
            :return numpy.array V     : Optimal values for every state at every
                                        time, dimension S*T
            :return numpy.array policy: Optimal time-varying policy at every state,
                                        dimension S*T
        """

        # The dynamic prgramming requires the knowledge of :
        # - Transition probabilities
        # - Rewards
        # - State space
        # - Action space
        # - The finite horizon
        p = self.transition_probabilities

        self.dynamic_rewards = self.__dynamic_rewards()
        r = self.dynamic_rewards
        n_states = self.n_states
        n_actions = self.n_actions
        T = horizon

        # The variables involved in the dynamic programming backwards recursions
        V = np.zeros((n_states, T + 1))
        policy = np.zeros((n_states, T + 1))
        Q = np.zeros((n_states, n_actions))

        # Initialization
        Q = np.copy(r[:, :, T-1])
        V[:, T] = np.max(Q, 1)
        policy[:, T] = np.argmax(Q, 1)

        # The dynamic programming backwards recursion
        for t in range(T - 1, -1, -1):
            # Update the value function according to the bellman equation
            for s in range(n_states):
                for a in range(n_actions):
                    # Update of the temporary Q values
                    Q[s, a] = r[s, a,t] + np.dot(p[:, s, a], V[:, t + 1])
            # Update by taking the maximum Q value w.r.t the action a
            V[:, t] = np.max(Q, 1)
            # The optimal action is the one that maximizes the Q function
            policy[:, t] = np.argmax(Q, 1)
        return V, policy



    def simulate(self, start, horizon, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        minotaur_path = self.__minotaur_path()
        if method == 'DynProg':
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # minotaur_path.append(self.minotaur_position)
            while t < horizon - 1 and self.states[s]!=self.exit:
                # Move to next state given the policy and the current state
                self.positions = {minotaur_path[t]: self.MINOTAUR_REWARD}
                _, policy = self.dynamic_programming(horizon-t)
                next_s = self.__move(s, policy[s, 0])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])

                # Update time and state for next iteration
                # self.minotaur_position = self.__minotaur_move()
                # minotaur_path.append(self.minotaur_position)
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
        return path, minotaur_path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.next_rewards)



def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.next_rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon / gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V);
        BV = np.max(Q, 1)
        # Show error
        # print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows, cols = maze.shape;
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows, cols = maze.shape;
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows);
        cell.set_width(1.0 / cols);


def animate_solution(maze, path, minotaur_path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')
        grid.get_celld()[(minotaur_path[i])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(minotaur_path[i])].get_text().set_text('Minotaur')
        if i > 0:
            if path[i] == path[i - 1] and path[i] == minotaur_path[0]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player is out')  

            elif path[i] == minotaur_path[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i])].get_text().set_text('Player is dead')
                grid.get_celld()[(minotaur_path[i - 1])].set_facecolor(col_map[maze[minotaur_path[i - 1]]])
                grid.get_celld()[(minotaur_path[i - 1])].get_text().set_text('')
                grid.get_celld()[(path[i - 1])].set_facecolor(col_map[maze[path[i - 1]]])
                grid.get_celld()[(path[i - 1])].get_text().set_text('')
                break
            else:
                grid.get_celld()[(path[i - 1])].set_facecolor(col_map[maze[path[i - 1]]])
                grid.get_celld()[(path[i - 1])].get_text().set_text('')
                grid.get_celld()[(minotaur_path[i - 1])].set_facecolor(col_map[maze[minotaur_path[i - 1]]])
                grid.get_celld()[(minotaur_path[i - 1])].get_text().set_text('')
                                
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
