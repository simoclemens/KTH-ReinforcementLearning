import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

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
    STEP_REWARD = -1
    GOAL_REWARD = 100
    IMPOSSIBLE_REWARD = -100
    MINOTAUR_REWARD = -100

    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        # maze structure
        self.maze = maze
        # define possible player actions
        self.actions = self.__actions()

        #define possible minotaur actions
        self.minotaur_actions=self.__minotaur_actions()

        # define possible states
        self.states, self.map = self.__states()

        self.n_actions = len(self.actions)
        self.n_states = len(self.states)

        self.exit = None
        self.key = None

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
        s = 0
        for i_p in range(self.maze.shape[0]):
            for j_p in range(self.maze.shape[1]):
                if self.maze[i_p, j_p] != 1:
                    if self.maze[i_p, j_p] == 2:
                        self.exit = (i_p, j_p)
                    elif self.maze[i_p, j_p] == 3:
                        self.key = (i_p, j_p)
                    for i_m in range(self.maze.shape[0]):
                        for j_m in range(self.maze.shape[1]):
                            states[s] = ((i_p, j_p), (i_m, j_m), 0)
                            map[((i_p, j_p), (i_m, j_m), 0)] = s
                            s += 1
                            states[s] = ((i_p, j_p), (i_m, j_m), 1)
                            map[((i_p, j_p), (i_m, j_m), 1)] = s
                            s += 1

        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        current_s = self.states[state]
        m_row = current_s[1][0]
        m_col = current_s[1][1]
        f_k = current_s[2]

        # Compute the future position given current (state, action)
        row = current_s[0][0] + self.actions[action][0]
        col = current_s[0][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row, col] == 1)
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state
        else:
            if (row, col) == self.key:
                return self.map[((row, col), (m_row, m_col), 1)]
            else:
                return self.map[((row, col), (m_row, m_col), f_k)]

    def __get_reward (self, state, action):
        old_state = self.states[state]
        n_state = self.__move(state, action)
        new_state = self.states[n_state]
        m_pos = self.__possible_minotaur_positions(old_state[1])
        if new_state[0] in m_pos:
            return self.MINOTAUR_REWARD
        else:
            if new_state[2] == 1 and old_state[2] == 0:
                return self.GOAL_REWARD
            elif new_state[1] == self.exit and old_state[2] == 1:
                return self.GOAL_REWARD
            elif state == n_state and action != self.STAY:
                return self.IMPOSSIBLE_REWARD
            else:
                return self.STEP_REWARD

    def __possible_minotaur_positions(self, position):
        possible_positions = []
        for action in self.minotaur_actions:
            row = position[0] + self.minotaur_actions[action][0]
            col = position[1] + self.minotaur_actions[action][1]
            if (row != -1) and (row != self.maze.shape[0]) and (col != -1) and (col != self.maze.shape[1]):
                possible_positions.append((row, col))

        return possible_positions

    def __minotaur_move(self, state):
        p_pos = self.states[state][0]
        m_pos = self.states[state][1]
        next_positions = self.__possible_minotaur_positions(m_pos)
        n = len(next_positions)

        p = random.uniform(0, 1)

        if p <= 0.35:
            pos = get_closer_pos(p_pos, next_positions)
        else:
            next_move = random.randint(0, n - 1)
            pos = next_positions[next_move]

        return pos


    def simulate(self, start, policy, method):

        return None
    
    def simulate_q_learning(self, alpha, gamma, epsilon, start = (0, 0), testing=False, N = 1000):
        Q = np.zeros((self.states, self.actions))
        Q_c = Q.copy()
        for k in range(N):
            death_prob = 1/50
            life = np.random.geometric(death_prob)
            state = self.map[(start, self.exit, 0)]
            t = 0
            while t < life and self.states[state][0] != self.exit and  self.states[state][1] != self.states[state][0] :
                if testing :
                    action = np.argmax(Q[state])
                    next_state = self.__move(state, action)
                else :
                    rand = np.random.uniform(0, 1)
                    if rand < epsilon:
                        action = np.random.randint(0, 4)
                    else:
                        action = np.argmax(Q[state])
                    reward = self.__get_reward(state, action)
                    next_state = self.__move(state, action)
                    Q_c[state, action ] += 1
                    Q = self.__Q_learning(Q, state, action, reward, next_state, 1/Q_c[state, action]**alpha, gamma)
                state = next_state
                t += 1
        return Q


    def __Q_learning(self, Q, state, action, reward, next_state, alpha, gamma):
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        return Q
        


def get_closer_pos(p_pos, m_positions):

    min = int("+inf")
    new_pos = None
    for m_pos in m_positions:
        dist = abs(p_pos[0]-m_pos[0]) + abs(p_pos[1]-m_pos[1])
        if dist < min:
            new_pos = m_pos
            min = dist

    return new_pos


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
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


def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

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
        if i > 0:
            if path[i][0] == path[i - 1][0]:
                grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0])].get_text().set_text('Player is out')
                grid.get_celld()[(path[i - 1][1])].set_facecolor(col_map[maze[path[i - 1][1]]])
                grid.get_celld()[(path[i - 1][1])].get_text().set_text('')
                break
            elif path[i][0] == path[i][1]:
                grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i][0])].get_text().set_text('Player is catched')
                break
            else:
                grid.get_celld()[(path[i - 1][0])].set_facecolor(col_map[maze[path[i - 1][0]]])
                grid.get_celld()[(path[i - 1][0])].get_text().set_text('')
                grid.get_celld()[(path[i - 1][1])].set_facecolor(col_map[maze[path[i - 1][1]]])
                grid.get_celld()[(path[i - 1][1])].get_text().set_text('')
        
        grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0])].get_text().set_text('Player')
        grid.get_celld()[(path[i][1])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(path[i][1])].get_text().set_text('Minotaur')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
