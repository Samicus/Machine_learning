import numpy as np
import random
import math
import h5py
from functools import reduce
import random
from support_scripts import plot_rewards
from support_scripts import binatodeci
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Deep_Q_Network import DQN
from replay_memory import ReplayMemory


class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count
        self.dict = {}
        self.new_state_index = 0
        self.strat = 0  # Argmax action
        self.reward_tots = np.zeros(episode_count)

    def fn_init(self,gameboard):
        self.gameboard=gameboard

        n =  (self.gameboard.N_col * self.gameboard.N_row )
        nr_states = 2**20
        nr_actions = 16

        self.state = np.zeros((self.gameboard.N_row * self.gameboard.N_col + len(self.gameboard.tiles)))
        #self.q_table = np.random.uniform(low=0, high=0.2, size=(nr_states, nr_actions))
        self.q_table = np.zeros((nr_states, nr_actions))

        # Every combination of placement & rotations
        self.action_list = {
             0: [0, 0],
             1: [0, 1],
             2: [0, 2],
             3: [0, 3],
             4: [1, 0],
             5: [1, 1],
             6: [1, 2],
             7: [1, 3],
             8: [2, 0],
             9: [2, 1],
             10: [2, 2],
             11: [2, 3],
             12: [3, 0],
             13: [3, 1],
             14: [3, 2],
             15: [3, 3],
         }

    def fn_load_strategy(self,strategy_file):
        #self.q_table = strategy_file
        pass
    def fn_read_state(self):
        """
        first 4 elements denotes which tile is currently used, The remaining
        elements are the state of the board flattened.

        """
        current_tile = self.gameboard.cur_tile_type # int 0 - 3
        self.state[:4] = [-1, -1, -1, -1]
        self.state[current_tile] = 1
        self.state[4:] = self.gameboard.board.flatten()

    def fn_select_action(self):
        binary_rep_state = np.where(self.state == -1, 0, self.state)
        index = binatodeci(binary_rep_state)
        r = random.random()

        # Choose best action
        if r > self.epsilon:
            self.action_index = np.argmax(self.q_table[index, :])
        else:
            self.action_index = random.randint(0, 15)

        self.action = self.action_list.get(self.action_index)
        return_code = self.gameboard.fn_move(self.action[0], self.action[1])


        # if move is invalid, set the Q_table value to a very low value and choose random actions
        # until a valid action is chosen
        while return_code == 1:
            self.q_table[index, self.action_index] = -np.inf
            # select new random action
            self.action_index = random.randint(0, 15)
            self.action = self.action_list.get(self.action_index)

            return_code = self.gameboard.fn_move(self.action[0], self.action[1])



    def fn_reinforce(self,old_state,reward):

        binary_rep_oldstate = np.where(old_state == -1, 0, old_state)
        binary_rep_newstate = np.where(self.state == -1, 0, self.state)
        old_state_idx = binatodeci(binary_rep_oldstate)
        new_state_idx = binatodeci(binary_rep_newstate)
        max_Q_new = max(self.q_table[new_state_idx, :])

        self.q_table[old_state_idx, self.action_index] += self.alpha*(reward + max_Q_new - self.q_table[old_state_idx, self.action_index])

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode == self.episode_count:

                    plot_rewards(self.reward_tots)
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:

            old_state = np.copy(self.state)
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state, reward)


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE =10
        self.GAMMA = 0.999

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        n_row = self.gameboard.N_row
        n_col = self.gameboard.N_col
        tile_size = self.gameboard.tile_size
        h = n_row + tile_size
        w = n_col
        self.state = torch.zeros((h, w))
        self.nr_actions = n_col*4
        self.policy_net = DQN(h, w, self.nr_actions).to(self.device)
        self.target_net = DQN(h, w, self.nr_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.parameters())
        self.memory = ReplayMemory(self.replay_buffer_size)


    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        curtile = self.gameboard.tiles[self.gameboard.cur_tile_type][self.gameboard.tile_orientation]

        n_row = self.gameboard.N_row
        n_col = self.gameboard.N_col
        tile_size = self.gameboard.tile_size
        state = np.zeros((n_row + tile_size, n_col))
        state[:n_row, :] = self.gameboard.board
        state[n_row:, :] = -1
        for xLoop in range(len(curtile)):
            state[self.gameboard.tile_y + curtile[xLoop][0]:self.gameboard.tile_y + curtile[xLoop][1],
            (xLoop + self.gameboard.tile_x) % self.gameboard.N_col] = 1
        self.state = torch.from_numpy(state)

    def fn_select_action(self):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.episode/self.EPS_DECAY)
        self.episode += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(self.state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.nr_actions)]], device=self.device, dtype=torch.long)
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 < tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self,batch):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                self.fn_reinforce(batch)

                if self.episode_count % self.sync_target_episode_count == 0:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network

class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()