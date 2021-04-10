import numpy as np
import random
import math
import h5py
from functools import reduce
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from ddqn_model import DeepDoubleQNetwork
from memory import ExperienceReplay
from support_scripts import calculate_q
from support_scripts import calculate_q_targets
from support_scripts import sample_batch_and_calculate_loss
from support_scripts import calculate_loss
from support_scripts import plot_rewards



from numpy import asarray
from numpy import savetxt

"""
from Deep_Q_Network import DQN
from replay_memory import ReplayMemory
from support_scripts import plot_rewards
from support_scripts import binatodeci
"""

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
            print(self.episode)
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
        self.sync_target_episode_count=sync_target_episode_count    # tau
        self.episode=0
        self.episode_count=episode_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_idx = 0
        self.Transition =  namedtuple("Transition", ["s", "a", "r", "next_s", "term_state"])
        self.reward_tots = np.zeros(episode_count)
        self.alpha = alpha
        self.gamma = 0.99
        self.update_count = 0


    def fn_init(self,gameboard):
        self.gameboard=gameboard
        self.n_row = self.gameboard.N_row
        self.n_col = self.gameboard.N_col
        tile_size = self.gameboard.tile_size
        h = self.n_row + tile_size
        w = self.n_col

        self.fn_read_state()

        nr_states = len(self.state[0])
        self.nr_actions = self.n_col*4

        self.replay_buffer = ExperienceReplay(device=self.device, num_states=nr_states)
        self.model = DeepDoubleQNetwork(self.device, nr_states,self.nr_actions, lr=0.007)

        idx = 0
        self.action_dir = {}
        
        for i in range(4):
            for col in range(self.n_col):
                self.action_dir[idx] = [col, i]
                idx += 1

        self.invalid_actions = self.get_valid_actions()

    def get_valid_actions(self):
        invalid_actions = {}
        for j in range(4):
            save_ind = []
            for i in range(len(self.action_dir)):
                self.gameboard.cur_tile_type = j
                action = self.action_dir.get(i)
                return_code = self.gameboard.fn_move(action[0], action[1])
                if return_code == 1:
                    save_ind.append(i)
            invalid_actions[j] = np.array(save_ind)

        return invalid_actions


    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        """
        curtile = self.gameboard.tiles[self.gameboard.cur_tile_type][self.gameboard.tile_orientation]

        n_row = self.gameboard.N_row
        n_col = self.gameboard.N_col
        tile_size = self.gameboard.tile_size
        state = np.zeros((n_row + tile_size, n_col))
        state[:n_row, :] = self.gameboard.board
        state[n_row:, :] = -1
        for xLoop in range(len(curtile)):
            #state[self.gameboard.tile_y + curtile[xLoop][0]:self.gameboard.tile_y + curtile[xLoop][1],
            state[self.gameboard.tile_y + curtile[xLoop][0]:self.gameboard.tile_y + curtile[xLoop][1],

            (xLoop + self.gameboard.tile_x) % self.gameboard.N_col] = 1
        self.state = np.zeros((1,n_row+tile_size, n_col))
        self.state[0, :, :] = state
        self.state = torch.Tensor(self.state).to(self.device)
        """
        self.state = np.zeros(self.n_row*self.n_col + 4)
        current_tile = self.gameboard.cur_tile_type # int 0 - 3
        self.state[:4] = [-1, -1, -1, -1]
        self.state[current_tile] = 1
        self.state[4:] = self.gameboard.board.flatten()
        self.state = self.state[None, :]
        self.state = torch.Tensor(self.state).to(self.device)

    def fn_select_action(self):
        sample = random.random()

        eps_threshold = max(self.epsilon, 1 - self.episode / self.epsilon_scale)
        if sample > eps_threshold:

            q_online = calculate_q(self.model.offline_model, self.state, self.device)
            q_online[0][self.invalid_actions.get(self.gameboard.cur_tile_type)] = -np.inf
            self.action_idx = np.argmax(q_online[0])
            action = self.action_dir.get(self.action_idx)
            return_code = self.gameboard.fn_move(action[0], action[1])
        else:
            self.action_idx = random.randint(0, len(self.action_dir) -1)
            action = self.action_dir.get(self.action_idx)
            return_code = self.gameboard.fn_move(action[0], action[1])
            while return_code == 1:
                self.action_idx = random.randint(0, len(self.action_dir) - 1)
                action = self.action_dir.get(self.action_idx)
                return_code = self.gameboard.fn_move(action[0], action[1])
        return self.action_idx

    def fn_reinforce(self):

        loss = sample_batch_and_calculate_loss(
            self.model, self.replay_buffer, self.batch_size, self.gamma, self.device
        )
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()



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
                data = asarray(self.reward_tots)
                savetxt('data.csv', data, delimiter=',')
                plot_rewards(self.reward_tots)
                raise SystemExit(0)
            else:

                self.gameboard.fn_restart()
                
        else:


            action = self.fn_select_action()
            action = torch.tensor([[action]], device=self.device)
            old_state = np.copy(self.state)
            reward = self.gameboard.fn_drop()
            reward = torch.tensor([[reward]], device=self.device)
            self.fn_read_state()
            next_state = np.copy(self.state)

            if self.gameboard.gameover:
                non_terminal_state = False
            else:

                non_terminal_state = True

            self.reward_tots[self.episode] += reward


            self.replay_buffer.add(self.Transition(old_state, action, reward, next_state, non_terminal_state))
            if self.replay_buffer.buffer_length >  self.replay_buffer_size:
                if self.update_count % self.sync_target_episode_count == 0:
                    self.model.update_target_network()
                self.fn_reinforce()
                self.update_count += 1


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