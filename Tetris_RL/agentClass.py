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
from collections import namedtuple





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
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_idx = 0
        self.Transition = namedtuple('Transition',
                                      ('state', 'action', 'next_state', 'reward'))
        self.GAMMA = 1 - alpha
        self.reward_tots = np.zeros(episode_count)
        self.alpha = alpha
        """
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE =10
        
        """

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        n_row = self.gameboard.N_row
        n_col = self.gameboard.N_col
        tile_size = self.gameboard.tile_size
        h = n_row + tile_size
        w = n_col
        self.state = self.fn_read_state
        self.nr_actions = n_col*4
        self.policy_net = DQN(h, w, self.nr_actions).to(self.device)
        self.target_net = DQN(h, w, self.nr_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(self.replay_buffer_size)
        idx = 0
        self.action_dir = {}

        for i in range(4):
            for col in range(n_col):
                self.action_dir[idx] = [col, i]
                idx += 1
        self.optimizer = optim.Adam(lr=self.alpha, params=self.policy_net.parameters())

        
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
            #state[self.gameboard.tile_y + curtile[xLoop][0]:self.gameboard.tile_y + curtile[xLoop][1],
            state[self.gameboard.tile_y + curtile[xLoop][0]:self.gameboard.tile_y + curtile[xLoop][1],

            (xLoop + self.gameboard.tile_x) % self.gameboard.N_col] = 1
        self.state = np.zeros((1, 1,n_row+tile_size, n_col))
        self.state[0, 0, :, :] = state
        self.state = torch.Tensor(self.state).to(self.device)

    def fn_select_action(self):
        sample = random.random()
        #eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.episode/self.EPS_DECAY)
        eps_threshold = max(self.epsilon, 1-self.episode/self.epsilon_scale)

        if sample > eps_threshold:
            with torch.no_grad():
                self.action_idx = self.policy_net(self.state).max(1)[1].view(1, 1).item()
                action = self.action_dir.get(self.action_idx)
                return_code = self.gameboard.fn_move(action[0], action[1])
                while return_code == 1:
                    self.action_idx = random.randint(0,len(self.action_dir)-1)
                    action = self.action_dir.get(self.action_idx)
                    return_code = self.gameboard.fn_move(action[0], action[1])
        else:
            return_code = 1
            while return_code == 1:

                self.action_idx = random.randint(0, len(self.action_dir) -1)
                action = self.action_dir.get(self.action_idx)
                return_code = self.gameboard.fn_move(action[0], action[1])

            #torch.tensor([[random.randrange(self.nr_actions)]], device=self.device, dtype=torch.long)


    def fn_reinforce(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)


        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values *self.GAMMA) + reward_batch

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


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

            
            old_state = self.state
            reward = self.gameboard.fn_drop()
            reward = torch.tensor([[reward]], device=self.device)
            self.fn_read_state()
            next_state =  self.state 
            self.fn_select_action()
            action = torch.tensor([[self.action_idx]], device=self.device)

            self.memory.push(old_state, action, next_state, reward)
            self.reward_tots[self.episode] += reward
            
            if (len(self.memory) >= self.replay_buffer_size):
                self.fn_reinforce()
                if ((self.episode % self.sync_target_episode_count)==0):
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                   
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