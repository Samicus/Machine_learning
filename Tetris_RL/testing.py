import gameboardClass
import agentClass
import numpy as np
from functools import reduce

N_row = 8
N_col = 8
tile_size = 4
max_tile_count = 50
stochastic_prob = 0

alpha = 0.001
epsilon = 0.001
episode_count = 10000

epsilon_scale = 50000

replay_buffer_size = 10000
batch_size = 32
sync_target_episode_count = 100


agent = agentClass.TDQNAgent(alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size,
                                 sync_target_episode_count, episode_count)
gameboard=gameboardClass.TGameBoard(N_row,N_col,tile_size,max_tile_count,agent,stochastic_prob)

#agent.fn_init(gameboard)
gameboard.fn_new_tile()
gameboard.fn_drop()
gameboard.fn_new_tile()
gameboard.fn_drop()
gameboard.fn_new_tile()
gameboard.fn_move(0,0)
#gameboard.fn_new_tile()
#gameboard.fn_drop()
curtile = gameboard.tiles[gameboard.cur_tile_type][gameboard.tile_orientation]
#gameboard.fn_new_tile()
state = np.zeros((12, 8))

state[:8, :] = gameboard.board
state[8:, :] = -1


for xLoop in range(len(curtile)):

    state[gameboard.tile_y +curtile[xLoop][0]:gameboard.tile_y + curtile[xLoop][1],(xLoop + gameboard.tile_x) % gameboard.N_col] = 1
print(state)
print(gameboard.tile_y)
#pygame.draw.rect(screen,COLOR_RED,[101+20*((xLoop+gameboard.tile_x)%gameboard.N_col),81+20*(gameboard.N_row-(yLoop+gameboard.tile_y)),18,18])
