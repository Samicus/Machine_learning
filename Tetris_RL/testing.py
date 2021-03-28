import gameboardClass
import agentClass
import numpy as np
from functools import reduce
N_row=4
N_col=4
tile_size=2
max_tile_count=50
stochastic_prob=0

alpha=0.2
epsilon=0
episode_count=1000

agent=agentClass.TQAgent(alpha,epsilon,episode_count)
gameboard=gameboardClass.TGameBoard(N_row,N_col,tile_size,max_tile_count,agent,stochastic_prob)

agent.fn_init(gameboard)
gameboard.fn_new_tile()

gameboard.fn_drop()
gameboard.fn_new_tile()
gameboard.fn_drop()

curTile=gameboard.tiles[gameboard.cur_tile_type][gameboard.tile_orientation]
gameboard.fn_new_tile()
#for xLoop in range(len(curTile)):
#    for yLoop in range(curTile[xLoop][0],curTile[xLoop][1]):
#        agent.state[yLoop, xLoop+gameboard.tile_x] = 1
print(np.shape(agent.q_table))
print(np.shape(agent.state))
print(np.shape(gameboard.board.flatten()))
current_tile = gameboard.cur_tile_type  # int 0 - 3
agent.state[:4] = [-1, -1, -1, -1]
agent.state[current_tile] = 1
agent.state[4:] = gameboard.board.flatten()

#pygame.draw.rect(screen,COLOR_RED,[101+20*((xLoop+gameboard.tile_x)%gameboard.N_col),81+20*(gameboard.N_row-(yLoop+gameboard.tile_y)),18,18])
