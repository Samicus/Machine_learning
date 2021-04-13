from collections import namedtuple
import random

Transitions = namedtuple('Transition',
                         ('state', 'action', 'next_state', 'reward', 'term_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capicity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        #Saves a transition
        if len(self.memory) < self.capicity:
            self.memory.append(None)
        self.memory[self.position] = Transitions(*args)
        self.position = (self.position +1) % self.capicity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)