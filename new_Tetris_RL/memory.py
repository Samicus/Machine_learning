from collections import deque

import numpy as np
import torch


class ExperienceReplay:
    "Experience Replay memory buffer used by DDQN algorithm"

    def __init__(self, device, num_states, buffer_size=1e6):
        self.device = device
        self.buffer = deque(maxlen=int(buffer_size))
        self.num_states = num_states

    @property
    def buffer_length(self):
        return len(self.buffer)

    def add(self, transition):
        """
        Adds a transition <s, a, r, s', term_state > to the replay buffer
        """
        self.buffer.append(transition)

    def sample_minibatch(self, batch_size=128):
        ids = np.random.choice(a=self.buffer_length, size=batch_size)
        state_batch = np.zeros([batch_size, self.num_states], dtype=np.float32)
        next_state_batch = np.zeros([batch_size, self.num_states], dtype=np.float32)
        action_batch = np.zeros(
            [
                batch_size,
            ],
            dtype=np.int64,
        )
        reward_batch = np.zeros(
            [
                batch_size,
            ],
            dtype=np.float32,
        )
        nonterminal_batch = np.zeros(
            [
                batch_size,
            ],
            dtype=np.bool,
        )

        for i, index in zip(range(batch_size), ids):
            state_batch[i, :] = self.buffer[index].s
            action_batch[i] = self.buffer[index].a
            reward_batch[i] = self.buffer[index].r
            nonterminal_batch[i] = self.buffer[index].term_state
            next_state_batch[i, :] = self.buffer[index].next_s

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            nonterminal_batch,
        )

    def sample_minibatch_tensor(self, batch_size=128):
        ids = np.random.choice(a=self.buffer_length, size=batch_size)
        state_batch = np.zeros([batch_size, self.num_states], dtype=np.float32)
        next_state_batch = np.zeros([batch_size, self.num_states], dtype=np.float32)
        action_batch = np.zeros(
            [
                batch_size,
            ],
            dtype=np.int64,
        )
        reward_batch = np.zeros(
            [
                batch_size,
            ],
            dtype=np.float32,
        )
        nonterminal_batch = np.zeros(
            [
                batch_size,
            ],
            dtype=np.bool,
        )

        for i, index in zip(range(batch_size), ids):
            state_batch[i, :] = self.buffer[index].s
            action_batch[i] = self.buffer[index].a
            reward_batch[i] = self.buffer[index].r
            next_state_batch[i, :] = self.buffer[index].next_s
            nonterminal_batch[i] = self.buffer[index].term_state
        return (
            torch.tensor(state_batch, dtype=torch.float, device=self.device),
            torch.tensor(action_batch, dtype=torch.long, device=self.device),
            torch.tensor(reward_batch, dtype=torch.float, device=self.device),
            torch.tensor(next_state_batch, dtype=torch.float, device=self.device),
            torch.tensor(nonterminal_batch, dtype=torch.bool, device=self.device),
        )