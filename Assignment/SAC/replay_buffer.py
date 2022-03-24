import numpy as np
from collections import deque


class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = int(buffer_size)
        self.buffer = deque([], maxlen=self.buffer_size) #TODO: Use Np arrays


    def store_transition(self, state, action, reward, terminal, next_state):
        self.buffer.append([state, action, reward, terminal, next_state])


    def get_transitions(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sampled_transitions = [self.buffer[index] for index in indices]
        states, actions, rewards, terminals, next_states = map(list, zip(*sampled_transitions))

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        next_states = np.array(next_states)
        
        return states, actions, rewards, terminals, next_states


    def is_sampling_possible(self, batch_size):
        return (len(self.buffer) >= batch_size)