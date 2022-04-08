import numpy as np


class ReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_size):
        self.buffer_size = int(buffer_size)
        self.states = np.zeros((self.buffer_size, *state_shape))
        self.actions = np.zeros((self.buffer_size, *action_size))
        self.rewards = np.zeros(self.buffer_size)
        self.terminals = np.zeros(self.buffer_size)
        self.next_states = np.zeros((self.buffer_size, *state_shape))
        self.pointer = 0
        self.sample_size = 0


    def store_transition(self, state, action, reward, terminal, next_state):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.terminals[self.pointer] = terminal
        self.next_states[self.pointer] = next_state
        
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.sample_size = min(self.sample_size + 1, self.buffer_size)


    def get_transitions(self, batch_size):
        indices = np.random.choice(self.sample_size, batch_size, replace=False)
        return (self.states[indices], self.actions[indices], self.rewards[indices], 
            self.terminals[indices], self.next_states[indices])


    def is_sampling_possible(self, batch_size):
        return self.sample_size >= batch_size