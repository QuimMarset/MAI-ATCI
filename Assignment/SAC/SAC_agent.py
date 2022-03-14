from SAC.replay_buffer import ReplayBuffer
from SAC.SAC_model import SACModel


class SACAgent:

    def __init__(self, learning_rate, gradient_clipping, state_shape, action_space_info, buffer_size, gamma, tau, alpha):
        self.buffer = ReplayBuffer(buffer_size)
        self.model = SACModel(learning_rate, gradient_clipping, state_shape, action_space_info, gamma, tau, alpha)

    @classmethod
    def test(cls, path, action_size, min_action, max_action):
        agent = cls.__new__(cls)
        agent.model = SACModel.test(path, action_size, min_action, max_action)
        return agent


    def load_models_weights(self, path):
        self.model.load_models_weights(path)


    def step(self, states):
        action = self.model.forward(states)
        return action


    def store_transition(self, state, action, reward, terminal, next_state):
        self.buffer.store_transition(state, action, reward, terminal, next_state)


    def train(self, batch_size):
        losses = {}

        if self.buffer.is_sampling_possible(batch_size):
            states, actions, rewards, terminals, next_states = self.buffer.get_transitions(batch_size)

            actor_loss = self.model.update_actor(states)
            critic_1_loss, critic_2_loss = self.model.update_critics(states, actions, rewards, terminals, next_states)
            self.model.update_target_critics()

            losses = {'actor_loss' : actor_loss, 'critic_1_loss': critic_1_loss, 'critic_2_loss' : critic_2_loss}

        return losses


    def save_models_weights(self, path):
        self.model.save_weights(path)