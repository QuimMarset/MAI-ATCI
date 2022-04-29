import numpy as np
from PPO.PPO_model import PPOModel
from PPO.PPO_buffer import PPOBuffer

class PPOAgent:

    def __init__(self, state_shape, action_space, buffer_size, num_envs, gamma, gae_lambda, epsilon, epochs, 
        learning_rate, gradient_clipping, max_kl_diverg):

        self.epochs = epochs
        self.buffer = PPOBuffer(buffer_size, num_envs, state_shape, action_space[0], gamma, gae_lambda)
        self.model = PPOModel(learning_rate, gradient_clipping, state_shape, action_space, epsilon, max_kl_diverg)
        self.last_values = None
        self.last_actions = None
        self.last_actions_log_prob = None

    
    @classmethod
    def test(cls, path, state_shape, action_space):
        agent = cls.__new__(cls)
        agent.model = PPOModel.test(path, state_shape, action_space)
        return agent


    def step(self, states):
        self.last_actions, self.last_actions_log_prob, self.last_values = self.model.forward(states)
        return self.last_actions


    def test_step(self, state):
        return self.model.test_forward(state)


    def store_transitions(self, states, rewards, terminals):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, self.last_values,
            self.last_actions_log_prob)


    def train(self, batch_size, last_next_states, current_iteration, total_iterations):
        _, _, bootstrapped_values = self.model.forward(last_next_states)

        states, actions, returns, advantages, actions_log_prob, values = self.buffer.get_transitions(bootstrapped_values)

        num_transitions = states.shape[0]
        indices = np.arange(num_transitions)
        num_batches = int(np.ceil(num_transitions/batch_size))

        annealing_fraction = 1 - current_iteration/total_iterations
        self.model.apply_annealing(annealing_fraction) 

        for _ in range(self.epochs):

            np.random.shuffle(indices)

            for i in range(num_batches):

                start_index = i*batch_size
                end_index = start_index+batch_size if start_index+batch_size < num_transitions else num_transitions
                indices_batch = indices[start_index:end_index]

                batch_states = states[indices_batch]
                batch_actions = actions[indices_batch]
                batch_log_probs = actions_log_prob[indices_batch]
                batch_advantages = advantages[indices_batch]
                batch_returns = returns[indices_batch]
                batch_values = values[indices_batch]

                train_metrics = self.model.update_models(batch_states, batch_actions, batch_log_probs, batch_advantages, 
                    batch_returns, batch_values)
        
        (actor_loss, critic_loss, kl_divergence) = train_metrics
        return {'actor_loss': actor_loss, 'critic_loss': critic_loss, 'kl_divergence': kl_divergence}


    def load_model(self, path):
        self.model.load_models(path)


    def save_model(self, path):
        self.model.save_models(path)