import tensorflow as tf
from PPO.PPO_actor_critic import Actor, Critic
from utils.utils import is_folder_empty, exists_folder


class PPOModel:

    def __init__(self, learning_rate, gradient_clipping, state_shape, num_actions, epsilon, max_kl_diverg):
        self.max_kl_diverg = max_kl_diverg
        self.epsilon = epsilon
        self.original_epsilon = epsilon
        self.actor = Actor(learning_rate, gradient_clipping, state_shape, num_actions)
        self.critic = Critic(learning_rate, gradient_clipping, state_shape)


    @classmethod
    def test(cls, model_path, state_shape, action_space):
        (action_size, min_action, max_action) = action_space
        model = cls.__new__(cls)
        model.actor = Actor.test(state_shape, action_size, min_action, max_action)
        model.critic = Critic.test(state_shape)
        model.load_models(model_path)
        return model


    def forward(self, states):
        actions, actions_log_prob = self.actor(states)
        state_values = self.critic(states)
        return actions.numpy(), actions_log_prob.numpy(), state_values.numpy()


    def test_forward(self, state):
        state = tf.expand_dims(state, axis=0)
        action = self.actor.call_test(state)
        return tf.squeeze(action, axis=0)


    def apply_annealing(self, annealing_fraction):
        self.epsilon = self.original_epsilon * annealing_fraction
        self.actor.apply_annealing(annealing_fraction)
        self.critic.apply_annealing(annealing_fraction)

    
    def compute_actor_loss(self, tape, states, actions, actions_old_log_prob, advantages):
        with tape:
            actions_log_prob = self.actor.call_update(states, actions)
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            clip_surrogate = tf.clip_by_value(ratios, 1-self.epsilon, 1+self.epsilon)*advantages
            kl_diverg = tf.reduce_mean(actions_old_log_prob - actions_log_prob)

            actor_loss = -tf.reduce_mean(tf.minimum(ratios*advantages, clip_surrogate))

            actor_loss = tf.where(tf.abs(kl_diverg) >= self.max_kl_diverg, 
                tf.stop_gradient(actor_loss), actor_loss)

        return actor_loss, kl_diverg


    def compute_critic_loss(self, tape, states, returns, old_values):
        with tape:
            state_values = self.critic(states)
            state_values_clipped = old_values + tf.clip_by_value(state_values - old_values, -self.epsilon, self.epsilon)
            critic_loss_unclipped = (returns - state_values)**2
            critic_loss_clipped = (returns - state_values_clipped)**2
            
            critic_loss = 0.5 * tf.reduce_mean(tf.maximum(critic_loss_clipped, critic_loss_unclipped))

        return critic_loss


    def update_actor(self, states, actions, actions_log_prob, advantages):
        tape = tf.GradientTape()
        loss, kl_diverg = self.compute_actor_loss(tape, states, actions, actions_log_prob, advantages)
        trainable_variables = self.actor.trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)

        has_nans = tf.reduce_any([tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients])
        if has_nans:
            print("Actor NANs!")
            print(loss)

        self.actor.update(gradients)
        return loss.numpy(), kl_diverg.numpy()
 

    def update_critic(self, states, returns, state_values):
        tape = tf.GradientTape()

        loss = self.compute_critic_loss(tape, states, returns, state_values)
        trainable_variables = self.critic.trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)

        has_nans = tf.reduce_any([tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients])
        if has_nans:
            print("Critic NANs!")
            print(loss)

        self.critic.update(gradients)
        return loss.numpy()


    def update_models(self, states, actions, actions_log_prob, advantages, returns, state_values):
        actor_loss, kl_diverg = self.update_actor(states, actions, actions_log_prob, advantages)
        critic_loss = self.update_critic(states, returns, state_values)
        return actor_loss, critic_loss, kl_diverg
        

    def load_models(self, path):
        if exists_folder(path) and not is_folder_empty(path):
            self.load_weights(path)
        else:
            print('Trying to load weights from an empty folder')


    def load_weights(self, path):
        self.actor.load_weights(f'{path}/actor_weights')
        self.critic.load_weights(f'{path}/critic_weights')


    def save_models(self, path):
        self.actor.save_weights(f'{path}/actor_weights')
        self.critic.save_weights(f'{path}/critic_weights')