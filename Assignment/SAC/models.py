import tensorflow as tf
from tensorflow import keras
import numpy as np


class Actor:

    def __init__(self, learning_rate, gradient_clipping, state_shape, action_size, min_action, max_action):
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping
        self.min_action = min_action
        self.max_action = max_action
        self._create_model(state_shape, action_size)


    @classmethod
    def test(cls, state_shape, action_size, min_action, max_action):
        actor = cls.__new__(cls)
        actor.min_action = min_action
        actor.max_action = max_action
        actor._create_model(state_shape, action_size)
        return actor


    def _create_model(self, state_shape, action_size):
        state_input = keras.Input(state_shape)
        dense_1_out = keras.layers.Dense(units = 128, activation = 'relu')(state_input)
        dense_2_out = keras.layers.Dense(units = 256, activation = 'relu')(dense_1_out)
        mean = keras.layers.Dense(units = action_size, activation = 'linear')(dense_2_out)
        log_std = keras.layers.Dense(units = action_size, activation = 'tanh')(dense_2_out)

        self.model = keras.Model(state_input, [mean, log_std])

    
    def _gaussian_log_likelihood(self, actions, mus, log_stds):
        stds = tf.exp(log_stds)
        pre_sum = -0.5 * (((actions - mus)/(stds + 1e-8))**2 + 2*log_stds + tf.math.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=-1)

    
    def _rescale_actions(self, actions):
        actions = self.min_action + (actions + 1.0)*(self.max_action - self.min_action)/2.0
        return actions

    
    def _invertible_squashing(self, mus, unbounded_actions, log_prob_unbounded_actions):
        bounded_mus = tf.tanh(mus)
        actions = tf.tanh(unbounded_actions)
        log_prob_actions = log_prob_unbounded_actions - tf.reduce_sum(tf.math.log(1 - tf.tanh(unbounded_actions)**2 + 1e-6))

        """
        batch_size = unbound_actions.shape[0]
        constant_term = tf.repeat(tf.expand_dims(tf.math.log(1e-6 + (self.max_action - self.min_action)/2.0), axis = 0), 
            batch_size, axis = 0)
        log_jacobian_determinant = tf.reduce_sum(constant_term + tf.math.log(1 - tf.tanh(unbound_actions)**2 + 1e-6), 
            axis = -1)
        actions_log_prob = unbound_actions_log_probs - log_jacobian_determinant
        """

        return bounded_mus, actions, log_prob_actions


    def _compute_actions(self, states, min_log_std=-20, max_log_std=2):
        [mus, log_stds] = self.model(states)
        log_stds = tf.clip_by_value(log_stds, min_log_std, max_log_std)
        stds = tf.exp(log_stds)

        unbounded_actions = mus + tf.random.normal(mus.shape)*stds
        log_prob_unbounded_actions = self._gaussian_log_likelihood(unbounded_actions, mus, log_stds)

        bounded_mus, actions, log_prob_actions = self._invertible_squashing(mus, unbounded_actions, 
            log_prob_unbounded_actions)

        bounded_mus = self._rescale_actions(bounded_mus)
        actions = self._rescale_actions(actions)

        return bounded_mus, actions, log_prob_actions


    def __call__(self, states):
        _, actions, _ = self._compute_actions(states)
        return actions


    def call_update(self, states):
        _, actions, log_prob_actions = self._compute_actions(states)
        return actions, log_prob_actions


    def call_test(self, states):
        mus, _, _ = self._compute_actions(states)
        return mus


    def update(self, gradients):
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    
    def trainable_variables(self):
        return self.model.trainable_variables


    def save_weights(self, path):
        self.model.save_weights(path)
    


class Critic:

    def __init__(self, state_shape, action_shape, learning_rate, gradient_clipping):
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping
        self._create_model(state_shape, action_shape)


    @classmethod
    def test(cls, state_shape, action_shape):
        critic = cls.__new__(cls)
        critic._create_model(state_shape, action_shape)
        return critic


    @classmethod
    def clone(cls, critic):
        target = cls.__new__(cls)
        target.model = keras.models.clone_model(critic._get_model())
        target.model.set_weights(critic.get_weights())
        return target


    def _create_model(self, state_shape, action_shape):
        state_input = keras.Input(state_shape)
        action_input = keras.Input(action_shape)
        concat_input = tf.concat([state_input, action_input], axis=-1)
        dense_1_out = keras.layers.Dense(units = 128, activation = 'relu')(concat_input)
        dense_2_out = keras.layers.Dense(units = 256, activation = 'relu')(dense_1_out)
        q_value = keras.layers.Dense(units = 1, activation = 'linear')(dense_2_out)

        self.model = keras.Model([state_input, action_input], q_value)


    def __call__(self, states, actions):
        q_value = self.model([states, actions])
        return q_value


    def update(self, gradients):
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    def trainable_variables(self):
        return self.model.trainable_variables

    
    def _get_model(self):
        return self.model

    
    def get_weights(self):
        return self.model.get_weights()

    
    def save_weights(self, path):
        self.model.save_weights(path)