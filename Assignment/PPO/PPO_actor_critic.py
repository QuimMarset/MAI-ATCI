import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np


class Actor:

    def __init__(self, learning_rate, gradient_clipping, state_shape, action_size, min_action, max_action):
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping
        self.min_action = min_action
        self.max_action = max_action
        self.create_model(state_shape, action_size)


    @classmethod
    def test(cls, state_shape, action_size, min_action, max_action):
        actor = cls.__new__(cls)
        actor.min_action = min_action
        actor.max_action = max_action
        actor.create_model(state_shape, action_size)
        return actor


    def create_model(self, state_shape, action_size):
        state_input = keras.Input(state_shape)
        dense_1_out = keras.layers.Dense(units = 128, activation = 'relu')(state_input)
        dense_2_out = keras.layers.Dense(units = 256, activation = 'relu')(dense_1_out)
        mean = keras.layers.Dense(units = action_size, activation = 'linear')(dense_2_out)
        log_std = keras.layers.Dense(units = action_size, activation = 'tanh')(dense_2_out)

        self.model = keras.Model(state_input, [mean, log_std])

    
    def gaussian_log_likelihood(self, actions, mus, log_stds):
        stds = tf.exp(log_stds)
        pre_sum = -0.5 * (((actions - mus)/(stds + 1e-8))**2 + 2*log_stds + tf.math.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=-1)

    
    def _rescale_actions(self, actions):
        actions = self.min_action + (actions + 1.0)*(self.max_action - self.min_action)/2.0
        return actions


    def compute_actions(self, states):
        mus, log_stds = self.model(states)
        stds = tf.exp(log_stds)

        actions = mus + tf.random.normal(mus.shape)*stds
        log_probs = self.gaussian_log_likelihood(actions, mus, log_stds)

        mus = self._rescale_actions(mus)
        actions = self._rescale_actions(actions)

        return mus, actions, log_probs


    def __call__(self, states):
        _, actions, log_prob_actions = self.compute_actions(states)
        return actions, log_prob_actions


    def call_update(self, states, actions):
        mus, log_stds = self.model(states)
        log_prob_actions = self.gaussian_log_likelihood(actions, mus, log_stds)
        return log_prob_actions


    def call_test(self, states):
        mus, _, _ = self.compute_actions(states)
        return mus


    def update(self, gradients):
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    def apply_annealing(self, annealing_fraction):
        K.set_value(self.optimizer.learning_rate, self.learning_rate*annealing_fraction)

    
    def trainable_variables(self):
        return self.model.trainable_variables


    def load_weights(self, path):
        self.model.load_weights(path)


    def save_weights(self, path):
        self.model.save_weights(path)
    


class Critic:

    def __init__(self, learning_rate, gradient_clipping, state_shape):
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping
        self.create_model(state_shape)


    @classmethod
    def test(cls, state_shape):
        critic = cls.__new__(cls)
        critic.create_model(state_shape)
        return critic


    @classmethod
    def clone(cls, critic):
        target = cls.__new__(cls)
        target.model = keras.models.clone_model(critic._get_model())
        target.model.set_weights(critic.get_weights())
        return target


    def create_model(self, state_shape):
        state_input = keras.Input(state_shape)
        dense_1_out = keras.layers.Dense(units = 128, activation = 'relu')(state_input)
        dense_2_out = keras.layers.Dense(units = 256, activation = 'relu')(dense_1_out)
        v_value = keras.layers.Dense(units = 1, activation = 'linear')(dense_2_out)

        self.model = keras.Model(state_input, v_value)


    def __call__(self, states):
        state_values = self.model(states)
        return tf.squeeze(state_values, axis=-1)


    def update(self, gradients):
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    
    def apply_annealing(self, annealing_fraction):
        K.set_value(self.optimizer.learning_rate, self.learning_rate*annealing_fraction)


    def trainable_variables(self):
        return self.model.trainable_variables


    def load_weights(self, path):
        self.model.load_weights(path)

        
    def save_weights(self, path):
        self.model.save_weights(path)