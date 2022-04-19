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
        
        self.log_stds = -0.5 * np.ones(action_size, np.float32)
        self.stds = np.exp(self.log_stds)


    @classmethod
    def test(cls, state_shape, action_size, min_action, max_action):
        actor = cls.__new__(cls)
        actor.min_action = min_action
        actor.max_action = max_action
        actor.create_model(state_shape, action_size)
        return actor


    def create_model(self, state_shape, action_size):
        state_input = keras.Input(state_shape)

        dense_1 = keras.layers.Dense(512, activation = 'relu')(state_input)
        batch_1 = keras.layers.BatchNormalization()(dense_1)
        dense_2 = keras.layers.Dense(256, activation = 'relu')(batch_1)
        batch_2 = keras.layers.BatchNormalization()(dense_2)
        dense_3 = keras.layers.Dense(64, activation = 'relu')(batch_2)
        batch_3 = keras.layers.BatchNormalization()(dense_3)

        mean = keras.layers.Dense(units = action_size, activation = 'tanh')(batch_3)
        #log_std = keras.layers.Dense(units = action_size, activation = 'tanh')(batch_3)

        self.model = keras.Model(state_input, mean)

    
    def gaussian_log_likelihood(self, actions, mus):
        pre_sum = -0.5 * (((actions - mus)/(self.stds + 1e-8))**2 + 2*self.log_stds + tf.math.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=-1)


    def compute_actions(self, states):
        mus = self.model(states)

        actions = mus + tf.random.normal(mus.shape)*self.stds
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        log_probs = self.gaussian_log_likelihood(actions, mus)

        mus = tf.clip_by_value(mus, self.min_action, self.max_action)

        return mus, actions, log_probs


    def __call__(self, states):
        _, actions, log_prob_actions = self.compute_actions(states)
        return actions, log_prob_actions


    def call_update(self, states, actions):
        mus = self.model(states)
        log_prob_actions = self.gaussian_log_likelihood(actions, mus)
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


    def create_model(self, state_shape):
        state_input = keras.Input(state_shape)

        dense_1 = keras.layers.Dense(512, activation = 'relu')(state_input)
        batch_1 = keras.layers.BatchNormalization()(dense_1)
        dense_2 = keras.layers.Dense(256, activation = 'relu')(batch_1)
        batch_2 = keras.layers.BatchNormalization()(dense_2)
        dense_3 = keras.layers.Dense(64, activation = 'relu')(batch_2)
        batch_3 = keras.layers.BatchNormalization()(dense_3)

        v_value = keras.layers.Dense(units = 1, activation = 'linear')(batch_3)

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


class ActorCritic:

    def __init__(self, learning_rate, gradient_clipping, state_shape, action_size, min_action, max_action):
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping
        self.min_action = min_action
        self.max_action = max_action
        self.create_model(state_shape, action_size)

        self.log_std = -0.5 * np.ones(action_size, np.float32)
        self.std = np.exp(self.log_std)


    @classmethod
    def test(cls, state_shape, action_size, min_action, max_action):
        model = cls.__new__(cls)
        model.min_action = min_action
        model.max_action = max_action
        model.create_model(state_shape, action_size)
        return model


    def create_model(self, state_shape, action_size):
        state_input = keras.Input(state_shape)

        dense_1 = keras.layers.Dense(512, activation = 'relu')(state_input)
        batch_1 = keras.layers.BatchNormalization()(dense_1)
        dense_2 = keras.layers.Dense(256, activation = 'relu')(batch_1)
        batch_2 = keras.layers.BatchNormalization()(dense_2)
        dense_3 = keras.layers.Dense(64, activation = 'relu')(batch_2)
        batch_3 = keras.layers.BatchNormalization()(dense_3)


        mu = keras.layers.Dense(units=action_size, activation='tanh')(batch_3)
        v_value = keras.layers.Dense(units=1, activation='linear')(batch_3)

        self.model = keras.Model(state_input, [mu, v_value])


    def gaussian_log_likelihood(self, actions, mus):
        pre_sum = -0.5 * (((actions - mus)/(self.std + 1e-8))**2 + 2*self.log_std + tf.math.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=-1)


    def compute_actions(self, mus):
        actions = mus + tf.random.normal(mus.shape) * self.std
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        log_probs = self.gaussian_log_likelihood(actions, mus)

        mus = tf.clip_by_value(mus, self.min_action, self.max_action)

        return mus, actions, log_probs


    def __call__(self, states):
        mus, state_values = self.model(states)
        _, actions, log_prob_actions = self.compute_actions(mus)
        state_values = tf.squeeze(state_values, axis=-1)
        return actions, log_prob_actions, state_values


    def call_update(self, states, actions):
        mus, state_values = self.model(states)
        log_prob_actions = self.gaussian_log_likelihood(actions, mus)
        state_values = tf.squeeze(state_values, axis=-1)
        return log_prob_actions, state_values


    def call_test(self, states):
        mus, _ = self.model(states)
        mus, _, _ = self.compute_actions(mus)
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