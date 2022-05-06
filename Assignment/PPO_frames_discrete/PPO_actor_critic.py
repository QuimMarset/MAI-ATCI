import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
from tensorflow_probability.python.distributions import Categorical


class Actor:

    def __init__(self, learning_rate, gradient_clipping, state_shape, num_actions):
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping
        self.create_model(state_shape, num_actions)


    @classmethod
    def test(cls, state_shape, action_size, min_action, max_action):
        actor = cls.__new__(cls)
        actor.min_action = min_action
        actor.max_action = max_action
        actor.create_model(state_shape, action_size)
        return actor


    def create_model(self, state_shape, num_actions):
        state_input = keras.Input(state_shape)
        
        dense_1_out = keras.layers.Dense(256)(state_input)
        lrelu_1_out = keras.layers.LeakyReLU(0.1)(dense_1_out)
        bn_1_out = keras.layers.BatchNormalization()(lrelu_1_out)
        dense_2_out = keras.layers.Dense(128)(bn_1_out)
        lrelu_2_out = keras.layers.LeakyReLU(0.1)(dense_2_out)
        bn_2_out = keras.layers.BatchNormalization()(lrelu_2_out)

        logits = keras.layers.Dense(num_actions)(bn_2_out)

        self.model = keras.Model(state_input, logits)

    
    def compute_log_prob(self, prob_dists, actions):
        probs = prob_dists.prob(actions)
        offsets = tf.cast(probs == 0, dtype=tf.float32)*1e-6
        probs = probs + offsets
        return tf.math.log(probs)


    def compute_actions(self, states):
        logits = self.model(states)
        prob_dists = Categorical(logits=logits)
        actions = prob_dists.sample()
        log_probs = self.compute_log_prob(prob_dists, actions)
        return actions, log_probs


    def __call__(self, states):
        actions, log_prob_actions = self.compute_actions(states)
        return actions, log_prob_actions


    def call_update(self, states, actions):
        logits = self.model(states)
        prob_dists = Categorical(logits=logits)
        log_probs = self.compute_log_prob(prob_dists, actions)
        return log_probs


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

        dense_1_out = keras.layers.Dense(256)(state_input)
        lrelu_1_out = keras.layers.LeakyReLU(0.1)(dense_1_out)
        bn_1_out = keras.layers.BatchNormalization()(lrelu_1_out)
        dense_2_out = keras.layers.Dense(128)(bn_1_out)
        lrelu_2_out = keras.layers.LeakyReLU(0.1)(dense_2_out)
        bn_2_out = keras.layers.BatchNormalization()(lrelu_2_out)

        v_value = keras.layers.Dense(units = 1, activation='linear')(bn_2_out)

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