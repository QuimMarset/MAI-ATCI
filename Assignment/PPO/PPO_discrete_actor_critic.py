import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow_probability.python.distributions import Categorical


class DiscreteActor:

    def __init__(self, learning_rate, gradient_clipping, state_shape, num_actions):
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping
        self.create_model(state_shape, num_actions)


    @classmethod
    def test(cls, state_shape, num_actions):
        actor = cls.__new__(cls)
        actor.create_model(state_shape, num_actions)
        return actor


    def create_model(self, state_shape, num_actions):
        state_input = keras.Input(state_shape)
        
        conv_1 = keras.layers.Conv2D(32, 3, activation='relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv_1)
        conv_2 = keras.layers.Conv2D(64, 3, activation='relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv_2)
        conv_3 = keras.layers.Conv2D(128, 3, activation='relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv_3)
        flatten = keras.layers.Flatten()(avg_pool3)
        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)

        prob_dists = keras.layers.Dense(num_actions, activation='softmax')(dense_1)
        self.model = keras.Model(state_input, prob_dists)

    
    def get_log_prob_actions(self, log_probs, actions):
        batch_size = log_probs.shape[0]
        indices_dim_batch = tf.constant(range(batch_size), shape=(batch_size, 1))
        actions = tf.convert_to_tensor(actions, dtype = tf.int32)
        indices = tf.concat([indices_dim_batch, tf.expand_dims(actions, axis=-1)], axis=-1)
        log_probs_actions = tf.gather_nd(log_probs, indices)
        return log_probs_actions

    
    def compute_log_prob(self, prob_dists, actions):
        offsets = tf.cast(prob_dists == 0, dtype=tf.float32)*1e-6
        prob_dists = prob_dists + offsets
        log_probs = tf.math.log(prob_dists)
        return self.get_log_prob_actions(log_probs, actions)


    def compute_actions(self, states):
        prob_dists = self.model(states)
        categ_dists = Categorical(probs=prob_dists)
        actions = categ_dists.sample()
        log_probs = self.compute_log_prob(prob_dists, actions)
        return actions, log_probs


    def __call__(self, states):
        prob_dists = self.model(states)
        return tf.argmax(prob_dists, axis=0)
        

    def call_test(self, states):
        actions, _ = self.compute_actions(states)
        return actions


    def call_update(self, states, actions):
        prob_dists = self.model(states)
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


class DiscreteCritic:

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

        conv_1 = keras.layers.Conv2D(32, 3, activation='relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv_1)
        conv_2 = keras.layers.Conv2D(64, 3, activation='relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv_2)
        conv_3 = keras.layers.Conv2D(128, 3, activation='relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv_3)
        flatten = keras.layers.Flatten()(avg_pool3)
        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)

        v_value = keras.layers.Dense(1)(dense_1)
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