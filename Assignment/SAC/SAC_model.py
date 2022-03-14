import tensorflow as tf
from tensorflow import keras
import numpy as np
from SAC.models import Actor, Critic
from utils.utils import is_folder_empty, exists_folder


class SACModel:
    
    def __init__(self, learning_rate, gradient_clipping, state_shape, action_space_info, gamma, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        (action_shape, min_action, max_action) = action_space_info

        self.actor = Actor(learning_rate, gradient_clipping, state_shape, action_shape[0], min_action, max_action)
        self.critic_1 = Critic(state_shape, action_shape, learning_rate, gradient_clipping)
        self.critic_2 = Critic(state_shape, action_shape, learning_rate, gradient_clipping)
        self.target_1 = Critic.clone(self.critic_1)
        self.target_2 = Critic.clone(self.critic_2)

    
    @classmethod
    def test(cls, path, state_size, action_shape, min_action, max_action):
        model = cls.__new__(cls)
        model.actor = Actor.test(state_size, action_shape[0], min_action, max_action)
        model.critic_1 = Critic.test(state_size, action_shape)
        model.critic_2 = Critic.test(state_size, action_shape)
        model.target_1 = keras.models.clone_model(model.critic_1)
        model.target_2 = keras.models.clone_model(model.critic_2)
        model.load_models_weights(path)
        return model

    
    def load_models_weights(self, path):
        if exists_folder(path) and not is_folder_empty(path):
            self.load_weights(path)
        else:
            print('Trying to load weights from an empty folder')


    def forward(self, state):
        state = tf.expand_dims(state, axis=0)
        action = self.actor(state)
        action = tf.squeeze(action, axis=0)
        return action


    def _compute_actor_loss(self, tape, states):
        with tape:
            actions, actions_log_prob = self.actor.call_update(states)
            
            q1_values = tf.squeeze(self.critic_1(states, actions), axis = -1)
            q2_values = tf.squeeze(self.critic_2(states, actions), axis = -1)
            q_values = tf.minimum(q1_values, q2_values)
            
            loss = tf.reduce_mean(self.alpha*actions_log_prob - q_values)
        return loss


    def _compute_critic_loss(self, critic, tape, states, actions, target):
        with tape:
            q_values = tf.squeeze(critic(states, actions), axis = -1)
            loss = keras.losses.MSE(q_values, target)
        return loss


    def _compute_target_critic_update(self, tape, rewards, terminals, next_states):
        with tape:
            next_actions, next_actions_log_prob = self.actor.call_update(next_states)
            
            q1_target_values = tf.squeeze(self.target_1(next_states, next_actions), axis = -1)
            q2_target_values = tf.squeeze(self.target_2(next_states, next_actions), axis = -1)
            q_target_values = tf.minimum(q1_target_values, q2_target_values)
            
            target = rewards + self.gamma*(1 - terminals)*(q_target_values - self.alpha*next_actions_log_prob)
        return target


    def _update_critic(self, critic, tape, states, actions, target):
        loss = self._compute_critic_loss(critic, tape, states, actions, target)
        trainable_variables = critic.trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)

        has_nans = tf.reduce_any([tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients])
        if has_nans:
            print("NANs!")
            print(loss)

        critic.update(gradients)
        return loss.numpy()

    
    def _update_target_critic(self, critic, target_critic):
        target_weights = target_critic.get_weights()
        critic_weights = critic.get_weights()
        
        for critic_layer_weights, target_layer_weights in zip(critic_weights, target_weights):
            target_layer_weights[:] = target_layer_weights*(1 - self.tau) + critic_layer_weights*self.tau


    def update_actor(self, states):
        tape = tf.GradientTape()
        loss = self._compute_actor_loss(tape, states)
        trainable_variables = self.actor.trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)

        has_nans = tf.reduce_any([tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients])
        if has_nans:
            print("NANs!")
            print(loss)

        self.actor.update(gradients)
        return loss.numpy()


    def update_critics(self, states, actions, rewards, terminals, next_states):
        tape = tf.GradientTape(persistent = True)
        target = self._compute_target_critic_update(tape, rewards, terminals, next_states)
        loss_1 = self._update_critic(self.critic_1, tape, states, actions, target)
        loss_2 = self._update_critic(self.critic_2, tape, states, actions, target)
        return loss_1, loss_2


    def update_target_critics(self):
        self._update_target_critic(self.critic_1, self.target_1)
        self._update_target_critic(self.critic_2, self.target_2)


    def load_weights(self, path):
        self.actor.load_weights(f'{path}/actor_weights')
        self.critic_1.load_weights(f'{path}/critic_1_weights')
        self.critic_2.load_weights(f'{path}/critic_2_weights')
        self.target_1.load_weights(f'{path}/target_1_weights')
        self.target_2.load_weights(f'{path}/target_2_weights')


    def save_weights(self, path):
        self.actor.save_weights(f'{path}/actor_weights')
        self.critic_1.save_weights(f'{path}/critic_1_weights')
        self.critic_2.save_weights(f'{path}/critic_2_weights')
        self.target_1.save_weights(f'{path}/target_1_weights')
        self.target_2.save_weights(f'{path}/target_2_weights')