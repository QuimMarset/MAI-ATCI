import gym
import random


class Environment:

    def __init__(self, env_name, reward_scale=1, render=False):
        self.env = gym.make(env_name)
        self.render = render
        self.reward_scale = reward_scale


    def start(self):
        return self.env.reset(seed=random.randint(0, 9999))

    
    def step(self, action):
        if self.render:
            self.env.render()
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward*self.reward_scale, terminal


    def get_state_shape(self):
        return self.env.observation_space.shape

    
    def get_action_space(self):
        return self.env.action_space


    def end(self):
        self.env.close()