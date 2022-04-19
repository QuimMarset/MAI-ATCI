from constants.constants_general import ENVIRONMENT
import gym
import random


class Environment:

    def __init__(self, reward_scale=1, render=False):
        self.env = gym.make(ENVIRONMENT)
        #self.env.seed(random.randint(0, 9999))
        self.render = render
        self.reward_scale = reward_scale


    def start(self):
        return self.env.reset()

    
    def step(self, action):
        if self.render:
            self.env.render()
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward*self.reward_scale, terminal


    def get_state_shape(self):
        return self.env.observation_space.shape

    
    def get_action_space(self):
        return self.env.action_space


    def sample_action(self):
        return self.env.action_space.sample()


    def end(self):
        self.env.close()