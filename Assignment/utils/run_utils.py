from constants import *
from PPO.PPO_agent import DiscretePPOAgent, ContinuousPPOAgent
from environments.environment import Environment
from environments.vizdoom_environment import VizDoomEnvironment
from environments.multi_environment_manager import MultiEnvironmentManager


def create_train_environments(environment_name):
    if environment_name is VIZDOOM:
        env_function = VizDoomEnvironment
        env_params = {'env_name': environment_name, 'num_stacked': FRAMES_STACKED, 'num_skipped': FRAMES_SKIPPED, 
            'frame_resize': FRAME_RESIZE, 'reward_scale': REWARD_SCALE}

    else:
        env_function = Environment
        env_params = {'env_name': environment_name, 'reward_scale': REWARD_SCALE}

    return MultiEnvironmentManager(NUM_ENVS, env_function, env_params)


def create_test_environment(environment_name, render):
    if environment_name is VIZDOOM:
        env = VizDoomEnvironment(environment_name, frame_resize=FRAME_RESIZE, render=render)
    else:
        env = Environment(environment_name, render=render)
    return env


def create_train_agent(environment_name, state_shape, action_space):
    if environment_name is VIZDOOM:
        num_actions = action_space
        agent = DiscretePPOAgent(state_shape, num_actions, BUFFER_SIZE, NUM_ENVS, GAMMA, GAE_LAMBDA, EPSILON, 
            EPOCHS, LEARNING_RATE, GRADIENT_CLIPPING, MAX_KL_DIVERG)

    else:
        action_space_info = (action_space.shape[0], action_space.low, action_space.high)
        agent = ContinuousPPOAgent(state_shape, action_space_info, BUFFER_SIZE, NUM_ENVS, GAMMA, GAE_LAMBDA, EPSILON, EPOCHS, 
            LEARNING_RATE, GRADIENT_CLIPPING, MAX_KL_DIVERG)

    return agent


def create_test_agent(environment_name, state_shape, action_space, weights_path):
    if environment_name is VIZDOOM:
        num_actions = action_space
        agent = DiscretePPOAgent.test(weights_path, state_shape, num_actions)

    else:
        action_space_info = (action_space.shape[0], action_space.low, action_space.high)
        agent = ContinuousPPOAgent.test(weights_path, state_shape, action_space_info)
    
    return agent