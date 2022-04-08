import numpy as np
from environments.environment import Environment
from utils.results_plotter_SAC import TrainResults
from constants.constants import *
from SAC.SAC_agent import SACAgent


def train_experiment(env: Environment, agent: SACAgent, results_plotter: TrainResults, experiment_steps, episode_index, best_avg_reward):
    state = env.start()
    done = False
    episode_length = 0
    episode_reward = 0

    for step in range(experiment_steps):

        if step < SAMPLING_STEPS:
            action = env.sample_action()
        else:
            action = agent.step(state)

        next_state, reward, done = env.step(action)
        episode_reward += reward
        episode_length += 1

        agent.store_transition(state, action, reward, done, next_state)
        
        state = next_state

        if done:
            state = env.start()
            episode_index += 1
            print(f'Episode {episode_index} ended with {episode_length} steps and reward {episode_reward:.2f}')
            results_plotter.add_episode_info(episode_reward)
            episode_reward = 0
            episode_length = 0

        if step >= START_UPDATING and step%UPDATE_EVERY == 0:
            train_metrics = {}
            train_metrics = agent.train(SAC_BATCH_SIZE)
            results_plotter.add_train_info(train_metrics)

        if step > 0 and step%STEPS_PER_EPOCH == 0:
            results_plotter.plot_results()

            last_100_avg_reward = results_plotter.get_last_100_avg_reward()
            if last_100_avg_reward >= best_avg_reward:
                best_avg_reward = last_100_avg_reward
                agent.save_models_weights(WEIGHTS_SAC)

    return episode_index, best_avg_reward


def train_agent():
    env = Environment()
    state_shape = env.get_state_shape()
    action_space = env.get_action_space()
    action_space_info = (action_space.shape, action_space.low, action_space.high)
    
    agent = SACAgent(LEARNING_RATE, GRADIENT_CLIPPING, state_shape, action_space_info, SAC_BUFFER_SIZE, GAMMA, TAU, ALPHA)

    total_steps = SAC_EPOCHS * STEPS_PER_EPOCH
    episode_index = 0
    best_avg_reward = -100000

    results_plotter = TrainResults(TRAIN_EXPERIMENTS)

    for _ in range(TRAIN_EXPERIMENTS):

        episode_index, best_avg_reward = train_experiment(env, agent, results_plotter, total_steps, episode_index, best_avg_reward)
        results_plotter.end_experiment()

    env.end()


def test_agent(render=True, env=None, agent=None, test_episodes=TEST_EPISODES):

    if env is None and agent is None:
        env = Environment(render=render)
        state_size = env.get_state_shape()[0]
        action_space = env.get_action_space()
        agent = SACAgent.test(WEIGHTS_PATH, state_size, action_space.shape[0], action_space.low, action_space.high)

    episode_rewards = []

    for _ in test_episodes:

        done = False
        state = env.start()
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)
    print(f'Avg reward of {TEST_EPISODES}: {avg_reward}')

    env.end()

    return avg_reward


"""if episode > 0 and episode%300:
    avg_test_episode_reward = test_agent(False, env, agent, 10)

    if avg_test_episode_reward > SOLVE_CONDITON/2:
        avg_test_episode_reward = test_agent(False, env, agent, 100)
        if avg_test_episode_reward >= SOLVE_CONDITION:"""