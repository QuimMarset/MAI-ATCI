import numpy as np
from environments.environment import Environment
from environments.multi_environment_manager import MultiEnvironmentManager
from utils.results_plotter import TrainResults
from constants import *
from PPO.PPO_agent import PPOAgent


def pretty_print_episode(experiment, episode, iteration, env, episode_reward, last_100_average):
    print(f'Experiment {experiment+1}/{TRAIN_EXPERIMENTS}: Iteration {iteration+1}/{ITERATIONS} Episode {episode+1}/{TRAIN_EPISODES}, ' +
        f'Env {env}, Reward: {episode_reward:.2f}, Last 100 Average: {last_100_average:.2f}')


def train_experiment(env: MultiEnvironmentManager, agent: PPOAgent, results_plotter: TrainResults, best_avg_reward, experiment_index, weights_path):
    states = env.start()
    done = False
    episode_rewards = np.zeros(env.get_num_envs())
    episode_index = 0

    for iteration in range(ITERATIONS):

        for _ in range(ITERATION_STEPS):

            actions = agent.step(states)
            next_states, rewards, dones = env.step(actions)
            episode_rewards[:] += rewards
            agent.store_transitions(states, rewards, dones)
            states = next_states

            for (index, done) in enumerate(dones):
                
                if done and episode_index < TRAIN_EPISODES:
                    results_plotter.add_episode_info(episode_rewards[index])
                    pretty_print_episode(experiment_index, episode_index, iteration, index, episode_rewards[index], 
                        results_plotter.get_last_100_avg_reward())
                    episode_index += 1
                    episode_rewards[index] = 0

        last_100_average = results_plotter.get_last_100_avg_reward()
        if iteration > 0 and iteration%5 == 0 or iteration == ITERATIONS-1: 
            if last_100_average >= best_avg_reward:
                best_avg_reward = last_100_average
                agent.save_model(weights_path)
        
        train_metrics = agent.train(BATCH_SIZE, states, iteration, ITERATIONS)
        results_plotter.add_metrics_info(train_metrics)

        if episode_index == TRAIN_EPISODES:
            break

    return best_avg_reward


def train_agent(results_path, weights_path):
    envs = MultiEnvironmentManager(Environment, NUM_ENVS)
    state_shape = envs.get_state_shape()
    action_space = envs.get_action_space()
    action_space_info = (action_space.shape[0], action_space.low, action_space.high)

    best_avg_reward = -100000
    results_plotter = TrainResults(results_path)

    for i in range(TRAIN_EXPERIMENTS):
        agent = PPOAgent(state_shape, action_space_info, BUFFER_SIZE, NUM_ENVS, GAMMA, GAE_LAMBDA, 
            EPSILON, EPOCHS, LEARNING_RATE, GRADIENT_CLIPPING, MAX_KL_DIVERG)

        best_avg_reward = train_experiment(envs, agent, results_plotter, best_avg_reward, i, weights_path)
        results_plotter.plot_results()
        results_plotter.end_experiment()
    
    envs.end()


def test_agent(results_path, weights_path, render=True):

    env = Environment(render=render)
    state_shape = env.get_state_shape()
    action_space = env.get_action_space()
    action_space_info = (action_space.shape[0], action_space.low, action_space.high)
    agent = PPOAgent.test(weights_path, state_shape, action_space_info)

    episode_rewards = []

    for i in range(TEST_EPISODES):

        done = False
        state = env.start()
        total_reward = 0

        while not done:
            action = agent.test_step(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)
        print(f'Episode {i+1}/{TEST_EPISODES}: Reward: {total_reward:.2f}')

    avg_reward = np.mean(episode_rewards)
    print(f'Avg reward of {TEST_EPISODES} episodes: {avg_reward}')

    env.end()