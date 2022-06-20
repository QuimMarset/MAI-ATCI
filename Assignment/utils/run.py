from time import sleep
import numpy as np
from environments.multi_environment_manager import MultiEnvironmentManager
from utils.results_plotter import TrainResults, plot_test_results
from constants import *
from PPO.PPO_agent import PPOAgent
from utils.run_utils import *


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
            episode_rewards[:] += rewards/REWARD_SCALE
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


def train_agent(results_path, weights_path, environment_name):
    envs = create_train_environments(environment_name)
    state_shape = envs.get_state_shape()
    action_space = envs.get_action_space()

    best_avg_reward = -100000
    results_plotter = TrainResults(results_path, environment_name)

    for i in range(TRAIN_EXPERIMENTS):
        agent = create_train_agent(environment_name, state_shape, action_space)

        best_avg_reward = train_experiment(envs, agent, results_plotter, best_avg_reward, i, weights_path)
        results_plotter.plot_results()
        results_plotter.end_experiment()
    
    envs.end()
    results_plotter.save_results_to_npy()
    results_plotter.save_parameter_configuration()


def test_agent(results_path, weights_path, environment_name, render=True):

    env = create_test_environment(environment_name, render)
    state_shape = env.get_state_shape()
    action_space = env.get_action_space()
    agent = create_test_agent(environment_name, state_shape, action_space, weights_path)

    episode_rewards = []

    sleep_time = 0.05 if environment_name is VIZDOOM else 0

    for i in range(TEST_EPISODES):

        done = False
        state = env.start()
        total_reward = 0

        while not done:
            action = agent.test_step(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

            sleep(sleep_time)

        episode_rewards.append(total_reward)
        print(f'Episode {i+1}/{TEST_EPISODES}: Reward: {total_reward:.2f}')

    avg_reward = np.mean(episode_rewards)
    print(f'Avg reward of {TEST_EPISODES} episodes: {avg_reward}')
    plot_test_results(episode_rewards, environment_name, results_path)

    env.end()