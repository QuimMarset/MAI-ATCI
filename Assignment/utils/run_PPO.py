import numpy as np
from environments.environment import Environment
from environments.multi_environment_manager import MultiEnvironmentManager
from utils.results_plotter_PPO import TrainResults
from constants.constants import *
from PPO.PPO_agent import PPOAgent


def train_experiment(env: MultiEnvironmentManager, agent: PPOAgent, results_plotter: TrainResults, iterations, iteration_steps, episode_index, best_avg_reward):
    states = env.start()
    done = False
    episode_rewards = np.zeros(env.get_num_envs())

    for iteration in range(iterations):

        for _ in range(iteration_steps):

            actions = agent.step(states)
            next_states, rewards, dones = env.step(actions)

            episode_rewards[:] += rewards

            agent.store_transitions(states, rewards, dones)

            states = next_states

            for (index, done) in enumerate(dones):
                
                if done:
                    
                    episode_reward = episode_rewards[index]

                    results_plotter.add_episode_info(episode_reward)
                    results_plotter.plot_results()

                    last_100_average = results_plotter.get_last_100_avg_reward()
                    
                    print(f'Iteration {iteration}/{iterations}: Episode {episode_index}, Env {index}, Reward: {episode_reward:.2f},' + 
                        f' Last 100 Average: {last_100_average:.2f}')

                    episode_index += 1

                    if iteration > 0 and iteration%100 == 0:
                        if last_100_average >= best_avg_reward:
                            best_avg_reward = last_100_average
                            agent.save_models(WEIGHTS_PPO)
                    
                    episode_rewards[index] = 0
        
        train_metrics = agent.train(PPO_BATCH_SIZE, states, iteration, iterations)
        results_plotter.add_metrics_info(train_metrics)

    return best_avg_reward, episode_index


def train_agent():
    envs = MultiEnvironmentManager(Environment, NUM_ENVS)
    state_shape = envs.get_state_shape()
    action_space = envs.get_action_space()
    action_space_info = (action_space.shape[0], action_space.low, action_space.high)
    
    agent = PPOAgent(state_shape, action_space_info, PPO_BUFFER_SIZE, NUM_ENVS, GAMMA, GAE_LAMBDA, EPSILON, PPO_EPOCHS, 
        LEARNING_RATE, GRADIENT_CLIPPING, MAX_KL_DIVERG)

    best_avg_reward = -100000
    episode_index = 0

    results_plotter = TrainResults(TRAIN_EXPERIMENTS)

    for _ in range(TRAIN_EXPERIMENTS):

        best_avg_reward, episode_index = train_experiment(envs, agent, results_plotter, ITERATIONS, ITERATION_STEPS, 
            episode_index, best_avg_reward)
        results_plotter.end_experiment()

    envs.end()


def test_agent(render=True, env=None, agent=None, test_episodes=TEST_EPISODES):

    if env is None and agent is None:
        env = Environment(render=render)
        state_size = env.get_state_shape()[0]
        action_space = env.get_action_space()
        action_space_info = (action_space.shape[0], action_space.low, action_space.high)
        agent = PPOAgent.test(WEIGHTS_PATH, state_size, action_space_info)

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