import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import os
from constants import ALGORITHM, ITERATIONS, ENVIRONMENT, TRAIN_EXPERIMENTS, TRAIN_EPISODES
sns.set(style="whitegrid")


class TrainResults:

    def __init__(self, results_path):
        self.episode = 0
        self.experiment = 0
        self.iteration = 0
        self.results_path = results_path

        self.last_episodes_reward = deque(maxlen=100)
        self.experiments_avg_reward = np.zeros((TRAIN_EXPERIMENTS, TRAIN_EPISODES))

        self.experiments_actor_loss = np.zeros((TRAIN_EXPERIMENTS, ITERATIONS))
        self.experiments_critic_loss = np.zeros((TRAIN_EXPERIMENTS, ITERATIONS))
        self.experiments_kl_divergence = np.zeros((TRAIN_EXPERIMENTS, ITERATIONS))

        self.experiment_iterations = np.zeros(TRAIN_EXPERIMENTS, dtype=int)


    def add_episode_info(self, episode_reward):
        self.last_episodes_reward.append(episode_reward)
        self.experiments_avg_reward[self.experiment, self.episode] = np.mean(self.last_episodes_reward)
        self.episode += 1


    def add_metrics_info(self, train_metrics):
        if train_metrics:
            self.experiments_actor_loss[self.experiment, self.iteration] = train_metrics['actor_loss']
            self.experiments_critic_loss[self.experiment, self.iteration] = train_metrics['critic_loss']
            self.experiments_kl_divergence[self.experiment, self.iteration] = train_metrics['kl_divergence']
        self.iteration += 1


    def _plot_rewards_results(self):
        data = self.experiments_avg_reward[:self.experiment+1]
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        plt.figure(figsize=(7, 5))
        plt.plot(means, label='mean')
        plt.fill_between(range(means.shape[0]), means-stds, means+stds, alpha=0.3, label='mean+-std')
        plt.title(f'Last 100 episodes average reward on {ENVIRONMENT} using {ALGORITHM}')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'episode_rewards.png'))
        plt.close()


    def _plot_train_metric_results(self, data, model_title_name, model_fig_name):
        end_iteration = min(np.max(self.experiment_iterations), ITERATIONS) 
        data = data[:self.experiment+1, :end_iteration]
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        plt.figure(figsize=(7, 5))
        plt.plot(means, label='mean')
        plt.fill_between(range(means.shape[0]), means-stds, means+stds, alpha=0.3, label='mean+-std')
        plt.title(f'{model_title_name} evolution on {ENVIRONMENT} using {ALGORITHM}')
        plt.xlabel('Iteration')
        plt.ylabel(f'{model_title_name} ')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f'{model_fig_name}.png'))
        plt.close()


    def plot_results(self):
        self.experiment_iterations[self.experiment] = self.iteration
        self._plot_rewards_results()
        self._plot_train_metric_results(self.experiments_actor_loss, 'Actor Loss', 'actor_loss')
        self._plot_train_metric_results(self.experiments_critic_loss, 'Critic Loss', 'critic_loss')
        self._plot_train_metric_results(self.experiments_kl_divergence, 'KL Divergence', 'kl_divergence')


    def get_last_100_avg_reward(self):
        return np.mean(self.last_episodes_reward)


    def end_experiment(self):
        self.episode = 0
        self.iteration = 0
        self.experiment += 1
        self.last_episodes_reward.clear()