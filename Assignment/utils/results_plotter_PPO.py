import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from constants.constants import ALGORITHM, ENVIRONMENT, RESULTS_PATH


class TrainResults:

    def __init__(self, num_experiments):

        self.num_episodes = 0
        self.current_experiment = 0

        self.last_episodes_reward = deque(maxlen=100)
        self.experiments_avg_reward = [[] for _ in range(num_experiments)]

        self.experiments_actor_loss = [[] for _ in range(num_experiments)]
        self.experiments_critic_loss = [[] for _ in range(num_experiments)]


    def add_episode_info(self, episode_reward):
        self.last_episodes_reward.append(episode_reward)
        self.experiments_avg_reward[self.current_experiment].append(np.mean(self.last_episodes_reward))
        self.num_episodes += 1


    def add_metrics_info(self, train_metrics):
        if train_metrics:
            self.experiments_actor_loss[self.current_experiment].append(train_metrics['actor_loss'])
            self.experiments_critic_loss[self.current_experiment].append(train_metrics['critic_loss'])


    def _plot_rewards_results(self):
        means = np.mean(self.experiments_avg_reward, axis=0)
        stds = np.std(self.experiments_avg_reward, axis=0)

        plt.figure(figsize=(7, 5))
        plt.plot(means, label='mean')
        plt.fill_between(range(means.shape[0]), means-stds, means+stds, alpha=0.3, label='mean+-std')
        plt.title(f'Last 100 episodes average reward on {ENVIRONMENT} using {ALGORITHM}')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULTS_PATH}episode_rewards.png')
        plt.close()


    def _plot_losses_results(self, data, model_title_name, model_fig_name):
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        plt.figure(figsize=(7, 5))
        plt.plot(means, label='mean')
        plt.fill_between(range(means.shape[0]), means-stds, means+stds, alpha=0.3, label='mean+-std')
        plt.title(f'{model_title_name} loss evolution on {ENVIRONMENT} using {ALGORITHM}')
        plt.xlabel('Episode')
        plt.ylabel(f'{model_title_name} Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{RESULTS_PATH}{model_fig_name}_loss.png')
        plt.close()


    def plot_results(self):
        self._plot_rewards_results()
        self._plot_losses_results(self.experiments_actor_loss, 'Actor', 'actor')
        self._plot_losses_results(self.experiments_critic_loss, 'Critic', 'critic')


    def get_last_100_avg_reward(self):
        return np.mean(self.last_episodes_reward)


    def end_experiment(self):
        self.current_experiment += 1
        self.last_episodes_reward.clear()