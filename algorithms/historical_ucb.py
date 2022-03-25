import numpy as np
from .algorithm import Algorithm

class HistoricalUCB(Algorithm):

    def __init__(self, dataset, num_arms):
        self.dataset = dataset
        self.num_arms = num_arms
        self.means = np.asarray([max(0.0,np.mean(dataset[str(i)])) for i in range(self.num_arms)])
        self.selection = np.asarray([len(dataset[str(i)]) for i in range(self.num_arms)])

    def reset(self, dataset):
        self.means = np.asarray([max(0.0,np.mean(dataset[str(i)])) for i in range(self.num_arms)])
        self.selection = np.asarray([len(dataset[str(i)]) for i in range(self.num_arms)])

    def update_obs(self, action, reward_obs):
        self.means[action] = (self.means[action]*self.selection[action] + reward_obs) / (self.selection[action]+1)
        self.selection[action] += 1

    def pick_action(self, t):
        ucb = np.asarray([self.means[i] + (np.sqrt(2*np.log(1+t) / self.selection[i])) if self.selection[i] > 0 else np.inf for i in range(self.num_arms)])
        return np.argmax(ucb)