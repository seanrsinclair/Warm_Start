import numpy as np
from .algorithm import Algorithm

class OnlineWrapper(Algorithm):

    def __init__(self, dataset, num_arms, algorithm):
        self.dataset = dataset
        self.num_arms = num_arms
        self.algorithm = algorithm
        self.algorithm.reset(dataset, num_arms)

    def reset(self, dataset):
        self.algorithm.reset(dataset, self.num_arms)

    def update_obs(self, action, reward_obs):
        self.algorithm.update_obs(action, reward_obs)
        
    def pick_action(self, t):
        ucb = np.asarray([self.means[i] + (np.sqrt(2*np.log(1+t) / self.selection[i])) if self.selection[i] > 0 else np.inf for i in range(self.num_arms)])
        return np.argmax(ucb)