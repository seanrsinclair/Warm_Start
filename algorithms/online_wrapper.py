import numpy as np
from .algorithm import Algorithm

class OnlineWrapper(Algorithm):

    def __init__(self, true_means, dataset, num_arms, algorithm):
        self.dataset = dataset
        self.num_arms = num_arms
        self.algorithm = algorithm
        self.true_means = true_means

        self.algorithm.reset(dataset)

        self.dataset_index = np.zeros(self.num_arms) # keeps track of which datapoints have been used in the historical dataset
        self.regret = 0 # tracker for cumulative regret
        self.regret_iterations = 0 # tracker for number of online calls when sampling an arm


    def reset(self, dataset):
        self.regret = 0 # resets all of the quantities
        self.regret_iterations = 0
        self.dataset = dataset
        self.dataset_index = np.zeros(self.num_arms)
        self.algorithm.reset(dataset)

    def update_obs(self, action, reward_obs):
        self.algorithm.update_obs(action, reward_obs) # appeals to the sub-algorithm to update based on the observation
        
    def one_step(self, t):
        flag = False
        action = self.algorithm.pick_action(self.regret_iterations) # calls its sub algorithm to pick an action

        # Check if we have value in dataset that we can use and feed back to algorithm
        if self.dataset_index[action] < len(self.dataset[str(action)]):
            obs = self.dataset[str(action)][int(self.dataset_index[action])]# gets a value from the dataset
            self.dataset_index[action] += 1 # updates index within the dataset
        else:
        # Otherwise, take online sample:
            obs = np.random.binomial(p=self.true_means[action], n=1)
            self.regret += np.max(self.true_means) - self.true_means[action]
            self.regret_iterations += 1
            self.algorithm.regret_iterations += 1
            flag = True
        self.update_obs(action, obs)
        return flag