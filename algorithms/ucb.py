import numpy as np
from .algorithm import Algorithm
from .common import conf_r, rd_argmax

class UCB(Algorithm):
    '''
        Implementation of the UCB algorithm which is agnostic of the dataset
    '''
    def __init__(self, true_means, dataset, num_arms):
        self.dataset = dataset # save the dataset (although is ignored)
        self.num_arms = num_arms
        self.true_means = true_means

        self.means = np.zeros(self.num_arms) # initializes estimates of the mean
        self.selection = np.zeros(self.num_arms) # and number of samples


        self.regret = 0 # keeps track of its regret
        self.regret_iterations = 0 # and current index in the regret


    def reset(self, dataset):
        self.regret = 0 # resets the estimates back to zero
        self.regret_iterations = 0

        self.means = np.zeros(self.num_arms)
        self.selection = np.zeros(self.num_arms)

    def update_obs(self, action, reward_obs):
        self.means[action] = (self.means[action]*self.selection[action] + reward_obs) / (self.selection[action]+1) # updates the mean estimate
        self.selection[action] += 1 # and the number of samples

    def pick_action(self, t):
        ucb = np.asarray([self.means[i] + conf_r(t, self.selection[i]) if self.selection[i] > 0 else np.inf for i in range(self.num_arms)])   
            # calculates the UCB
        return rd_argmax(ucb) # returns the argmax