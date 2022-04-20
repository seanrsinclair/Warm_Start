import numpy as np
from .algorithm import Algorithm
from .common import conf_r, rd_argmax

class UCB(Algorithm):
    '''
        Implementation of the UCB algorithm which is agnostic of the dataset
    '''
    def __init__(self, true_means, dataset, K):
        self.dataset = dataset # save the dataset (although is ignored)
        self.K = K
        self.true_means = true_means

        self.means = np.zeros(self.K) # initializes estimates of the mean
        self.selection = np.zeros(self.K) # and number of samples


        self.regret = 0 # keeps track of its regret
        self.regret_iterations = 0 # and current index in the regret


    def reset(self, dataset):
        self.regret = 0 # resets the estimates back to zero
        self.regret_iterations = 0

        self.means = np.zeros(self.K)
        self.selection = np.zeros(self.K)

    def update_obs(self, action, reward_obs):
        self.means[action] = (self.means[action]*self.selection[action] + reward_obs) / (self.selection[action]+1) # updates the mean estimate
        self.selection[action] += 1 # and the number of samples

    def pick_action(self, t):
        ucb = np.asarray([self.means[k] + conf_r(t, self.selection[k]) if self.selection[k] > 0 else np.inf for k in range(self.K)])
            # calculates the UCB
        return rd_argmax(ucb) # returns the argmax
