import numpy as np
from .algorithm import Algorithm
from .common import conf_r

class ThompsonSampling(Algorithm):
    '''
        Implementation of the TS algorithm which is agnostic of the dataset
    '''
    def __init__(self, true_means, dataset, num_arms):
        self.dataset = dataset # save the dataset (although is ignored)
        self.num_arms = num_arms
        self.true_means = true_means

        self.alpha = np.ones(self.num_arms) # initializes posterior alpha and beta
        self.beta = np.ones(self.num_arms)


        self.regret = 0 # keeps track of its regret
        self.regret_iterations = 0 # and current index in the regret


    def reset(self, dataset):
        self.regret = 0 # resets the estimates back to zero
        self.regret_iterations = 0

        self.alpha = np.ones(self.num_arms) # initializes posterior alpha and beta
        self.beta = np.ones(self.num_arms)

    def update_obs(self, action, reward_obs):
        if reward_obs == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

    def pick_action(self, t):
        samples_list = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.num_arms)]
        return np.argmax(samples_list) # returns the argmax