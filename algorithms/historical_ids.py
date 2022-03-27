import numpy as np
from .algorithm import Algorithm
from .common import conf_r, rd_argmax
from .ids import IDS


class HistoricalIDS(IDS):
    '''
        Implementation of the IDS algorithm which is agnostic of the dataset
        param VIDS: boolean, if True choose arm which delta**2/v quantity. Default: False
    '''
    def __init__(self, true_means, dataset, num_arms, VIDS=False):



        self.dataset = dataset # save the dataset (although is ignored)
        self.num_arms = num_arms
        self.true_means = true_means
        self.VIDS = VIDS

        self.M = 1000
        self.threshold = 0.99
        self.regret = 0 # keeps track of its regret
        self.regret_iterations = 0 # and current index in the regret

        self.flag = False
        self.alpha = np.ones(self.num_arms) # initializes posterior alpha and beta
        self.beta = np.ones(self.num_arms)
        for i in range(self.num_arms):
            self.alpha[i] = 1 + np.sum(dataset[str(i)])
            self.beta[i] = 1 + len(dataset[str(i)]) - np.sum(dataset[str(i)])

        self.thetas = np.array([np.random.beta(self.alpha[arm], self.beta[arm], self.M) for arm in range(self.num_arms)])
        self.Maap, self.p_a = np.zeros((self.num_arms, self.num_arms)), np.zeros(self.num_arms)

    def reset(self, dataset):
        self.regret = 0 # resets the estimates back to zero
        self.regret_iterations = 0

        self.flag = False
        self.alpha = np.ones(self.num_arms) # initializes posterior alpha and beta
        self.beta = np.ones(self.num_arms)
        for i in range(self.num_arms):
            self.alpha[i] = 1 + np.sum(dataset[str(i)])
            self.beta[i] = 1 + len(dataset[str(i)]) - np.sum(dataset[str(i)])
        self.thetas = np.array([np.random.beta(self.alpha[arm], self.beta[arm], self.M) for arm in range(self.num_arms)])
        self.Maap, self.p_a = np.zeros((self.num_arms, self.num_arms)), np.zeros(self.num_arms)
