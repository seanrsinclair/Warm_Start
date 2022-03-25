import numpy as np
from .algorithm import Algorithm
from .thompson_sampling import ThompsonSampling
from .common import conf_r


class HistoricalThompsonSampling(ThompsonSampling): # inherits the original TS algorithm
    '''
        Implementation of the TS algorithm which uses the historical dataset to start off
        the alpha beta posterior parameters
    '''
    def __init__(self, true_means, dataset, num_arms): 
        self.dataset = dataset
        self.true_means = true_means
        self.num_arms = num_arms
            # same as original algorithm but updates the alpha and beta parameters from the dataset
        self.alpha = np.zeros(self.num_arms)
        self.beta = np.zeros(self.num_arms)
        
        for i in range(self.num_arms):
            counts = np.unique(dataset[str(i)], return_counts=True)
            self.alpha[i] = counts[1]
            self.beta[i] = counts[0]

        self.regret = 0
        self.regret_iterations = 0

    def reset(self, dataset):
        self.regret = 0
        self.regret_iterations = 0
            # similar difference in the reset function
        for i in range(self.num_arms):
            counts = np.unique(dataset[str(i)], return_counts=True)
            self.alpha[i] = counts[1]
            self.beta[i] = counts[0]