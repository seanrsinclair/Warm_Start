class Algorithm(object):

    def __init__(self, dataset, num_arms):
        self.dataset = dataset
        pass

    def reset(self, dataset):
        pass

    def update_config(self, config):
        self.config = config

    def update_obs(self, action, reward):
        pass

    def pick_action(self, t):
        pass