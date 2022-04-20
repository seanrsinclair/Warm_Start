import numpy as np

def generate_dataset_two_arms(n, mean_arms, alpha):
    '''
    generate historical samples for setting with K=2 arms
    alpha - percentage of dataset coming from first arm
    '''
    dataset = {'0': [], '1': []}
    for _ in range(int(alpha*n)):
        dataset['0'].append(np.random.binomial(p=mean_arms[0], n=1))
    for _ in range(n - int(alpha*n)):
        dataset['1'].append(np.random.binomial(p=mean_arms[1], n=1))
    return dataset


class Environment:
    def __init__(self, K, N):
        '''
        MAB simulator for K-armed bandit with N historical samples

        returns dictionary: k -> array of reward from historical pulls
        '''
        self.K = K
        self.N = N

        # TODO: adjust means to be sorted on features
        self.mean_arms = np.random.rand(K)

        # historical pulls
        self.history = {}
        for k in range(K):

        historical_distrib = np.random.rand(K)  # distribution of pulls to each arm in history; not based on reward
        historical_distrib /= historical_distrib.sum()
        arms_pulled = np.random.choice(K, size=N, p=historical_distrib)

        for k in range(K):
            num_pulls = np.sum(arms_pulled == k)
            rewards = np.random.binomial(n=1, p=self.mean_arms[k], size=num_pulls)
            self.history[k] = rewards

        self.history_pos = np.zeros(K)  # track position of views in historical data

        self.online_pulls = []  # tracker for online pulls


    def pull_arm(self, k):
        ''' return observed reward from pulling arm k '''
        reward = np.random.binomial(n=1, p=self.mean_arms[k])
        self.online_pulls.append((k, reward))


    def get_historical_pull(self, k):
        ''' returns None if we have exhausted all historical samples from arm k '''
        if self.history_pos[k] >= len(self.history[k]):
            return None

        reward = self.history[k][self.history_pos[k]]
        self.history_pos[k] += 1
        return reward
