import numpy as np

def generate_dataset(n, mean_arms, alpha):
    '''
        generate historical samples
        alpha - percentage of dataset coming from first arm
    '''
    dataset = {'0': [], '1': []}
    for _ in range(int(alpha*n)):
        dataset['0'].append(np.random.binomial(p=mean_arms[0], n=1))
    for _ in range(n - int(alpha*n)):
        dataset['1'].append(np.random.binomial(p=mean_arms[1], n=1))
    return dataset
