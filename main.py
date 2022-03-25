import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from algorithms import historical_ucb, ucb, online_wrapper, thompson_sampling, historical_thompson_sampling
from common import generate_dataset

# initialize params
seed = 294

num_iters = 50
T = 1000  # time horizon
n = 200   # num historical samples
delta = 0.1
alpha = 0.5 # percentage of time to pull arm 0
mean_arms = [.5, .5-delta]  # note: arm 0 always better
n_arms = len(mean_arms)


np.random.seed(seed)
regret_data = []

dataset = generate_dataset(0, mean_arms, alpha) # generates unused dataset just to initialize the various algorithms
algo_list = {
        'Historical UCB': historical_ucb.HistoricalUCB(mean_arms, dataset, n_arms),
        'Ignorant UCB': ucb.UCB(mean_arms, dataset, n_arms),
        'Ignorant Thompson Sampling': thompson_sampling.ThompsonSampling(mean_arms, dataset, n_arms),
        'Historical Thompson Sampling': historical_thompson_sampling.HistoricalThompsonSampling(mean_arms, dataset, n_arms),
        'Pseudo Online UCB': online_wrapper.OnlineWrapper(mean_arms, dataset, n_arms, ucb.UCB(mean_arms, dataset, n_arms)),
        'Pseudo Online TS': online_wrapper.OnlineWrapper(mean_arms, dataset, n_arms, thompson_sampling.ThompsonSampling(mean_arms, dataset, n_arms)),
    }

online_ucb_use_all_data_count = 0.0
online_ucb_data_use_percentage = []


for i in range(num_iters):
    dataset = generate_dataset(n, mean_arms, alpha)
    for algo in algo_list:
        algorithm = algo_list[algo]
        algorithm.reset(dataset)

    for t in range(T+n):
        for algo in algo_list:
            algorithm = algo_list[algo]
            if algorithm.regret_iterations < T: # if the algorithm is not yet finished 
                flag = algorithm.one_step(algorithm.regret_iterations) # run a one-step update of getting arm, observation, and calculating regret
                if flag: regret_data.append({'Algo': algo, 'Iter': i, 't': algorithm.regret_iterations, 'Regret': algorithm.regret}) # add on its regret to the dataset

    # meta trackers on behavior of online algorithm
    pseudo_online_ucb = algo_list['Pseudo Online UCB']
    if pseudo_online_ucb.dataset_index[0] == len(dataset['0']) and pseudo_online_ucb.dataset_index[1] == len(dataset['1']):
        online_ucb_use_all_data_count += 1
    
    online_ucb_data_use_percentage.append((pseudo_online_ucb.dataset_index[0] + pseudo_online_ucb.dataset_index[1]) / n)

print('----------------------------')
print('Stats from online algorithm')
print(f'  Percentage of trials entire dataset used:   {100 * online_ucb_use_all_data_count / num_iters}')
print(f'  Average percentage of historical data used: {100*np.mean(online_ucb_data_use_percentage):.2f}')



df = pd.DataFrame(regret_data)
p = sns.lineplot(data = df, x="t", y="Regret", hue="Algo", ci=None) #, ci="sd")
p.set_title(f"n = {n}, delta = {delta}, alpha = {alpha}")
plt.show()
