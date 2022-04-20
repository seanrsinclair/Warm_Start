import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from algorithms import historical_ucb, ucb, online_wrapper, thompson_sampling, historical_thompson_sampling, ids, historical_ids
from common import generate_dataset_two_arms, Environment

# initialize params
seed = 294

num_iters = 50
K = 5     # num arms
T = 1000  # time horizon
N = 200   # num historical samples

# delta = 0.1
# alpha = 0.5 # percentage of time to pull arm 0
# mean_arms = [.5, .5-delta]  # note: arm 0 always better
# n_arms = len(mean_arms)


np.random.seed(seed)
regret_data = []

mean_arms, dataset = None, None # generates unused dataset just to initialize the various algorithms



online_ucb_use_all_data_count = 0.0
online_ucb_data_use_percentage = []


for iter in range(num_iters):

    # dataset = generate_dataset_two_arms(N, mean_arms, alpha)
    env = Environment(K, N)
    dataset = env.history
    mean_arms = env.mean_arms

    print('-------------------')
    print(f'iteration {iter}, means {np.round(mean_arms, 3)}')

    algo_list = {
            'Historical UCB':               historical_ucb.HistoricalUCB(mean_arms, dataset, K),
            'Ignorant UCB':                 ucb.UCB(mean_arms, dataset, K),
            'Ignorant Thompson Sampling':   thompson_sampling.ThompsonSampling(mean_arms, dataset, K),
            'Historical Thompson Sampling': historical_thompson_sampling.HistoricalThompsonSampling(mean_arms, dataset, K),
            'Pseudo Online UCB':            online_wrapper.OnlineWrapper(mean_arms, dataset, N, K, ucb.UCB(mean_arms, dataset, K)),
            'Pseudo Online TS':             online_wrapper.OnlineWrapper(mean_arms, dataset, N, K, thompson_sampling.ThompsonSampling(mean_arms, dataset, K)),
            'IDS':                          ids.IDS(mean_arms, dataset, K, False),
            'Historical IDS':               historical_ids.HistoricalIDS(mean_arms, dataset, K, False),
            'Pseudo Online IDS':            online_wrapper.OnlineWrapper(mean_arms, dataset, N, K, ids.IDS(mean_arms, dataset, K, False))
        }

    # for algo in algo_list:
    #     algorithm = algo_list[algo]
    #     algorithm.reset(env.dataset, env.mean_arms)

    for t in range(T+N):
        for algo in algo_list:
            algorithm = algo_list[algo]
            if algorithm.regret_iterations < T: # if the algorithm is not yet finished
                flag = algorithm.one_step(algorithm.regret_iterations) # run a one-step update of getting arm, observation, and calculating regret
                if flag: regret_data.append({'Algo': algo, 'Iter': iter, 't': algorithm.regret_iterations, 'Regret': algorithm.regret}) # add on its regret to the dataset

    # meta trackers on behavior of online algorithm
    pseudo_online_ucb = algo_list['Pseudo Online UCB']

    if pseudo_online_ucb.used_all_history():
        online_ucb_use_all_data_count += 1

    online_ucb_data_use_percentage.append(pseudo_online_ucb.history_use_percentage())

print('----------------------------')
print('Stats from online algorithm')
print(f'  Percentage of trials entire dataset used:   {100 * online_ucb_use_all_data_count / num_iters}')
print(f'  Average percentage of historical data used: {100 * np.mean(online_ucb_data_use_percentage):.2f}')



df = pd.DataFrame(regret_data)
p = sns.lineplot(data = df, x='t', y='Regret', hue='Algo', ci='sd') # ci=None) #, ci='sd')
p.set_title(f'K = {K}, N = {N}')
# p.set_title(f'K = {K}, N = {N}, delta = {delta}, alpha = {alpha}')
plt.show()
