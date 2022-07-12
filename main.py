import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from algorithms import historical_ucb, ucb, online_wrapper, thompson_sampling, historical_thompson_sampling, ids, historical_ids
from common import generate_dataset_two_arms, Environment, SimpleEnvironment

# initialize params
seed = 294

num_iters = 100
K = 20     # num arms
T = 1000  # time horizon
N = 200   # num historical samples
RESET_REWARD_FLAG = True
MONOTONE_FLAG = True
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
    env = Environment(K, N, T)
    # env = SimpleEnvironment(.05, 0, 10)
    dataset = env.history
    mean_arms = env.mean_arms

    print('-------------------')
    print(f'iteration {iter}, means {np.round(mean_arms, 3)}')




    # TODO:
        # Easy solution: adjust the confidence radius for the UCB algo's to go from t to T
        # Slightly harder solution: The UCB algo's take as imput the environment which maintains the "timestep", and the
        # UCB algo's use the confidence radius with log numerator being env.time_step(arm) or something like that.
    algo_list = {
            'Ignorant UCB':      ucb.UCB(T, mean_arms, dataset, K, MONOTONE_FLAG=MONOTONE_FLAG),
            'Historical UCB':    historical_ucb.HistoricalUCB(T, mean_arms, dataset, K, MONOTONE_FLAG=MONOTONE_FLAG),
            'Pseudo Online UCB': online_wrapper.OnlineWrapper(mean_arms, dataset, N, K, ucb.UCB(T, mean_arms, dataset, K, MONOTONE_FLAG=MONOTONE_FLAG)),
            # 'Ignorant TS':       thompson_sampling.ThompsonSampling(mean_arms, dataset, K),
            # 'Historical TS':     historical_thompson_sampling.HistoricalThompsonSampling(mean_arms, dataset, K),
            # 'Pseudo Online TS':  online_wrapper.OnlineWrapper(mean_arms, dataset, N, K, thompson_sampling.ThompsonSampling(mean_arms, dataset, K)),
            # 'IDS':               ids.IDS(mean_arms, dataset, K, False),
            # 'Historical IDS':    historical_ids.HistoricalIDS(mean_arms, dataset, K, False),
            # 'Pseudo Online IDS': online_wrapper.OnlineWrapper(mean_arms, dataset, N, K, ids.IDS(mean_arms, dataset, K, False))
        }


    for t in range(T+N):
        for algo in algo_list:
            algorithm = algo_list[algo]
            if algorithm.regret_iterations < T: # if the algorithm is not yet finished
                flag = algorithm.one_step(algorithm.regret_iterations) # run a one-step update of getting arm, observation, and calculating regret
                if flag: regret_data.append({'Algo': algo, 'Iter': iter, 't': algorithm.regret_iterations, 'Regret': algorithm.regret}) # add on its regret to the dataset
        env.reset_algo()
    env.reset_data(RESET_REWARD_FLAG)

    # meta trackers on behavior of online algorithm
    pseudo_online_ucb = algo_list['Pseudo Online UCB']

    if pseudo_online_ucb.used_all_history():
        online_ucb_use_all_data_count += 1

    online_ucb_data_use_percentage.append(pseudo_online_ucb.history_use_percentage())

print('----------------------------')
print('Stats from online algorithm (pseudo online UCB)')
print(f'  Percentage of trials entire dataset used:   {100 * online_ucb_use_all_data_count / num_iters}')
print(f'  Average percentage of historical data used: {100 * np.mean(online_ucb_data_use_percentage):.2f}')

palette = ['darkgreen', 'lime', 'seagreen', 'mediumblue', 'royalblue', 'cornflowerblue', 'darkred', 'indianred', 'salmon']  # 'darkolivegreen', 'olivedrab', 'yellowgreen'

hue_order = ['Ignorant UCB', 'Historical UCB', 'Pseudo Online UCB', 'Ignorant TS', 'Historical TS', 'Pseudo Online TS', 'IDS', 'Historical IDS', 'Pseudo Online IDS']

df = pd.DataFrame(regret_data)
p = sns.lineplot(data = df, hue_order=hue_order, x='t', y='Regret', hue='Algo', palette=palette, ci=None) # ci=None) #, ci='sd')
p.set_title(f'K = {K}, N = {N}')
# p.set_title(f'K = {K}, N = {N}, delta = {delta}, alpha = {alpha}')
plt.show()
