# здесь строим графики для простого случая постоянного потока покупателей

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats


def get_distr(q: float, N: int):
    """get array of probailities for current paramters. Result in procents
    """
    if q == 1:
        return ((100 / (N + 1)) * np.ones(N + 1)).tolist()

    pi_0 = (q - 1) / (q ** (N + 1) - 1)
    distr = [(q ** j) * pi_0 * 100 for j in range(0, N + 1)]

    return distr


# generating distibutions
# task paramters
N = 10
q_ar = [0.5, 0.9, 2, 5]

# for num, q in enumerate(q_ar):
#     cur_distr = get_distr(q, N)
#     states = list(range(0, N + 1))
#     # checking correctness of distribution
#     #print(sum(cur_distr))

#     fig, ax = plt.subplots()
#     ax.bar(states, cur_distr, edgecolor='violet')
#     ax.grid(True)
#     ax.set_title(f'N = {N}; q = {q}')
#     ax.set_xlabel('State')
#     ax.set_ylabel('p, %')

#     fig.savefig(f'img\stationary_case\distr_{num}.png', )
#     pass


# computing averages for diffrent q with grid on N
q_ar = [0.6, 0.9, 0.99, 1, 1.01, 1.1, 1.5]
fig, ax = plt.subplots(figsize=(10, 10))

avers_container = []

for q in q_ar:
    N_grid = list(range(1, 200 + 1))
    cur_avers = np.empty(len(N_grid))
    
    for num, cur_N in enumerate(N_grid):
        cur_distr = np.array(get_distr(q, cur_N)) / 100
        cur_avers[num] = np.sum( cur_distr * np.arange(cur_N + 1) )

    avers_container.append(cur_avers)

    # plotting
    if q == 1:
        ax.plot(N_grid, cur_avers, label=f'q = {q}', linestyle=':')
    else:
        ax.plot(N_grid, cur_avers, label=f'q = {q}')

# plot maximum possible average
ax.plot(np.arange(1, 200 + 1), np.arange(1, 200 + 1), label='max E', linestyle='--')

ax.set_xlabel('N')
ax.set_ylabel(r'$\mathbb{E}[S]$')
ax.grid(True)
ax.legend()

fig.savefig(r'img\stationary_case\averages.png')
