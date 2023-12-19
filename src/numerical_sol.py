# находим численное решение системы диффуров на вероятности состояний

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.integrate as integrate

from typing import Callable


# определяем функции интесивности прибытия людей в очередь
def phi_lin(t: float, alpha: float=1):
    return alpha * t

def phi_poly(t: float, alpha: float=1, deg: float=2):
    return alpha * (t ** deg)

def phi_exp(t: float, alpha: float=1):
    return alpha * np.exp(t)


# определяем правую часть в СОДУ
def rhs(t: float, pi: np.ndarray, lamb: float, phi_func: Callable):
    """
    :param float t: момент времени
    :param np.ndarray y: набор вероятностей в данный момент времени (может быть ненормированным в общем случе)
    :param float lamb: интесивность кассира
    :param Callable phi_func: функция интесивности наплыва людей
    """
    # container for answer
    rhs_value = np.empty(pi.shape[0])
    # current parameters
    cur_phi_val = phi_func(t)
    
    rhs_value[0] = - cur_phi_val * pi[0] + lamb * pi[1]
    rhs_value[-1] =  cur_phi_val * pi[-2] - lamb * pi[-1]

    for i in range(1, pi.shape[0] - 1):
        rhs_value[i] = lamb * pi[i + 1] + cur_phi_val * pi[i - 1] - (lamb + cur_phi_val) * pi[i]

    return rhs_value


# кол-во людей в очереди
N = 10
# конечное время
T = 10
# параметр lambda
cur_lamb = 3
# конеретная функция phi
def cur_phi_func(t: float):
    return phi_exp(t, alpha=1)


# начинаем из состояния пустой очереди. В идеале, сумма pi_i должна сохряняться в процессе
pi_init = np.zeros(N + 1)
pi_init[0] = 1

# интегрируем
ode_sol = integrate.solve_ivp(rhs, (0, T), pi_init, args=(cur_lamb, cur_phi_func))
print(f'ODE status = {ode_sol["status"]}')

t_grid, pi_vals = ode_sol['t'], ode_sol['y']
print(f'Num of available points = {t_grid.size}')

# проверяем, что все pi_i суммируются в единицы
print(np.sum(pi_vals, axis=0))

# вычисляем среднее кол-во людей в каждой точке
mean_vals = np.sum(pi_vals * np.arange(0, N + 1).reshape(-1, 1), axis=0)

# вычисляем дисперсии
std_vals = np.sqrt( np.sum(pi_vals * (np.arange(0, N + 1).reshape(-1, 1) ** 2), axis=0) - mean_vals ** 2 )


# отрисовка вероятностей
fig, ax = plt.subplots(figsize=(8, 8))

for i in range(0, N + 1):
    ax.plot(t_grid, pi_vals[i], label='$\\pi_{num}$'.format(num = i))

ax.grid(True)
ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('$\\pi$')
ax.set_title(f'N = {N}; $\\lambda = {cur_lamb}$')

fig.savefig('img\\numeric_sol\\probabilities_N_{N_num}_lambda_{lamb_v}_poly.png'.format(N_num = N, lamb_v = cur_lamb))

# отрисовка средних значений
fig, ax = plt.subplots(figsize=(8, 8))

ax.fill_between(t_grid, mean_vals - std_vals, mean_vals + std_vals)
ax.plot(t_grid, mean_vals, color='orange')

ax.grid(True)
ax.set_xlabel('t')
ax.set_ylabel(r'$\mathbb{E}[S]$')
ax.set_title(f'N = {N}; $\\lambda = {cur_lamb}$')

fig.savefig('img\\numeric_sol\\avers_N_{N_num}_lambda_{lamb_v}_poly.png'.format(N_num = N, lamb_v = cur_lamb))