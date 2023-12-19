import typing as tp

from pathlib import Path

import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt


# функция для генерации нескольких траекторий
def generate_trajectories(
        phi: tp.Callable[[float], float],
        lambd: float,
        N: int,
        t_max: float,
        n_sampling_points: int=1000,
        n_trajectories: int=1000
    ) -> np.ndarray:
    time_ticks = np.linspace(0, t_max, n_sampling_points)
    queue_length_history = np.zeros((n_trajectories, n_sampling_points))
    for i in range(1, time_ticks.shape[0]):
        t = time_ticks[i]
        dt = time_ticks[i] - time_ticks[i-1]
        p_appear = 1 - np.exp(-phi(t) * dt)
        p_leave = 1 - np.exp(-lambd * dt)
        appeared = sps.bernoulli(p_appear).rvs(n_trajectories)
        left = sps.bernoulli(p_leave).rvs(n_trajectories)
        queue_length_history[:, i] = np.clip(queue_length_history[:, i-1] + appeared - left, 0, N)
    return queue_length_history


# общая функция для отрисовки динамики
def plot_dynamics(
        phi: tp.Callable[[float], float],
        lambd: float,
        N: int,
        t_max: float,
        save_path: Path,
        img_name: str
    ):
    trajectories = generate_trajectories(phi, lambd, N, t_max)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        np.linspace(0, t_max, trajectories.shape[1]),
        trajectories.mean(axis=0)
    )
    ax.fill_between(
        np.linspace(0, t_max, trajectories.shape[1]),
        trajectories.mean(axis=0) - trajectories.std(axis=0),
        trajectories.mean(axis=0) + trajectories.std(axis=0),
        alpha = 0.5
    )
    ax.set_title(f'N = {N}, $\lambda$ = {lambd}')
    ax.set_xlabel('t')
    ax.set_ylabel('E[S]')
    ax.grid(True)
    fig.savefig(save_path / img_name)


# Параметры, для которых будем моделировать
params = {
    'phi': [
        lambda t: t,
        lambda t: t,
        lambda t: t**2,
        lambda t: t**2,
        lambda t: np.exp(t)
    ],
    'lambda': [1, 3, 3, 10, 20],
    'N': [10, 10, 10, 10, 10],
    't_max': [8, 15, 20, 40, 10],
    'img_name': [
        't_10_1.png',
        't_10_3.png',
        't^2_10_3.png',
        't^2_10_10.png',
        'e^t_10_20.png'
    ]
}


save_path: Path = Path('../img/sampling')


for i in range(5):
    plot_dynamics(
        params['phi'][i],
        params['lambda'][i],
        params['N'][i],
        params['t_max'][i],
        save_path,
        params['img_name'][i]
    )
