import json
import os
import time
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import random
from typing import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib as mpl
import numpy as np

from utils import evaluate_agent, dict_from_task, make_env

class Teacher:
    def __init__(self, model):
        self.evaluate_envs = []
        evaluate_tasks = []
        elements_1 = np.linspace(3, 100, 10)
        elements_2 = np.linspace(24, 5, 10)
        for e1, e2 in zip(elements_1, elements_2):
            evaluate_tasks.append([float(e1), float(e2)])
        for task in evaluate_tasks:
            self.evaluate_envs.append(DummyVecEnv([make_env(0, config_dict=dict_from_task(task))]))
        self.competences = []
        self.model= model
        self.seed = 111
        self.random_state = np.random.RandomState(self.seed)

    def compute_competence(self):
        sum = 0
        for env in self.evaluate_envs:
            sum += evaluate_agent(self.model, env)
        return sum


class OracleTeacher(Teacher):
    def __init__(self, model, fit_every: int = 3, initial_state: np.ndarray = None, direction_vector: np.ndarray = None):
        super().__init__(model)
        self.fit_every = fit_every
        if initial_state is None:
            self.state = np.array([[3, 24]])
        else:
            self.state = initial_state
        if direction_vector is None:
            self.direction = np.array([[1, -0.2]])
        else:
            self.direction = direction_vector
        self.last_sum = -np.inf
        self.current_sum = 0
        self.step = 0
        
    def sample_task(self):
        return (self.state + self.direction * random.random())[0, :]
    
    def update(self, task, reward):
        self.step += 1

        if self.step % self.fit_every == 0:
            self.current_sum = self.compute_competence()
            self.competences.append(self.current_sum)
            if self.current_sum > self.last_sum:
                print(f"Fitted with r = {self.current_sum} agains r_old = {self.last_sum}")
                self.last_sum = self.current_sum
                self.state = self.state + self.direction
            else:
                print(f"Not fitted with r_old = {self.last_sum}")
            print(f"Current state: {self.state}")
            self.current_sum = 0
            
            x = np.linspace(0,self.step, len(self.competences))
            fig, ax = plt.subplots(1, 1)
            ax.plot(x, np.array(self.competences))
            fig.savefig(f"various_imgs/no_{self.step}")
            plt.close(fig)

class RandomTeacher(Teacher):
    def __init__(self, param_bounds, model):
        super().__init__(model)
        self.param_bounds = param_bounds
        self.mins = np.array([low for (low, _) in self.param_bounds])
        self.maxs = np.array([high for (_, high) in self.param_bounds])
        self.steps = 0

    def sample_task(self):
        self.steps += 1
        return self._sample_random()
    
    def update(self, task, reward):
        self.competences.append(self.compute_competence())
        print("\n\n\n")
        print(self.steps)
        print("\n\n\n")
        x = np.linspace(0,self.steps, len(self.competences))
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, np.array(self.competences))
        fig.savefig(f"various_imgs/no_rand_{self.steps}")
        plt.close(fig)

    def _sample_random(self):
        return np.array([(high-low) * self.random_state.rand() + low for (low, high) in self.param_bounds])



        

    
def proportional_choice(v, random_state):
    probas = np.array(v) / np.sum(v)
    return np.random.choice(range(len(v)), p=probas)

def _get_covariance_matrix(gmm, idx, save_path=None):
    cov_type = gmm.covariance_type
    if cov_type == 'full':
        return gmm.covariances_[idx]
    elif cov_type == 'tied':
        return gmm.covariances_
    elif cov_type == 'diag':
        return np.diag(gmm.covariances_[idx])
    elif cov_type == 'spherical':
        D = gmm.means_.shape[1]
        return np.eye(D) * gmm.covariances_[idx]
    
def scale_to_range(x, old_min, old_max, new_min, new_max):
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)
    
def plot_gmm_2d(gmm, tasks_scaled, alps, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Normalize ALP values for colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["hot_r"]  # red = high ALP

    # Scatter plot with ALP coloring
    for point, alp in zip(tasks_scaled, alps):
        ax.scatter(np.array(point[0]), np.array(point[1]), color=cmap(norm(alp)), s=8, alpha=0.8)

    # Plot GMM components as ellipses
    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        cov = _get_covariance_matrix(gmm, i)

        # Ensure covariance is 2x2 for 2D plotting
        cov_2d = cov[:2, :2] if cov.shape[0] > 2 else cov
        lambda_, v = np.linalg.eigh(cov_2d)
        lambda_ = np.sqrt(lambda_)
        angle = np.degrees(np.arctan2(*v[:, 0][::-1]))

        ellipse = Ellipse(
            xy=mean,
            width=2 * lambda_[0],
            height=2 * lambda_[1],
            angle=angle,
            alpha=0.3,
            color='blue'
        )
        ax.add_patch(ellipse)

    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Static Y Limit")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Absolute Learning Progress")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ GMM plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

class ALPGMMTeacher(Teacher):
    def __init__(self, param_bounds, model, max_history=500, fit_every=2):
        super().__init__(model)
        self.param_bounds = param_bounds
        self.max_history = max_history
        self.task_history = deque(maxlen=max_history)
        self.alp_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=None)
        self.gmm = None
        self.fit_every = fit_every
        self.steps = 0
        self.gmm_components = len(param_bounds)+1
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')

        self.mins = np.array([low for (low, _) in self.param_bounds])
        self.maxs = np.array([high for (_, high) in self.param_bounds])

    def sample_task(self):
        self.steps += 1

        if self.gmm is None or np.random.rand() < 0.2 or len(self.task_history) < 10:
            return self._sample_random()

        self.alp_means = [mean[-1] for mean in self.gmm.means_]
        idx = proportional_choice(self.alp_means, self.random_state)

        # Sample from the selected GMM component
        new_task = self.random_state.multivariate_normal(
            self.gmm.means_[idx], _get_covariance_matrix(self.gmm, idx)
        )

        # Inverse-transform GMM-scaled data
        print(f"new task: {new_task}")
        new_task = self._inverse_scale_task(np.array([new_task.reshape(1, -1)[0][:-1]])).T  # Remove ALP dim
        print(f"new task after inverse transform: {new_task}")

        # Clip to bounds
        new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
        print(f"new task after clipping: {new_task[0, :]}")

        return new_task[0, :]

    def update(self, task, reward):
        """Call this after evaluating agent on a task."""

        alp = self._compute_alp(task, reward)

        self.reward_history.append(reward)
        self.task_history.append(task)
        self.alp_history.append(alp)

        self.competences.append(self.compute_competence())
        print("\n\n\n")
        print(self.steps)
        print("\n\n\n")
        x = np.linspace(0,self.steps, len(self.competences))
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, np.array(self.competences))
        fig.savefig(f"various_imgs/no_alp_{self.steps}")
        plt.close(fig)

        if self.steps % self.fit_every == 0 and self.steps != 0 and len(self.task_history) >= 10:
            self._fit_gmm()
            


    def _sample_random(self):
        return np.array([(high-low) * self.random_state.rand() + low for (low, high) in self.param_bounds])

    def _clip_task(self, task):
        return np.clip(task, [low for (low, _) in self.param_bounds], [high for (_, high) in self.param_bounds])
    
    def _scale_task(self, task):
        scaled_task = np.array([
            scale_to_range(task[:, i], self.mins[i], self.maxs[i], 0, 1)
            for i in range(task.shape[1])
        ]).T
        return scaled_task
    
    def _inverse_scale_task(self, scaled_task):
        inv_scaled_task = np.array([
            scale_to_range(scaled_task[:, i], 0, 1, self.mins[i], self.maxs[i])
            for i in range(len(scaled_task[0]))
        ])
        return inv_scaled_task
    
    def _scale_alp(self, alp):
        if len(self.alp_history) == 0:
            return alp
        
        min_alp = np.min(self.alp_history)
        max_alp = np.max(self.alp_history)
        if max_alp == min_alp:
            return 0.0
        return (alp - min_alp) / (max_alp - min_alp)

    def _fit_gmm(self):
        for t in self.task_history:
            print("Task:", t)
        tasks = np.array(self.task_history)
        alps = np.array(self.alp_history)

        print(tasks)
        print(alps)

        tasks_scaled = self._scale_task(tasks)
        alps_scaled = self._scale_alp(alps.reshape(-1, 1))

        print(tasks_scaled)
        print(alps_scaled)

        X_scaled = np.hstack([tasks_scaled, alps_scaled])

        gmm_configs = [
            {"n_components": self.gmm_components, "covariance_type": "full"},
            {"n_components": self.gmm_components, "covariance_type": "tied"},
        ]

        for config in gmm_configs:
            gmm = GaussianMixture(**config, random_state=self.seed)
            gmm.fit(X_scaled)
            if self.gmm is None or gmm.aic(X_scaled) < self.gmm.aic(X_scaled):
                self.gmm = gmm

        print(f"Fitted GMM with {self.gmm_components} components after {self.steps} steps.")
        plot_gmm_2d(self.gmm, tasks_scaled, alps_scaled, save_path=f"gmm_plots/gmm_plot_{self.steps}.png")

    def _compute_alp(self, task, reward):
        if len(self.task_history) == 0:
            return 0.0
        
        self.knn.fit(self.task_history)
        distances, indices = self.knn.kneighbors([task], n_neighbors=1)
        reward_old = self.reward_history[indices[0][0]]
        return abs(reward - reward_old)
    
    

 

    