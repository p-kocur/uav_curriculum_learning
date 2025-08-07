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
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib as mpl
import numpy as np

from scripts.gym_wrapper import DroneForestEnv
import scripts.json_utils as jutils


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

class ALPGMMTeacher:
    def __init__(self, param_bounds, max_history=100, fit_every=2):
        self.param_bounds = param_bounds
        self.max_history = max_history
        self.task_history = deque(maxlen=max_history)
        self.alp_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=None)
        self.gmm = None
        self.fit_every = fit_every
        self.steps = 0
        self.gmm_components = len(param_bounds)+1
        self.seed = 123
        self.random_state = np.random.RandomState(self.seed)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')

        self.mins = np.array([low for (low, _) in self.param_bounds])
        self.maxs = np.array([high for (_, high) in self.param_bounds])

    def sample_task(self):
        self.steps += 1

        if self.gmm is None or np.random.rand() < 0.2 or len(self.task_history) < self.max_history // 2:
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

        if self.steps % self.fit_every == 0 and self.steps != 0 and len(self.task_history) >= self.max_history // 2:
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
    
    


def make_env(rank: int, seed: int = 0, config_dict: Dict = {}) -> DroneForestEnv:
    """Make the drone forest environment."""

    def _init() -> DroneForestEnv:
        env = DroneForestEnv(
            actions=config_dict["actions"],
            dt=config_dict["sim_step"],
            x_lim=(config_dict["x_lim"]["min"], config_dict["x_lim"]["max"]),
            y_lim=(config_dict["y_lim"]["min"], config_dict["y_lim"]["max"]),
            y_static_limit=config_dict["y_static_limit"],
            n_trees=config_dict["n_trees"],
            tree_radius_lim=(
                config_dict["tree_radius_lim"]["min"],
                config_dict["tree_radius_lim"]["max"],
            ),
            n_lidar_beams=config_dict["n_lidar_beams"],
            lidar_range=config_dict["lidar_range"],
            min_tree_spare_distance=config_dict["min_tree_spare_distance"],
            max_spawn_attempts=config_dict["max_spawn_attempts"],
            max_speed=config_dict["max_speed"],
            max_acceleration=config_dict["max_acceleration"],
            drone_width_m=config_dict["drone_width"],
            drone_height_m=config_dict["drone_height"],
        )
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def evaluate_agent(model, eval_envs, n_episodes=4):
    total_rewards = []
    for _ in range(n_episodes):
        obs = eval_envs.reset()
        done = [False] * eval_envs.num_envs
        ep_rewards = [0.0 for _ in range(eval_envs.num_envs)]
        
        while not all(done):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = eval_envs.step(actions)
            for i, r in enumerate(rewards):
                if not done[i]:
                    ep_rewards[i] += r
            done = [d or d_ for d, d_ in zip(done, dones)]

        total_rewards.extend(ep_rewards)
    
    return np.mean(total_rewards)

def dict_from_task(task: list):
    config_dict = jutils.read_env_config("./env_config.json")
    config_dict["n_trees"] = int(task[0])
    config_dict["y_static_limit"] = float(task[1])
    return config_dict





if __name__ == "__main__":
    '''
    Initialize as in the original run_rl.py, but with curriculum learning.
    '''
    # Read environment configuration
    config_dict = jutils.read_env_config("./env_config.json")
    if config_dict is None:
        raise ValueError("The environment configuration is invalid.")
    
    # Check gym wrapper
    check_env(make_env(0, config_dict=config_dict)())

    # Read the RL configuration
    rl_dict = jutils.read_rl_config("./rl_config.json")
    if rl_dict is None:
        raise ValueError("The RL configuration is invalid.")

    # Create log dir where evaluation results will be saved
    exp_dir = str(int(time.time()))
    log_dir = os.path.join(f"./logs_{rl_dict['algorithm']}", exp_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Preserve the environment configuration
    with open(os.path.join(log_dir, "env_config.json"), "w") as config_file:
        json.dump(config_dict, config_file)

    # Preserve the RL configuration
    with open(os.path.join(log_dir, "rl_config.json"), "w") as rl_file:
        json.dump(rl_dict, rl_file)

    # Create the environments
    train_envs = SubprocVecEnv(
        [
            make_env(i, config_dict=config_dict)
            for i in range(rl_dict["nb_training_envs"])
        ]
    )

    eval_envs = SubprocVecEnv(
        [make_env(i, config_dict=config_dict) for i in range(rl_dict["nb_eval_envs"])]
    )

    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(rl_dict["nb_eval_every"] // rl_dict["nb_training_envs"], 1),
        n_eval_episodes=4,
        deterministic=True,
        render=False,
    )

    # Create the reinforcement learning model
    if rl_dict["activation_fn"] == "sigmoid":
        activation_fn = torch.nn.Sigmoid
    elif rl_dict["activation_fn"] == "tanh":
        activation_fn = torch.nn.Tanh
    else:
        activation_fn = torch.nn.ReLU

    net_arch = [rl_dict["nb_neurons"]] * rl_dict["nb_layers"]

    model = PPO(
        "MlpPolicy",
        train_envs,
        verbose=0,
        tensorboard_log=log_dir,
        policy_kwargs={"activation_fn": activation_fn, "net_arch": net_arch},
        target_kl=0.1,
        clip_range=0.2,

    )

    '''
    Perform curriculum learning using ALPGMMTeacher.
    '''
    # Define parameter bounds for the task
    param_bounds = [
        (3, 100), # n_trees
        (5.0, 24) # y_static_limit
    ]
    teacher = ALPGMMTeacher(param_bounds)
    total_steps = rl_dict["nb_training_steps"]
    step_chunk = 1000  # update curriculum every X steps

    for t in range(0, total_steps, step_chunk):
        task = teacher.sample_task()  
        config_dict = dict_from_task(task)

        print(f"\n\n\nTask: {task}\n\n\n")

        # Nowe treningowe środowiska
        train_envs = SubprocVecEnv(
            [make_env(i, config_dict=config_dict) 
            for i in range(rl_dict["nb_training_envs"])]
        )
        
        model.set_env(train_envs)
        
        model.learn(total_timesteps=step_chunk, reset_num_timesteps=False, callback=eval_callback)

        eval_envs_task = SubprocVecEnv(
            [make_env(i, config_dict=config_dict) for i in range(rl_dict["nb_eval_envs"])]
        )

        reward = evaluate_agent(model, eval_envs_task, n_episodes=4)

        teacher.update(task, reward)
