import json
import os
import time
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import deque
import random
from typing import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from scripts.gym_wrapper import DroneForestEnv
import scripts.json_utils as jutils


def proportional_choice(v, random_state):
    probas = np.array(v) / np.sum(v)
    return np.where(random_state.multinomial(1, probas) == 1)[0][0]

class ALPGMMTeacher:
    def __init__(self, param_bounds, max_history=100, gmm_components=3, fit_every=3):
        self.param_bounds = param_bounds
        self.task_history = deque(maxlen=max_history)
        self.alp_history = deque(maxlen=max_history)
        self.last_rewards = {}
        self.gmm = None
        self.fit_every = fit_every
        self.steps = 0
        self.gmm_components = gmm_components
        self.seed = np.random.randint(42, 424242)
        self.random_state = np.random.RandomState(self.seed)
        self.eps = 0.9

    def sample_task(self):
        self.steps += 1
        self.eps *= 0.9
        # 10% wybierz zadanie losowo
        if self.gmm is None or np.random.rand() < self.eps:
            return self._sample_uniform()

        self.alp_means = []
        for pos, _, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
            self.alp_means.append(pos[-1])
        
        idx = proportional_choice(self.alp_means, self.random_state)

        mins = np.array([low for (low, _) in self.param_bounds])
        maxs = np.array([high for (_, high) in self.param_bounds])

        new_task = self.random_state.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]
        new_task = np.clip(new_task, mins, maxs).astype(np.float32)


        return new_task

    def update(self, task, reward):
        """Call this after evaluating agent on a task."""
        task_key = tuple(task)
        last_reward = self.last_rewards.get(task_key, reward)
        alp = abs(reward - last_reward)

        self.last_rewards[task_key] = reward
        self.task_history.append(task)
        self.alp_history.append(alp)

        print(self.steps)

        if self.steps % self.fit_every == 0 and self.steps != 0:
            self._fit_gmm()

    def _sample_uniform(self):
        return np.array([(high-low) * np.random.sample() + low for (low, high) in self.param_bounds])

    def _clip_task(self, task):
        return np.clip(task, [low for (low, _) in self.param_bounds], [high for (_, high) in self.param_bounds])

    def _fit_gmm(self):
        tasks = np.array(self.task_history)

        alps = np.array(self.alp_history)
        weights = alps #/ (alps.sum() + 1e-8)

        X = np.hstack([tasks, weights.reshape(-1, 1)])

        print(X)
        self.gmm = GaussianMixture(n_components=self.gmm_components, covariance_type='full')
        self.gmm.fit(X=X)
        print(f"Fitted GMM with {self.gmm_components} components after {self.steps} steps.")


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
    config_dict["min_tree_spare_distance"] = float(task[1])
    config_dict["y_static_limit"] = float(task[2])
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
        (0.1, 2.0), # min_tree_spare_distance
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
