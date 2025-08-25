import numpy as np
from typing import Dict
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from typing import Callable, Dict, Optional
from stable_baselines3.common.utils import set_random_seed

from scripts.gym_wrapper import DroneForestEnv
import scripts.json_utils as jutils

def evaluate_agent(model, eval_envs, n_episodes=4, return_partials=False):
    total_rewards = []
    partial_rewards = np.zeros(eval_envs.num_envs)
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
        partial_rewards = partial_rewards + np.array(ep_rewards)
    
    if return_partials:
        return np.mean(total_rewards), partial_rewards
    else:
        return np.mean(total_rewards)

def make_env(rank: int, seed: int = 0, config_dict: Optional[Dict] = None) -> Callable[[], DroneForestEnv]:
    """Factory function for DroneForestEnv, compatible with SubprocVecEnv."""

    if config_dict is None:
        config_dict = {}

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


def dict_from_task(task: list):
    config_dict = jutils.read_env_config("./env_config.json")
    config_dict["n_trees"] = int(task[0])
    config_dict["y_static_limit"] = float(task[1])
    return config_dict

