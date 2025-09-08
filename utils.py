import numpy as np
from typing import Dict
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from typing import Callable, Dict, Optional
import gymnasium as gym
try:
    from carl.envs import CARLCartPole
except ImportError:
    pass

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from bipedal_parametrized import ParamBipedalWalker
#from scripts.gym_wrapper import DroneForestEnv
import scripts.json_utils as jutils



class SqueezeObsWrapper(gym.ObservationWrapper):
    """Ensure obs has shape (4,) not (4,1)."""
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        print(observation)
        if isinstance(observation, dict) and "obs" in observation:
            return observation["obs"]
        else:
            return observation["obs"]
        
class RemoveContextWrapper(gym.ObservationWrapper):
    """Keeps only the 'obs' part of CARL environments (removes context)."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["obs"]

    def observation(self, observation):
        return np.squeeze(observation['obs']).astype(np.float32)

    # def reset(self, **kwargs):
    #     obs, info = super().reset(**kwargs)
    #     return self.observation(obs), info
    

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
    
def make_env(rank: int, seed: int = 0, config_dict: Optional[Dict] = None, env_type: str = "drone") -> Callable[[], object]:
    """Factory function for DroneForestEnv or BipedalWalker, compatible with SubprocVecEnv."""

    if config_dict is None:
        config_dict = {}

    if env_type == "drone":
        def _init() -> object:
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
    elif env_type == "cart":
        def _init() -> object:
            context = CARLCartPole.get_default_context()
            context_dict = {}
            for key, value in context.items():
                if key in config_dict:
                    context_dict[key] = np.array([config_dict[key]], dtype=np.float32)
                else:
                    context_dict[key] = np.array([context.get(key)], dtype=np.float32)
            context = context_dict
            # env = CARLCartPole(contexts={
            #     "0": {"masspole": np.array([config_dict.get("masspole", 0.1)], dtype=np.float32), 
            #           "length": np.array([config_dict.get("length", 0.5)], dtype=np.float32)},
            # })
            env = CARLCartPole(contexts={0: context})
            env = RemoveContextWrapper(env)
            env.reset(seed=seed + rank)
            return env
    elif env_type == "bipedal": 
        def _init() -> object:
            stump_height = config_dict.get("stump_height", 1.0)
            stump_distance = config_dict.get("stump_distance", 1.0)
            env = TimeLimit(ParamBipedalWalker(stump_height=stump_height, stump_distance=stump_distance), max_episode_steps=2000)
            env.reset(seed=seed + rank)
            return env
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
        

    set_random_seed(seed)
    return _init


def dict_from_task(task: list, env_type: str = "drone"):
    if env_type == "drone":
        config_dict = jutils.read_env_config("./env_config.json")
        config_dict["n_trees"] = int(task[0])
        config_dict["y_static_limit"] = float(task[1])
        return config_dict
    elif env_type == "cartpole":
        # Example: task = [masspole, length]
        config_dict = {}
        config_dict["masspole"] = float(task[0])
        config_dict["length"] = float(task[1])
        # Add more parameters as needed
        return config_dict
    elif env_type == "bipedal":
        # Example: task = [stump_height, stump_distance]
        config_dict = {}
        config_dict["stump_height"] = float(task[0])
        config_dict["stump_distance"] = float(task[1])
        return config_dict
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
