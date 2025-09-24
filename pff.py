import json
import os
import time
import sys
from stable_baselines3 import PPO, SAC

from stable_baselines3 import PPO, SAC, TD3  # import the algorithm you trained with

from gymnasium.wrappers import TimeLimit

# Example: if you trained with SAC
model = SAC.load("sac_bipedalwalker.zip")
from bipedal_parametrized import ParamBipedalWalker

# You need an environment to use the model
env = TimeLimit(ParamBipedalWalker(stump_height=0.01, stump_distance=1, render_mode="human"), max_episode_steps=2000)

# If you used VecNormalize during training, load it too
from stable_baselines3.common.vec_env import VecNormalize

# Don’t forget to set env.training = False and env.norm_reward = False for evaluation
env.training = False
env.norm_reward = False

# Evaluate or run the policy
obs, _ = env.reset()
sum = 0
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    sum += reward
    if terminated or truncated:
        obs, _ = env.reset()
        sum = 0
    print(sum)
