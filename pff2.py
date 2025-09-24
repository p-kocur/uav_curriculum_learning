import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import SAC

from utils import dict_from_task, make_env

env_type = "bipedal"
mins = [0.01, 0.01]
maxs = [1, 1]
elements_1 = np.linspace(mins[0], maxs[0], 5)
elements_2 = np.linspace(mins[1], maxs[1], 5)
evaluate_tasks = []
evaluate_envs = []

model = SAC.load("sac_bipedalwalker.zip")

for e1 in elements_1:
    for e2 in elements_2:
        evaluate_tasks.append([float(e1), float(e2)])
for task in evaluate_tasks:
    print(dict_from_task(task, "bipedal"))
    evaluate_envs.append(make_env(0, config_dict=dict_from_task(task, "bipedal"), env_type=env_type)) 



for env, task in zip(evaluate_envs, evaluate_tasks):
    env.training = False
    env.norm_reward = False

    print("NEW ENV")

    # Evaluate or run the policy
    obs, _ = env.reset()
    sum = 0
    print(f"evaluate task: {task}")
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        result = env.step(action)
        obs, reward, terminated, truncated, _ = result
        sum += reward
        print(sum)
        if terminated:
            obs, _ = env.reset()
            sum = 0
            break
        