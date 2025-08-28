import json
import os
import time
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch

import scripts.json_utils as jutils
from teachers import OracleTeacher, ALPGMMTeacher, RandomTeacher
from utils import dict_from_task, make_env, evaluate_agent


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
    teacher = None
    if "alpgmm" in sys.argv:
        teacher = ALPGMMTeacher(param_bounds, model)
    elif "oracle" in sys.argv:
        teacher = OracleTeacher(model)
    elif "random" in sys.argv:
        teacher = RandomTeacher(param_bounds, model)
    
    total_steps = rl_dict["nb_training_steps"]
    step_chunk = 1000  # update curriculum every X steps

    try:
        for t in range(0, total_steps, step_chunk):
            
            task = teacher.sample_task()  
            config_dict = dict_from_task(task)

            print(f"\n\n\nTask: {task}\n\n\n")

            # Nowe treningowe środowiska
            train_envs = DummyVecEnv(
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
    except Exception as e:
        print(e)

    teacher.plot("test")
