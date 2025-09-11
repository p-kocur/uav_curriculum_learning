import json
import os
import time
import sys
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch
from torch import nn

from huggingface_sb3 import load_from_hub

import scripts.json_utils as jutils
from teachers import OracleTeacher, ALPGMMTeacher, RandomTeacher
from utils import dict_from_task, make_env, evaluate_agent

def get_scenario_config(scenario):
    rl_config_path = "./rl_config.json"
    if scenario == "drone_forest":
        env_config_path = "./env_config_drone_forest.json"
        param_bounds = [
            (3, 100),  # n_trees
            (5.0, 24)  # y_static_limit
        ]
    elif scenario == "cart_pole":
        env_config_path = "./env_config_cart_pole.json"
        param_bounds = [
            (0.01, 1.0),  # masspole
            (0.05, 5.0)   # length
        ]
    elif scenario == "bipedal_walker":
        env_config_path = "./env_config_bipedal_walker.json"
        param_bounds = [
            (0.1, 1.0),  # stump_height
            (0.1, 1.0)   # stump_distance
        ]
    else:
        raise ValueError("Unknown scenario: {}".format(scenario))
    return env_config_path, rl_config_path, param_bounds

def main(scenario, teacher_type):
    env_config_path, rl_config_path, param_bounds = get_scenario_config(scenario)
    
    config_dict = jutils.read_env_config(env_config_path)
    if config_dict is None:
        raise ValueError("The environment configuration is invalid.")

    check_env(make_env(0, config_dict=config_dict, env_type=scenario.split('_')[0])())
    
    rl_dict = jutils.read_rl_config(rl_config_path)
    if rl_dict is None:
        raise ValueError("The RL configuration is invalid.")

    exp_dir = str(int(time.time()))
    log_dir = os.path.join(f"./logs_{rl_dict['algorithm']}_{scenario}", exp_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "env_config.json"), "w") as config_file:
        json.dump(config_dict, config_file)
    with open(os.path.join(log_dir, "rl_config.json"), "w") as rl_file:
        json.dump(rl_dict, rl_file)

    train_envs = SubprocVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs"])]
    ) if torch.cuda.is_available() else DummyVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs"])]
    )
    eval_envs = SubprocVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_eval_envs"])]
    ) if torch.cuda.is_available() else DummyVecEnv(
        [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_eval_envs"])]
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

    # activation_fn = {
    #     "sigmoid": torch.nn.Sigmoid,
    #     "tanh": torch.nn.Tanh,
    #     "relu": torch.nn.ReLU,
    # }.get(rl_dict["activation_fn"], torch.nn.ReLU)
    # net_arch = [rl_dict["nb_neurons"]] * rl_dict["nb_layers"]

    # model = SAC(
    #     "MlpPolicy",
    #     train_envs,
    #     verbose=0,
    #     tensorboard_log=log_dir,
    #     policy_kwargs={"activation_fn": activation_fn, "net_arch": net_arch},
    #     target_kl=0.1,
    #     clip_range=0.2,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )

    # policy_kwargs = dict(
    #     net_arch=[400, 300],
    #     activation_fn=nn.ReLU
    # )

    # # Define SAC model
    # model = SAC(
    #     "MlpPolicy",
    #     train_envs,
    #     policy_kwargs=policy_kwargs,
    #     ent_coef=0.005,             # entropy coefficient
    #     learning_rate=0.001,        # learning rate
    #     train_freq=10,              # gradient updates every 10 steps
    #     batch_size=1000,            # number of samples per gradient step
    #     buffer_size=300_000,      # replay buffer size
    #     verbose=1,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )

    # policy_kwargs = dict(net_arch=[400, 300])  # TD3 default from original paper
    # model = TD3(
    #     "MlpPolicy",
    #     train_envs,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=1e-3,
    #     buffer_size=int(1e6),
    #     batch_size=256,
    #     verbose=1,
    #     seed=0,
    # )

    # policy_kwargs = dict(net_arch=[256, 256])  # two hidden layers (SB3's SAC default is 256)
    # model = SAC(
    #     "MlpPolicy",
    #     train_envs,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=3e-4,
    #     buffer_size=int(1e6),
    #     batch_size=256,
    #     train_freq=1,
    #     verbose=1,
    #     seed=0,
    # )

    # model = TD3.load_from_hub(
    #     repo_id="sb3/td3-BipedalWalker-v3",
    #     # optionally: load custom policy_kwargs if needed (but the saved model should include that)
    # )

    checkpoint = load_from_hub(
        repo_id="araffin/tqc-BipedalWalker-v3",
        filename="tqc-BipedalWalker-v3.zip"
    )

    model = TD3.load(checkpoint)

    if teacher_type == "alpgmm":
        teacher = ALPGMMTeacher(model, param_bounds, env_type=scenario.split('_')[0])
    elif teacher_type == "oracle":
        teacher = OracleTeacher(model, param_bounds, env_type=scenario.split('_')[0])
    elif teacher_type == "random":
        teacher = RandomTeacher(model, param_bounds, env_type=scenario.split('_')[0])
    else:
        raise ValueError("Unknown teacher type: {}".format(teacher_type))

    total_steps = rl_dict["nb_training_steps"]
    step_chunk = 2000

    for t in range(0, total_steps, step_chunk):
        print(f"Training step {t}/{total_steps}")
        task = teacher.sample_task()
        config_dict = dict_from_task(task)
        print(f"\n\n\nTask: {task}\n\n\n")

        train_envs = SubprocVecEnv(
            [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs"])]
        ) if torch.cuda.is_available() else DummyVecEnv(
            [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_training_envs"])]
        )
        model.set_env(train_envs)
        model.learn(total_timesteps=step_chunk, reset_num_timesteps=False, callback=eval_callback)

        eval_envs_task = SubprocVecEnv(
            [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_eval_envs"])]
        ) if torch.cuda.is_available() else DummyVecEnv(
            [make_env(i, config_dict=config_dict, env_type=scenario.split('_')[0]) for i in range(rl_dict["nb_eval_envs"])]
        )
        reward = evaluate_agent(model, eval_envs_task, n_episodes=4)
        teacher.update(task, reward)

    try:
        teacher.plot()
    except Exception as e:
        print(f"Error plotting teacher data: {e}")

    try:
        model.save(os.path.join(log_dir, "final_model"))
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_rl_with_curriculum.py <scenario> <teacher_type>")
        print("Example: python run_rl_with_curriculum.py drone_forest alpgmm")
        sys.exit(1)
    scenario = sys.argv[1]
    teacher_type = sys.argv[2]
    main(scenario, teacher_type)
