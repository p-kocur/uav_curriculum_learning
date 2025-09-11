import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC

def main():
    # Create the training environment
    train_env = gym.make("BipedalWalker-v3")

    # Define the policy network architecture
    policy_kwargs = dict(
        net_arch=[400, 300],
        activation_fn=nn.ReLU
    )

    # Initialize SAC model
    model = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        ent_coef=0.005,             # entropy coefficient
        learning_rate=0.001,        # learning rate
        train_freq=10,              # gradient updates every 10 steps
        batch_size=1000,            # number of samples per gradient step
        buffer_size=300_000,        # replay buffer size
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Train for 1,000,000 steps
    model.learn(total_timesteps=1_000_000)

    # Save the trained model
    model.save("sac_bipedalwalker")

    # Close env
    train_env.close()

if __name__ == "__main__":
    main()
