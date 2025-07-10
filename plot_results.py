import numpy as np
import matplotlib.pyplot as plt

data = np.load("./logs_ppo/1747756047/evaluations.npz")
print("Timesteps:", data["timesteps"])
print("Rewards:", data["results"])
print("Episode lengths:", data["ep_lengths"])

plt.figure(figsize=(10, 5))
plt.plot(data["timesteps"], data["results"], label="Rewards")
plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.title("PPO Training Rewards")
plt.legend()
plt.grid()
plt.savefig("ppo_training_rewards_4.png")
plt.show()