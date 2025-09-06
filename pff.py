import gymnasium as gym
from carl.envs import CARLCartPole, CARLMountainCar

CARLCartPole.render_mode = "human"
env = CARLCartPole()
env.reset()

for i in range(10000):
    action = env.action_space.sample()
    obs, reward, done, info, _ = env.step(action)
    frame = env.render()  # Should return an image frame (NumPy array)
    print(type(frame), frame.shape if frame is not None else None)
    if done:
        obs = env.reset()
        print("Environment reset")