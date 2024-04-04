import config.minetest.tasks
import gymnasium as gym
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import os
plt.ion()

from core.utils import make_minetest
from core.utils import WarpFrame, TimeLimit
from core.game import GameHistory
from config.minetest import game_config
from core.replay_buffer import ReplayBuffer

save_directory = "./saved_minetest_games/"
os.makedirs(save_directory, exist_ok=True)

env = make_minetest(
    "minetester-treechop_shaped-v0",
    idx=30,
    save_video=False
    #xvfb=True,
)
env = WarpFrame(env, width=96, height=96, grayscale=True)
env = TimeLimit(env, max_episode_steps=10)


env.reset()
env.step(4)
env.render()

game_config.set_obs_space()

print(game_config.obs_shape)

trajectory = GameHistory(env.action_space, max_length= game_config.history_length, config=game_config)


# for _ in range(12):
#     env.step(7)
# for _ in range(30):
#     env.step(9)

step = 0
total_reward = 0
while True:
    s = input()
    c = 4
    if s == 'exit' or s == 'q':
        break
    elif 'r' in s:
        env.reset()

    # Left/right
    if 'a' in s:
        c = 1
    elif 'd' in s:
        c = 7
    elif 'w' in s:
        c = 3
    elif 's' in s:
        c = 5

    if s in [str(i) for i in range(10)]:
        c = int(s)

    # Jump
    if ' ' in s:
        c += 9

    # Forward
    if 'e' in s:
        c += 9

    obs, r, done, info = env.step(c)
    trajectory.append(c, obs, r)
    print(step)
    print('r: ', r)
    env.render()
    # plt.imshow(obs)
    # plt.show()
    step+=1
    total_reward += r

env.close()

file_name = f"trajectory_step_{step}_reward_{total_reward}.pkl"

print(trajectory.__len__())
file_path = os.path.join(save_directory, file_name)

with open(file_path, "wb") as f:
    pickle.dump(trajectory, f)


# print("retrieving file")
# with open(file_path, "rb") as f:
#     loaded_trajectory = pickle.load(f)

# print(type(loaded_trajectory))