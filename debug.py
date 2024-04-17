import config.minetest.tasks
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
plt.ion()

from core.utils import make_minetest
from core.utils import WarpFrame, TimeLimit

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

    # Jump
    if ' ' in s:
        c += 9

    # Forward
    if 'e' in s:
        c += 18

    obs, r, done, info = env.step(c)
    print('r: ', r)
    env.render()
    # plt.imshow(obs)
    # plt.show()

env.close()


