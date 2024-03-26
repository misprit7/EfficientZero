from core.utils import make_minetest
from gym import envs

print(envs.registry.all())

env = make_minetest("treechop_shaped-v0")
env.reset()

step = 0
render = False

while True:
    try:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(step, observation, reward, done, info)
        if render:
            env.render()
        if done:
            env.reset()
        step += 1
    except KeyboardInterrupt:
        break
env.close()