import config.minetest.tasks

from core.utils import make_atari
from core.utils import make_minetest
from gym import envs

import time

if __name__ == "__main__":
    # print(envs.registry.all())

    envAtari = make_atari("BreakoutNoFrameskip-v4")
    envAtari.reset()

    envMinetest = make_minetest("minetester-treechop_shaped-v0")
    envMinetest.reset()

    print('envs reset')

    for env, name in [(envAtari, 'atari'), (envMinetest, 'minetest')]:
        print(name)
        print('action space:', env.action_space)
        print('obs space:', env.observation_space)
        obs, rew, done, info = env.step(env.action_space.sample())
        print('obs shape:', obs.shape)
        print('reward:', rew)
        print('info:', info)
        print('Length:', )
        # breakpoint()

    step = 0
    render = False
    
    while True:
        try:
            actionAtari = envAtari.action_space.sample()
            observation, reward, doneAtari, info = envAtari.step(actionAtari)
            actionMinetest = envMinetest.action_space.sample()
            observation, reward, doneMinetest, info = envMinetest.step(actionMinetest)
            if render:
                envAtari.render()
                envMinetest.render()
            if doneAtari:
                print('done atari')
                envAtari.reset()
            if doneMinetest:
                print('done minetest')
                envMinetest.reset()
            step += 1
        except KeyboardInterrupt:
            break
    envAtari.close()
    envMinetest.close()
