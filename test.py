import config.minetest.tasks

from core.utils import make_atari
from core.utils import make_minetest
from gym import envs

if __name__ == "__main__":
    # print(envs.registry.all())

    envAtari = make_atari("BreakoutNoFrameskip-v4")
    envAtari.reset()

    envMinetest = make_minetest("minetester-treechop_shaped-v0")
    print('starting minetest reset')
    envMinetest.reset()

    print('envs reset')

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
                envAtari.reset()
                envMinetest.reset()
            step += 1
            print('step')
        except KeyboardInterrupt:
            break
    envAtari.close()
    envMinetest.close()
