import numpy as np
from core.game import Game
from core.utils import arr_to_str, MaxAndSkipEnv, NoopResetEnv, TimeLimit
import gym


class MinetestWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=True):
        """Minetest Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.action_space.n, discount)
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def get_max_episode_steps(self):
        return self.env.get_max_episode_steps()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()
        
def make_minetest(env_id, skip=4, max_episode_steps=None, idx=0, xvfb=False):
    env = gym.make(
        env_id,
        world_seed=0, # TODO: Make adjustable
        start_xvfb=False,
        headless=(not xvfb), # TODO Make adjustable
        env_port=5555 + idx,
        server_port=30000 + idx,
        render_mode='rgb_array',
    )
    # env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


