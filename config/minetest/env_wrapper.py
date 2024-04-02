import numpy as np
from core.game import Game
from core.utils import arr_to_str, MaxAndSkipEnv, NoopResetEnv, TimeLimit
import gym


class MinetestWrapper(Game):
    total_reward = 0
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
        self.total_reward += reward

        return observation, reward, done, info

    def reset(self, **kwargs):
        print('Starting minetest reset')
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)
        print(f'Finished minetest reset: {self.total_reward}')
        self.total_reward = 0

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()
        

