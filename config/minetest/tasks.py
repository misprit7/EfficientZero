import gym
from minetester.minetest_env import Minetest

from config.minetest.utils import (
    AlwaysDig,
    DictToMultiDiscreteActions,
    DiscreteMouseAction,
    FlattenMultiDiscreteActions,
    PenalizeJumping,
    SelectKeyActions,
    ToFloat32Reward,
    Gymnasium2Gym,
)

## This module registers the minetest tasks


## An adapted treechop environment from minetest baselines
## Doesn't apply FrameResize and GrayScale, as these are now handled in __init__.py
def wrapped_treechop_env(**kwargs):
    env = Minetest(
        **kwargs,
    )
    # simplify mouse actions
    env = DiscreteMouseAction(
        env,
        num_mouse_bins=3,
        max_mouse_move=0.05,
        quantization_scheme="linear",
    )
    # make breaking blocks easier to learn
    env = AlwaysDig(env)
    # only allow basic movements
    env = SelectKeyActions(env, select_keys={"FORWARD", "JUMP"})
    # env = SelectKeyActions(env, select_keys={"FORWARD"})
    # jumping usually interrupts progress towards
    # breaking nodes; apply penalty to learn faster
    #env = PenalizeJumping(env, 0.2)
    # transform into pure discrete action space
    env = DictToMultiDiscreteActions(env)
    env = FlattenMultiDiscreteActions(env)
    # cast rewards to float32
    env = ToFloat32Reward(env)
    # Switch to gym api
    env = Gymnasium2Gym(env)
    return env


TASKS = [
    ("treechop", 0, wrapped_treechop_env),
    ("treechop", 1, wrapped_treechop_env),
    ("treechop_shaped", 0, wrapped_treechop_env),
]


for task, version, entry_point in TASKS:
    gym.register(
        f"minetester-{task}-v{version}",
        entry_point=f"{entry_point.__module__}:{entry_point.__name__}",
        kwargs=dict(clientmods=[f"{task}_v{version}"], render_mode="rgb_array"),
    )
