import warnings
from . import maze, models
from typing import List, Tuple
from functools import cache
import numpy as np
from procgen import ProcgenGym3Env
import pickle

# FIXME: Memory limit on cache (possibly configurable)

class Episode():
    """
    A recorded episode. Contains the following:
    - state_vals: dict of state vals on first timestep

    Has helpers to access: grid, inner_grid
    """
    __slots__ = (
        "initial_state_bytes", "mouse_positions_outer", "rewards", "actions",
        "sampler", "level_seed", "extra",
    )

    # the actual properties saved
    initial_state_bytes: bytes
    mouse_positions_outer: List[Tuple[int, int]]
    rewards: List[float]
    actions: List[int]
    sampler: str
    level_seed: int
    extra: dict

    def __init__(self, **kwargs):
        if 'extra' not in kwargs: kwargs['extra'] = {}

        for slot in self.__slots__:
            assert slot in kwargs, f"Missing {slot}"
        self.__setstate__(kwargs)

    def assert_valid(self):
        # copilot wrote most of this <3
        assert isinstance(self.initial_state_bytes, bytes)
        assert isinstance(self.mouse_positions_outer, list)
        assert isinstance(self.rewards, list)
        assert isinstance(self.actions, list)
        assert isinstance(self.sampler, str)
        assert isinstance(self.extra, dict)
        assert isinstance(self.level_seed, int)

        # FIXME: Should be equal to len(self.mouse_positions_outer), but gatherdata.py had a bug
        assert len(self.actions) == len(self.rewards) == len(self.mouse_positions_outer)-1
        assert all(isinstance(x, int) for x in self.actions)
        assert all(isinstance(x, float) for x in self.rewards)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in self.mouse_positions_outer)

    def __setitem__(self, key, value):
        if key in self.__slots__:
            warnings.warn('WARNING: dict access sets episode.extra[key] not episode[key]')
        self.extra[key] = value

    def __getitem__(self, key):
        if key in self.__slots__:
            warnings.warn('WARNING: dict access gets episode.extra[key] not episode[key]')
        return self.extra[key]

    def __setattr__(self, name, value):
        if name in self.__slots__:
            raise TypeError(f"Cannot set {name}")
        super().__setattr__(name, value)

    def __getstate__(self):
        self.assert_valid()
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        if 'extra' not in state: state['extra'] = {}
        for k,v in state.items():
            object.__setattr__(self, k, v)
        self.assert_valid()

    @property
    @cache
    def env_state(self) -> maze.EnvState:
        return maze.EnvState(self.initial_state_bytes)

    @property
    def steps(self) -> int:
        return len(self.mouse_positions_outer)

    @cache
    def grid(self, t=0):
        "Return the grid in inner coordinates, with the mouse at a specific timestep"
        g = self.env_state.full_grid(with_mouse=False).copy()
        g[self.mouse_positions_outer[t]] = maze.MOUSE
        return maze.inner_grid(g)


    @property
    @cache
    def got_cheese(self) -> bool:
        "Checks if mouse got the cheese. Currently a hack, will fix when we re-run gatherdata."
        g = self.grid(t=-1)
        # previously we checked if the mouse was adjacent to the cheese
        return (np.abs(np.array(maze.get_mouse_pos(g)) - np.array(maze.get_cheese_pos(g))).sum() == 1.).all()


def load_episode(file: str, load_venv=False) -> Episode:
    """
    Load an episode from a file, optionally load the venv, allowing us to reconstruct the high-def
    image of the maze (stored in run["start_info"]["rgb"])
    (Disabled by default for speed, since it requires creating a venv)
    """
    with open(file, 'rb') as f:
        state: dict = pickle.load(f)
        episode = Episode(**state)

    if load_venv:
        venv = ProcgenGym3Env(
            num=1, env_name='maze', num_levels=1, start_level=episode.level_seed,
            distribution_mode='hard', render_mode='rgb_array'
        )
        venv = maze.wrap_venv(venv)
        info = venv.env.get_info()[0]
        # TODO: Also check that level seed matches what is stored in episode.state_vals
        assert episode.level_seed == info["level_seed"], 'level seed doesnt match created env'
        episode['start_info'] = info

    return episode
