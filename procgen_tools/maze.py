"""
Code to create, serialize, deserialize and edit mazes through their c++ state.

Source due to monte, edited by uli
https://gist.github.com/montemac/6f9f636507ec92967071bb755f37f17b

!!!!!! WARNING: This code is for the *lauro branch* of procgenAISC, it breaks on master !!!!!!
"""

import struct
import typing
from typing import Tuple, Dict, Callable, List, Optional
from dataclasses import dataclass
import os
import numpy as np
import functools
import collections
import heapq
import networkx as nx
import copy
from warnings import warn
from tqdm.auto import tqdm
from ipywidgets import GridspecLayout, Button, Layout, HBox, Output
from IPython.display import display

from procgen import ProcgenGym3Env
from procgen_tools import models

# Constants in numeric maze representation
CHEESE = 2
EMPTY = 100
BLOCKED = 51
MOUSE = 25  # UNOFFICIAL. The mouse isn't in the grid in procgen.

WORLD_DIM = 25
AGENT_PX_WIDTH = 64  # width of the agent view input image
AGENT_PX_PER_TILE = AGENT_PX_WIDTH / WORLD_DIM

HUMAN_PX_WIDTH = 512  # Not actually divisible by WORLD_DIM, so I infer there's padding on the sides
HUMAN_PX_PADDING = 6  # Just my guess for what the padding is
REAL_PX_WIDTH = HUMAN_PX_WIDTH - 2 * HUMAN_PX_PADDING
HUMAN_PX_PER_TILE = REAL_PX_WIDTH / WORLD_DIM

DEBUG = (
    False  # slows everything down by ensuring parse & serialize are inverses.
)

# Types and things


@dataclass
class StateValue:
    val: typing.Any
    idx: int


# fancy type just caused excessive checking / errors ;(
StateValues = typing.Dict[
    str, typing.Any
]  # Union[StateValue, List[StateValue], 'StateValues']]
Square = typing.Tuple[int, int]

# Parse the environment state dict

MAZE_STATE_DICT_TEMPLATE = [
    ["int", "SERIALIZE_VERSION"],
    ["string", "game_name"],
    ["int", "options.paint_vel_info"],
    ["int", "options.use_generated_assets"],
    ["int", "options.use_monochrome_assets"],
    ["int", "options.restrict_themes"],
    ["int", "options.use_backgrounds"],
    ["int", "options.center_agent"],
    ["int", "options.debug_mode"],
    ["int", "options.distribution_mode"],
    ["int", "options.use_sequential_levels"],
    ["int", "options.use_easy_jump"],
    ["int", "options.plain_assets"],
    ["int", "options.physics_mode"],
    ["int", "grid_step"],
    ["int", "level_seed_low"],
    ["int", "level_seed_high"],
    ["int", "game_type"],
    ["int", "game_n"],
    # level_seed_rand_gen.serialize(b'],
    ["int", "level_seed_rand_gen.is_seeded"],
    ["string", "level_seed_rand_gen.str"],
    # end level_seed_rand_gen.serialize(b'],
    # rand_gen.serialize(b'],
    ["int", "rand_gen.is_seeded"],
    ["string", "rand_gen.str"],
    # end rand_gen.serialize(b'],
    ["float", "step_data.reward"],
    ["int", "step_data.done"],
    ["int", "step_data.level_complete"],
    ["int", "action"],
    ["int", "timeout"],
    ["int", "current_level_seed"],
    ["int", "prev_level_seed"],
    ["int", "episodes_remaining"],
    ["int", "episode_done"],
    ["int", "last_reward_timer"],
    ["float", "last_reward"],
    ["int", "default_action"],
    ["int", "fixed_asset_seed"],
    ["int", "cur_time"],
    ["int", "is_waiting_for_step"],
    # end Game::serialize(b'],
    ["int", "grid_size"],
    # write_entities(b, entities'],
    ["int", "ents.size"],
    # for (size_t i = 0; i < ents.size(', i++)
    [
        "loop",
        "ents",
        "ents.size",
        [
            # ents[i]->serialize(b'],
            ["float", "x"],
            ["float", "y"],
            ["float", "vx"],
            ["float", "vy"],
            ["float", "rx"],
            ["float", "ry"],
            ["int", "type"],
            ["int", "image_type"],
            ["int", "image_theme"],
            ["int", "render_z"],
            ["int", "will_erase"],
            ["int", "collides_with_entities"],
            ["float", "collision_margin"],
            ["float", "rotation"],
            ["float", "vrot"],
            ["int", "is_reflected"],
            ["int", "fire_time"],
            ["int", "spawn_time"],
            ["int", "life_time"],
            ["int", "expire_time"],
            ["int", "use_abs_coords"],
            ["float", "friction"],
            ["int", "smart_step"],
            ["int", "avoids_collisions"],
            ["int", "auto_erase"],
            ["float", "alpha"],
            ["float", "health"],
            ["float", "theta"],
            ["float", "grow_rate"],
            ["float", "alpha_decay"],
            [
                "float",
                "climber_spawn_x",
            ],
        ],
    ],
    # end ents[i]->serialize(b'],
    # end write_entities
    ["int", "use_procgen_background"],
    ["int", "background_index"],
    ["float", "bg_tile_ratio"],
    ["float", "bg_pct_x"],
    ["float", "char_dim"],
    ["int", "last_move_action"],
    ["int", "move_action"],
    ["int", "special_action"],
    ["float", "mixrate"],
    ["float", "maxspeed"],
    ["float", "max_jump"],
    ["float", "action_vx"],
    ["float", "action_vy"],
    ["float", "action_vrot"],
    ["float", "center_x"],
    ["float", "center_y"],
    ["int", "random_agent_start"],
    ["int", "has_useful_vel_info"],
    ["int", "step_rand_int"],
    # asset_rand_gen.serialize(b'],
    ["int", "asset_rand_gen.is_seeded"],
    ["string", "asset_rand_gen.str"],
    # end asset_rand_gen.serialize(b'],
    ["int", "main_width"],
    ["int", "main_height"],
    ["int", "out_of_bounds_object"],
    ["float", "unit"],
    ["float", "view_dim"],
    ["float", "x_off"],
    ["float", "y_off"],
    ["float", "visibility"],
    ["float", "min_visibility"],
    # grid.serialize(b'],
    ["int", "w"],
    ["int", "h"],
    # b->write_vector_int(data'],
    ["int", "data.size"],
    # for (auto i : v) {
    ["loop", "data", "data.size", [["int", "i"]]],
    # end b->write_vector_int(data'],
    # end grid.serialize(b'],
    # end BasicAbstractGame::serialize(b'],
    ["int", "maze_dim"],
    ["int", "world_dim"],
    ["int", "END_OF_BUFFER"],
]


# version of LRU cache that returns deepcopy of cached value
def lru_cache(maxsize: int):
    def decorator(func):
        cache = collections.OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(kwargs.items()))
            try:
                value = cache.pop(key)
            except KeyError:
                value = func(*args, **kwargs)
            cache[key] = value
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return copy.deepcopy(value)

        return wrapper

    return decorator


@lru_cache(maxsize=128)
def _parse_maze_state_bytes(state_bytes: bytes, assert_=DEBUG) -> StateValues:
    # Functions to read values of different types
    def read_fixed(sb, idx, fmt):
        sz = struct.calcsize(fmt)
        # print(f'{idx} chomp {sz} got {len(sb[idx:(idx+sz)])} fmt {fmt}')
        val = struct.unpack(fmt, sb[idx : (idx + sz)])[0]
        idx += sz
        return val, idx

    read_int = lambda sb, idx: read_fixed(sb, idx, "@i")
    read_float = lambda sb, idx: read_fixed(sb, idx, "@f")

    def read_string(sb, idx):
        sz, idx = read_int(sb, idx)
        val = sb[idx : (idx + sz)].decode("ascii")
        idx += sz
        return val, idx

    # Function to process a value definition and return a value (called recursively for loops)
    def parse_value(vals, val_def, idx):
        typ = val_def[0]
        name = val_def[1]
        # print((typ, name))
        if typ == "int":
            val, idx = read_int(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "float":
            val, idx = read_float(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "string":
            val, idx = read_string(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == "loop":
            len_name = val_def[2]
            loop_val_defs = val_def[3]
            loop_len = vals[len_name].val
            vals[name] = []
            for _ in range(loop_len):
                vals_this = {}
                for loop_val_def in loop_val_defs:
                    idx = parse_value(vals_this, loop_val_def, idx)
                vals[name].append(vals_this)
        return idx

    # Dict to hold values
    vals = {}

    # Loop over list of value defs, parsing each
    idx = 0
    for val_def in MAZE_STATE_DICT_TEMPLATE:
        idx = parse_value(vals, val_def, idx)

    if assert_:
        assert (
            _serialize_maze_state(vals, assert_=False) == state_bytes
        ), "serialize(deserialize(state_bytes)) != state_bytes"
    return vals


def _serialize_maze_state(state_vals: StateValues, assert_=DEBUG) -> bytes:
    # Serialize any value to a bytes object
    def serialize_val(val):
        if isinstance(val, StateValue):
            val = val.val
        if isinstance(val, int):
            return struct.pack("@i", val)
        elif isinstance(val, float):
            return struct.pack("@f", val)
        elif isinstance(val, str):
            return serialize_val(len(val)) + val.encode("ascii")
        else:
            raise ValueError(f"type(val)={type(val)} not handled")

    # Flatten the nested values into a single list of primitives
    def flatten_vals(vals, flat_list=[]):
        if isinstance(vals, dict):
            for val in vals.values():
                flatten_vals(val, flat_list)
        elif isinstance(vals, list):
            for val in vals:
                flatten_vals(val, flat_list)
        else:
            flat_list.append(vals)

    # Flatten the values, then serialize
    flat_vals = []
    flatten_vals(state_vals, flat_vals)

    state_bytes = b"".join([serialize_val(val) for val in flat_vals])

    if assert_:
        assert (
            _parse_maze_state_bytes(state_bytes, assert_=False) == state_vals
        ), "deserialize(serialize(state_vals)) != state_vals"
    return state_bytes


# Backwards compatability with data_utils
def get_grid(state_vals: StateValues):
    "Get the grid from state_vals"
    world_dim = state_vals["world_dim"].val
    grid_vals = np.array([dd["i"].val for dd in state_vals["data"]]).reshape(
        world_dim, world_dim
    )
    return grid_vals


def get_mouse_pos_sv(state_vals: StateValues) -> Square:
    """Get the mouse position from state_vals"""
    ents = state_vals["ents"][0]
    return int(ents["y"].val), int(ents["x"].val)


# EnvState functions
class EnvState:
    def __init__(self, state_bytes: bytes):
        self.state_bytes = state_bytes

    @property
    def state_vals(self):
        return _parse_maze_state_bytes(self.state_bytes)

    @property
    def world_dim(self):
        return self.state_vals["world_dim"].val

    def full_grid(self, with_mouse=True):
        "Get numpy (world_dim, world_dim) grid of the maze. Includes the mouse by default."
        world_dim = self.world_dim
        grid = np.array(
            [dd["i"].val for dd in self.state_vals["data"]]
        ).reshape(world_dim, world_dim)
        if with_mouse:
            grid[self.mouse_pos] = MOUSE

        return grid

    def inner_grid(self, with_mouse=True):
        "Get inner grid of the maze. Includes the mouse by default."
        return inner_grid(self.full_grid(with_mouse=with_mouse))

    @property
    def mouse_pos(self) -> Tuple[int, int]:
        "Get (x, y) position of mouse in grid."
        ents = self.state_vals["ents"][0]
        # flipped turns out to be oriented right for grid.
        return int(ents["y"].val), int(ents["x"].val)

    def set_mouse_pos(self, x: int, y: int):
        """
        Set the mouse position in the maze state bytes. Much more optimized than parsing and serializing the whole state.
        *WARNING*: This uses *outer coordinates*, not inner.
        """
        # FIXME(slow): grabbing state_vals requires a call to parse the state bytes.
        state_vals = self.state_vals
        state_vals["ents"][0]["x"].val = float(y) + 0.5
        state_vals["ents"][0]["y"].val = float(x) + 0.5
        self.state_bytes = _serialize_maze_state(state_vals)

    def set_grid(self, grid: np.ndarray, pad=False):
        """
        Set the grid of the maze.
        """
        if pad:
            grid = outer_grid(grid, assert_=False)
        assert grid.shape == (self.world_dim, self.world_dim)

        state_vals = self.state_vals
        grid = grid.copy()  # might need to remove mouse if in grid
        if (grid == MOUSE).sum() > 0:
            x, y = get_mouse_pos(grid)

            state_vals["ents"][0]["x"].val = (
                float(y) + 0.5
            )  # flip again to get back to original orientation
            state_vals["ents"][0]["y"].val = float(x) + 0.5

            grid[x, y] = EMPTY

        world_dim = state_vals["world_dim"].val
        assert grid.shape == (world_dim, world_dim)
        for i, dd in enumerate(state_vals["data"]):
            dd["i"].val = int(grid.ravel()[i])

        self.state_bytes = _serialize_maze_state(state_vals)


# ============== Grid helpers ==============


def state_from_venv(venv, idx: int = 0) -> EnvState:
    """
    Get the maze state from the venv.
    """
    state_bytes_list = venv.env.callmethod("get_state")
    return EnvState(state_bytes_list[idx])


def get_cheese_pos(grid: np.ndarray, flip_y: bool = False) -> Square:
    "Get (row, col) position of the cheese in the grid. Note that the numpy grid is flipped along the y-axis, relative to rendered images."
    num_cheeses = (grid == CHEESE).sum()
    if num_cheeses == 0:
        return None
    row, col = np.where(grid == CHEESE)
    row, col = row[0], col[0]
    return ((WORLD_DIM - 1) - row if flip_y else row), col


def remove_cheese(venv, idx: int = 0):
    """
    Remove the cheese from the grid, modifying venv in-place.
    """
    state_bytes_list = venv.env.callmethod("get_state")
    state = state_from_venv(venv, idx)

    # TODO(uli): The multiple sources of truth here suck. Ideally one object linked to venv auto-updates(?)
    grid = state.full_grid()
    grid[grid == CHEESE] = EMPTY
    state.set_grid(grid)
    state_bytes_list[idx] = state.state_bytes
    venv.env.callmethod("set_state", state_bytes_list)
    return venv


def move_cheese(venv, new_pos: Tuple[int, int], idx: int = 0):
    """
    Move the cheese to the given position, modifying venv in-place.
    """
    assert (
        0 <= new_pos[0] < WORLD_DIM and 0 <= new_pos[1] < WORLD_DIM
    ), f"new_pos={new_pos} out of bounds"
    state_bytes_list = venv.env.callmethod("get_state")
    state = state_from_venv(venv, idx)

    grid = state.full_grid()
    grid[grid == CHEESE] = EMPTY
    grid[new_pos] = CHEESE
    state.set_grid(grid)
    state_bytes_list[idx] = state.state_bytes
    venv.env.callmethod("set_state", state_bytes_list)
    return venv


def remove_all_cheese(venv):
    """
    Remove the cheese from each env in venv, inplace.
    """
    state_bytes_list = venv.env.callmethod("get_state")
    states = [EnvState(state_bytes) for state_bytes in state_bytes_list]
    for i, s in enumerate(states):
        g = s.full_grid()
        g[g == CHEESE] = EMPTY
        s.set_grid(g)
        state_bytes_list[i] = s.state_bytes
    venv.env.callmethod("set_state", state_bytes_list)
    return venv


def get_mouse_pos(
    grid: np.ndarray, flip_y: bool = False
) -> typing.Tuple[int, int]:
    "Get (x, y) position of the mouse in the grid"
    num_mouses = (grid == MOUSE).sum()
    assert num_mouses == 1, f"{num_mouses} mice, should be 1"
    row, col = np.where(grid == MOUSE)
    row, col = row[0], col[0]
    return ((WORLD_DIM - 1) - row if flip_y else row), col


def inner_grid(grid: np.ndarray, assert_=True) -> np.ndarray:
    """
    Get the inside of the maze, ie. the stuff within the outermost walls.
    inner_grid(inner_grid(x)) = inner_grid(x) for all x.
    """
    # Find the amount of padding on the maze, where padding is BLOCKED
    # Use numpy to test if each square is BLOCKED
    # If it is, then it's part of the padding
    bl = 0
    # Check if the top, bottom, left, and right are all blocked
    while (
        (grid[bl, :] == BLOCKED).all()
        and (grid[-bl - 1, :] == BLOCKED).all()
        and (grid[:, bl] == BLOCKED).all()
        and (grid[:, -bl - 1] == BLOCKED).all()
    ):
        bl += 1

    return (
        grid[bl:-bl, bl:-bl] if bl > 0 else grid
    )  # if bl == 0, then we don't need to do anything


def outer_grid(grid: np.ndarray, assert_=True) -> np.ndarray:
    """
    The inverse of inner_grid(). Could also be called "pad_grid".
    """
    bl = (WORLD_DIM - len(grid)) // 2
    outer = np.pad(grid, bl, "constant", constant_values=BLOCKED)
    if assert_:
        assert (inner_grid(outer, assert_=False) == grid).all()
    return outer


def _get_neighbors(x, y):
    "Get the neighbors of (x, y) in the grid"
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _ingrid(grid: np.ndarray, n):
    "Is (x, y) in the grid?"
    return 0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1]


def get_empty_neighbors(grid: np.ndarray, x, y):
    "Get the empty neighbors of (x, y) in the grid"
    return [
        n
        for n in _get_neighbors(x, y)
        if _ingrid(grid, n) and grid[n] != BLOCKED
    ]


def _euclidian_dist_to_cheese(grid: np.ndarray, coord: Tuple) -> float:
    """
    Euclidian distance from (x,y) to the cheese. default heuristic for A*
    """
    mx, my = coord
    cx, cy = get_cheese_pos(grid)
    return np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)


def shortest_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    stop_condition: Callable[[np.ndarray, Tuple], bool] = None,  # type: ignore
    heuristic: Callable[[np.ndarray, Tuple], float] = None,  # type: ignore
) -> Tuple[Dict[Square, int], Dict[Square, Square], Dict[str, typing.Any]]:
    """
    Compute the number of moves for the mouse to get the cheese (using A*)
    - default stop_condition is finding the cheese
    - default heuristic is euclidian distance to cheese
    """
    # assert (grid==MOUSE).sum() == 1, f'grid has {(grid==MOUSE).sum()} mice' # relaxed by start param
    # assert (
    #     grid == CHEESE
    # ).sum() == 1, f"grid has {(grid==CHEESE).sum()} cheeses"

    grid = inner_grid(grid).copy()

    if heuristic is None and stop_condition is None:
        heuristic = _euclidian_dist_to_cheese
    if stop_condition is None:
        stop_condition = lambda g, c: g[c] == CHEESE
    if heuristic is None:
        heuristic = lambda *_: 1  # disable heuristic, none given

    # A* search
    frontier = []
    heapq.heappush(frontier, (0, start))

    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    extra = {"last_square": None}

    while len(frontier) > 0:
        current = heapq.heappop(frontier)[1]
        if stop_condition(grid, current):
            extra["last_square"] = current
            break

        for next in get_empty_neighbors(grid, *current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(grid, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    return cost_so_far, came_from, extra


def reconstruct_path(
    came_from: Dict[Square, Square], end: Square
) -> List[Square]:
    path = [end]
    while path[-1] in came_from:
        path += [came_from[path[-1]]]

    path = path[::-1][1:]  # reverse, remove None
    return path


def is_tree(grid: np.ndarray, debug=False) -> bool:
    """
    Is there exactly one path between any two empty squares in the maze?
    (Also known as, is the set of empty squares a spanning tree)
    """
    grid = inner_grid(grid).copy()
    grid[grid == CHEESE] = EMPTY

    visited_edges = set()
    visited_nodes = set()
    stack = [(0, 0)]
    while stack:
        node = stack.pop()
        if node in visited_nodes:
            if debug:
                print(f"{node} already visited, a cycle!")
            return False
        visited_nodes.add(node)
        for neighbor in get_empty_neighbors(grid, *node):
            edge = (node, neighbor)
            if edge not in visited_edges and edge[::-1] not in visited_edges:
                stack.append(neighbor)
                visited_edges.add(edge)

    # There were no cycles, if we visited all the nodes, then it's a tree
    visited_all_nodes = len(visited_nodes) == (grid == EMPTY).sum()
    if debug:
        print(
            f"visited {len(visited_nodes)} out of"
            f" {(grid == EMPTY).sum()} nodes"
        )
    return visited_all_nodes


def on_distribution(
    grid: np.ndarray, p: Callable = print, full: bool = False
) -> bool:
    """
    Is the given *maze* something that could have been generated during training?
    If full is passed the maze must include a single mouse and cheese.
    """

    # Make a copy of the inner grid without the cheese and mouse
    g = inner_grid(grid).copy()
    if full:
        if (g == MOUSE).sum() != 1:
            p(f"grid has {(g == MOUSE).sum()} mice")
            return False
        if (g == CHEESE).sum() != 1:
            p(f"grid has {(g == CHEESE).sum()} cheeses")
            return False

    # For the rest of the checks, we don't care about the mouse or cheese
    g[g == CHEESE] = EMPTY
    g[g == MOUSE] = EMPTY

    # Assert invariants
    if not (g[1::2, 1::2] == BLOCKED).all():
        p("Squares where row,col are both odd must always be blocked")
        return False
    if not (g[0::2, 0::2] == EMPTY).all():
        p("Squares where row,col are both even must always be empty")
        return False
    if not is_tree(g):
        p("There must be exactly one path between any two empty squares")
        return False

    return True


def venv_from_grid(grid: np.ndarray) -> ProcgenGym3Env:  #
    "Get a venv with the given inner grid"
    grid_copy = grid.copy()
    n_mice = (grid_copy == MOUSE).sum()
    if n_mice == 0:
        blocked_idx = 0  # Assumes mouse should be placed on diagonal
        while grid_copy[blocked_idx, blocked_idx] == BLOCKED:
            blocked_idx += 1
        assert (
            grid_copy[blocked_idx, blocked_idx] == EMPTY
        ), f"grid[{blocked_idx},{blocked_idx}] is not empty"

        grid_copy[blocked_idx, blocked_idx] = MOUSE
    assert (grid_copy == MOUSE).sum() == 1, "grid has {} mice".format(
        (grid_copy == MOUSE).sum()
    )

    venv = create_venv(num=1, num_levels=1, start_level=0)
    state = state_from_venv(venv, idx=0)
    state.set_grid(grid_copy, pad=True)
    venv.env.callmethod("set_state", [state.state_bytes])
    return venv


def get_filled_venv(fill_type: int = EMPTY) -> ProcgenGym3Env:
    """Get a venv with a grid filled with fill_type (either EMPTY or CHEESE; BLOCKED throws an error for some reason)."""
    assert fill_type in (CHEESE, EMPTY), "fill_type must be EMPTY or CHEESE"
    grid = get_full_grid_from_seed(seed=0)
    mouse_pos = get_mouse_pos(grid)
    for block_type in (BLOCKED, CHEESE, EMPTY):
        grid[grid == block_type] = fill_type
    grid[mouse_pos] = MOUSE  # Don't overwrite the mouse
    return venv_from_grid(grid_copy=grid)


def get_padding(grid: np.ndarray) -> int:
    """Return the padding of the (inner) grid, i.e. the number of walls around the maze."""
    return (WORLD_DIM - grid.shape[0]) // 2


def inside_inner_grid(grid: np.ndarray, row: int, col: int) -> bool:
    """Return True if the given row, col is inside the inner grid."""
    padding = get_padding(grid)
    return (
        padding <= row < grid.shape[0] - padding
        and padding <= col < grid.shape[1] - padding
    )


def render_inner_grid(grid: np.ndarray):
    """Extract the human-sensible view given grid, assumed to be an inner_grid. Return the human view."""
    venv = venv_from_grid(grid)
    human_view = venv.env.get_info()[0]["rgb"]

    # Cut out the padding from the view. The padding is the walls around the maze.
    padding = get_padding(grid)
    rescale = human_view.shape[0] / WORLD_DIM

    return (
        human_view[
            int(padding * rescale) : int(-padding * rescale),
            int(padding * rescale) : int(-padding * rescale),
        ]
        if padding > 0
        else human_view
    )


def render_outer_grid(grid: np.ndarray):
    """Extract the human-sensible view given grid, assumed to be an outer_grid. Return the human view."""
    venv = venv_from_grid(grid)
    return venv.env.get_info()[0]["rgb"]


def grid_editor(
    grid: np.ndarray,
    node_radius="8px",
    delay=0.01,
    callback=None,
    check_on_dist=True,
    show_full: bool = False,
):
    import time

    CELL_TO_COLOR = {
        EMPTY: "#D9D9D6",
        BLOCKED: "#A47449",
        CHEESE: "#EAAA00",
        MOUSE: "#393D47",
    }
    CELL_TO_CHAR = {
        EMPTY: "Empty",
        BLOCKED: "Blocked",
        CHEESE: "🧀",
        MOUSE: "🐭",
    }

    num_mice = (grid == MOUSE).sum()
    assert num_mice in (0, 1), f"num_mice {num_mice}"

    # will maintain a pointer into grid
    g = grid if show_full else inner_grid(grid)
    rows, cols = g.shape
    wgrid = GridspecLayout(rows, cols, width="min-content")

    output = Output()

    def button_clicked(b: Button):
        i, j = getattr(b, "coord")
        if (g == CHEESE).sum() == 0:
            g[i, j] = CHEESE
        elif (g == MOUSE).sum() == 0 and num_mice > 0:
            g[i, j] = MOUSE
        else:
            g[i, j] = {
                EMPTY: BLOCKED,
                BLOCKED: EMPTY,
                CHEESE: EMPTY,
                MOUSE: EMPTY,
            }[g[i, j]]

        b.style.button_color = CELL_TO_COLOR[g[i, j]]  # type: ignore
        b.tooltip = CELL_TO_CHAR[g[i, j]]
        with output:
            output.clear_output()
            if check_on_dist:
                on_distribution(g)
            if callback is not None:
                callback(grid)

    for i in range(rows):
        for j in range(cols):
            b = Button(
                layout=Layout(
                    padding=node_radius,
                    width="0px",
                    height="0px",
                    margin="0px",
                )
            )
            b.tooltip = CELL_TO_CHAR[g[i, j]]
            setattr(b, "coord", (i, j))  # monkey patch to pass coords
            b.style.button_color = CELL_TO_COLOR[g[i, j]]  # type: ignore
            b.on_click(button_clicked)
            # flip the grid so it's oriented correctly, like origin=lower in matplotlib.
            wgrid[rows - i - 1, j] = b
        time.sleep(delay)
    return HBox([wgrid, output])


def venv_editors(
    venv: ProcgenGym3Env,
    check_on_dist: bool = True,
    env_nums=None,
    callback: Callable = None,
    **kwargs,
):
    """
    Run maze_editor on a venv, possibly with multiple mazes. Keep everything in sync.
    """
    if env_nums is None:
        env_nums = range(venv.num_envs)
    # TODO: Hook venv so after reset it maintains the edited version

    def make_cb(i: int):
        def _cb(gridm: np.ndarray):
            if (not check_on_dist) or on_distribution(
                gridm, p=lambda *_: None
            ):
                # print('Saving state to venv')
                env_states[i].set_grid(gridm)
                # FIXME: If the maze is edited externally this will break (state_vals_list is constant)
                venv.env.callmethod(
                    "set_state", [vs.state_bytes for vs in env_states]
                )
                if callback is not None:
                    callback(gridm)

        return _cb

    env_states = [
        EnvState(sb) for i, sb in enumerate(venv.env.callmethod("get_state"))
    ]
    editors = [
        grid_editor(
            vs.full_grid(),
            callback=make_cb(i),
            check_on_dist=check_on_dist,
            **kwargs,
        )
        for i, vs in enumerate(env_states)
        if i in env_nums
    ]
    return editors


def _vbox_hr(elements):
    from ipywidgets import VBox, HTML

    els = []
    for e in elements:
        els.append(e)
        els.append(HTML("<hr>"))
    return VBox(els)


def venv_editor(venv, **kwargs):
    "Wraps `venv_editors` in a VBox with a horizontal rule between each maze."
    return _vbox_hr(venv_editors(venv, **kwargs))


# ================ Maze-as-graph tools ===================
# TODO: put all this inside EnvState object, it's a horrible mess!


def maze_grid_to_graph(inner_grid):
    """Convert a provided maze inner grid to a networkX graph object"""

    def nodes_where(cond):
        return [(r, c) for r, c in zip(*np.where(cond))]

    # Create edges: each node may have an edge up, down, left or right, check
    # each direction for all nodes at the same time
    edges = []
    for dirs, g0, g1 in [
        ["RL", inner_grid[:, :-1], inner_grid[:, 1:]],
        ["UD", inner_grid[:-1, :], inner_grid[1:, :]],
    ]:
        # Find squares that are open in both g0 and g1, and add an edge
        node0s = nodes_where((g0 != BLOCKED) & (g1 != BLOCKED))
        node1s = [
            (r, c + 1) if dirs == "RL" else (r + 1, c) for r, c in node0s
        ]
        edges.extend([(n0, n1) for n0, n1 in zip(node0s, node1s)])
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def grid_graph_has_decision_square(inner_grid, graph):
    cheese_node = get_cheese_pos(inner_grid)
    corner_node = (inner_grid.shape[0] - 1, inner_grid.shape[1] - 1)
    pth = nx.shortest_path(graph, (0, 0), corner_node)
    return not cheese_node in pth


def get_path_to_cheese(inner_grid, graph, start_node=(0, 0)):
    cheese_node = get_cheese_pos(inner_grid)
    return nx.shortest_path(graph, start_node, cheese_node)


def get_path_to_corner(inner_grid, graph, start_node=(0, 0)):
    corner_node = (inner_grid.shape[0] - 1, inner_grid.shape[1] - 1)
    return nx.shortest_path(graph, start_node, corner_node)


def pathfind(grid: np.ndarray, start, end):
    _, came_from, extra = shortest_path(
        grid, start, stop_condition=lambda _, sq: sq == end
    )
    return reconstruct_path(came_from, extra["last_square"])


def geometric_probability_path(
    start: Tuple[int, int], end: Tuple[int, int], vf: Dict
) -> float:
    """Returns the geometric mean of `vf`'s probability of the path from
    `start` to `end` in the maze. If the path contains the cheese, the
    cheese is ignored in the mean."""
    for coord in (start, end):
        assert (coord[i] >= 0 and coord[i] < MAZE_SIZE for i in (0, 1))
    if start == end:
        idx: int = vf["legal_mouse_positions"].index(start)
        return vf["probs"][idx][4]  # The no-op probability

    path = pathfind(vf["grid"], start, end)
    cheese_loc: Tuple[int, int] = get_cheese_pos(vf["grid"])

    zipped_list = zip(vf["legal_mouse_positions"], vf["probs"])
    prob_dict: Dict[Tuple[int, int], float] = dict(zipped_list)

    sum_log_prob: float = 0.0
    for idx, coord in enumerate(path[:-1]):
        if coord == cheese_loc:
            continue
        action: str = None

        # Get the action by looking at the next coord
        for key, delta in models.MAZE_ACTION_DELTAS.items():
            if (
                coord[0] + delta[0] == path[idx + 1][0]
                and coord[1] + delta[1] == path[idx + 1][1]
            ):
                action = key
                break

        if action is None:
            raise ValueError(
                "Invalid path; cannot find action which leads to next"
                " coordinate"
            )

        # Get the probability of the action
        action_index: int = list(models.MAZE_ACTION_INDICES.keys()).index(
            action
        )
        sum_log_prob += np.log(prob_dict[coord][action_index])

    divisor: int = len(path) - (2 if cheese_loc in path[:-1] else 1)
    geom_mean_prob: float = np.exp(sum_log_prob / divisor)
    return geom_mean_prob


def deltas_from(grid: np.ndarray, sq):
    """Returns the deltas between the decision square and the cheese, and the decision square and the top-right corner."""
    path_to_cheese = maze.pathfind(grid, sq, maze.get_cheese_pos(grid))
    path_to_top_right = maze.pathfind(
        grid, sq, (grid.shape[0] - 1, grid.shape[1] - 1)
    )
    delta_cheese = (path_to_cheese[1][0] - sq[0], path_to_cheese[1][1] - sq[1])
    delta_tr = (
        path_to_top_right[1][0] - sq[0],
        path_to_top_right[1][1] - sq[1],
    )
    return delta_cheese, delta_tr


def get_decision_square_from_grid_graph(inner_grid, graph):
    path_to_cheese = get_path_to_cheese(inner_grid, graph)
    path_to_corner = get_path_to_corner(inner_grid, graph)
    for ii, cheese_path_node in enumerate(path_to_cheese):
        if ii >= len(path_to_corner):
            return cheese_path_node
        if cheese_path_node != path_to_corner[ii]:
            return path_to_cheese[ii - 1]


def maze_has_decision_square(states_bytes):
    maze_env_state = EnvState(states_bytes)
    inner_grid = maze_env_state.inner_grid()
    grid_graph = maze_grid_to_graph(inner_grid)
    return grid_graph_has_decision_square(inner_grid, grid_graph)


def get_decision_square_from_maze_state(state):
    inner_grid = state.inner_grid()
    grid_graph = maze_grid_to_graph(inner_grid)
    return get_decision_square_from_grid_graph(inner_grid, grid_graph)


def get_node_value_at_offset(outer_grid, node, offset):
    r, c = [n + off for n, off in zip(node, offset)]
    if (np.array([r, c]) >= outer_grid.shape).any():
        return BLOCKED
    return outer_grid[r, c]


NODE_TYPES = ["wall", "unconn", "end", "path", "branch2", "branch3"]


def get_node_type_by_world_loc(states_bytes, world_node):
    """Return node type of the square referred to by world_node
    (world_node should be in world coords, not inner coords,
    order is (row, col)).

    Returns a tuple of (node_type, lrdu_open), where possible node types
    are: wall, unconn, end (only one open neighbour), path (two open),
    branch2 (3 open), branch3 (4 open).  Second return enumerates
    the possible (closed, open) states of all 4 neighbours, so
    16 possibilities.  Returned as a bool array, even for walls."""
    maze_env_state = EnvState(states_bytes)
    outer_grid = maze_env_state.full_grid()
    node_value = outer_grid[world_node]
    lrdu_open = (
        np.array(
            [
                get_node_value_at_offset(outer_grid, world_node, (0, -1)),
                get_node_value_at_offset(outer_grid, world_node, (0, 1)),
                get_node_value_at_offset(outer_grid, world_node, (-1, 0)),
                get_node_value_at_offset(outer_grid, world_node, (1, 0)),
            ]
        )
        != BLOCKED
    )
    if node_value == BLOCKED:
        node_type = "wall"
    else:
        node_type = NODE_TYPES[1:][lrdu_open.sum()]
    return node_type, lrdu_open


def get_object_pos_in_grid(grid, obj_value):
    return np.argwhere(grid == obj_value)[0]


def get_legal_mouse_positions(grid: np.ndarray):
    """Return a list of legal mouse positions in the grid, returned as a list of tuples."""
    return [
        (x, y)
        for x in range(grid.shape[0])
        for y in range(grid.shape[1])
        if grid[x, y] == EMPTY
    ]


def get_object_pos_from_seq_of_states(state_bytes_seq, obj_value):
    """Extract object positions from a sequence of state_bytes, returning
    as a numpy array of shape (len(sequance), 2).  Note that the first
    column is y-position to stay consistent with row/col matrix ordering
    conventions."""
    mouse_pos = np.zeros((len(state_bytes_seq), 2), dtype=int)
    for ii, state_bytes in enumerate(state_bytes_seq):
        env_state = EnvState(state_bytes)
        y, x = np.argwhere(env_state.full_grid() == obj_value)[0]
        mouse_pos[ii, :] = np.array([y, x])
    return mouse_pos


def get_mouse_pos_from_seq_of_states(state_bytes_seq):
    """Extract mouse positions from a sequence of state_bytes, returning
    as a numpy array of shape (len(sequance), 2).  Note that the first
    column is y-position to stay consistent with row/col matrix ordering
    conventions."""
    get_object_pos_from_seq_of_states(state_bytes_seq, MOUSE)


def get_cheese_pos_from_seq_of_states(state_bytes_seq):
    """Extract cheese positions from a sequence of state_bytes, returning
    as a numpy array of shape (len(sequance), 2).  Note that the first
    column is y-position to stay consistent with row/col matrix ordering
    conventions."""
    get_object_pos_from_seq_of_states(state_bytes_seq, CHEESE)


def get_envstate_from_seed(seed: int):
    seed_env = create_venv(num=1, start_level=seed, num_levels=1)
    return state_from_venv(venv=seed_env, idx=0)


def get_full_grid_from_seed(seed: int):
    state = get_envstate_from_seed(seed)
    return state.full_grid()


def get_inner_grid_from_seed(seed: int):
    state = get_envstate_from_seed(seed)
    return state.inner_grid()


def rand_seed_with_size(min_size: int = 3, max_size: int = WORLD_DIM) -> int:
    """Generate a random seed with a maze of size between min_size and
    max_size. Rejection sampling is used to ensure that the maze size is
    in the desired range."""
    assert (
        3 <= min_size <= max_size <= WORLD_DIM
    ), f"Invalid size range. Must be 3 <= min_size <= max_size <= {WORLD_DIM}."

    max_seed = 100000
    while True:
        seed = np.random.randint(0, max_seed)
        if (
            get_inner_grid_from_seed(seed=seed).shape[0] <= max_size
            and get_inner_grid_from_seed(seed=seed).shape[0] >= min_size
        ):
            return seed


def get_cheese_pos_from_seed(seed: int, flip_y: bool = False):
    """Get the cheese position from a maze seed."""
    grid = get_full_grid_from_seed(seed)
    return get_cheese_pos(grid, flip_y=flip_y)


def get_mouse_pos_from_seed(seed: int, flip_y: bool = False):
    """Get the mouse position from a maze seed."""
    grid = get_full_grid_from_seed(seed)
    return get_mouse_pos(grid, flip_y=flip_y)


def get_mazes_with_mouse_at_location(
    cheese_location: Tuple[int, int], num_mazes: int = 5, skip_seed: int = -1
):
    """Generate a list of maze seeds with cheese at the specified location."""
    assert (
        len(cheese_location) == 2
    ), "Cheese location must be a tuple of length 2."
    assert (
        0 <= coord < WORLD_DIM for coord in cheese_location
    ), "Cheese location must be within the maze."

    seeds = []
    seed = 0
    while len(seeds) < num_mazes:
        if seed != skip_seed and (
            get_cheese_pos_from_seed(seed) == cheese_location
        ):
            seeds.append(seed)
        seed += 1
    return seeds


def generate_mazes_with_cheese_at_location(
    cheese_location: Tuple[int, int], num_mazes: int = 50, skip_seed: int = -1
):
    """Generate the first num_mazes seeds which have an empty/cheese square at cheese_location, except the mazes are modified to instead have cheese at cheese_location. Returns a list of full grids."""
    assert (
        len(cheese_location) == 2
    ), "Cheese location must be a tuple of length 2."
    assert (
        0 <= coord < WORLD_DIM for coord in cheese_location
    ), "Cheese location must be within the maze."

    seeds, grids = [], []
    seed = 0
    while len(grids) < num_mazes:
        if seed != skip_seed:
            grid = get_full_grid_from_seed(seed)
            if (
                grid[cheese_location] == EMPTY
                or grid[cheese_location] == CHEESE
            ):
                old_cheese = get_cheese_pos(grid)
                grid[old_cheese] = EMPTY
                grid[cheese_location] = CHEESE
                grids.append(
                    inner_grid(grid)
                )  # venv_from_grid expects inner grid
                seeds.append(seed)
        seed += 1
    return seeds, grids


# ================ Venv Wrappers ===================

from .procgen_wrappers import (
    TransposeFrame,
    ScaledFloatFrame,
    VecExtractDictObs,
)
from gym3 import ToBaselinesVecEnv


def wrap_venv(venv) -> ToBaselinesVecEnv:
    "Wrap a vectorized env, making it compatible with the gym apis, transposing, scaling, etc."
    # TODO: Move this to another file (same thing is used for coinrun)

    venv = ToBaselinesVecEnv(venv)  # gym3 env to gym env converter
    venv = VecExtractDictObs(venv, "rgb")

    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv  # type: ignore - ToBaselinesVecEnv gives best type annotations


from procgen import ProcgenGym3Env


def create_venv(
    num: int, start_level: int, num_levels: int, num_threads: int = 1
):
    """
    Create a wrapped venv. See https://github.com/openai/procgen#environment-options for params

    num=1 - The number of parallel environments to create in the vectorized env.

    num_levels=0 - The number of unique levels that can be generated. Set to 0 to use unlimited levels.

    start_level=0 - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
    """
    venv = ProcgenGym3Env(
        num=num,
        env_name="maze",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="hard",
        num_threads=num_threads,
        render_mode="rgb_array",
        # rand_region=5,
    )
    venv = wrap_venv(venv)
    return venv


def copy_venv(venv, idx: int):
    "Return a copy of venv number idx. WARNING: After level is finished, the copy will be reset."
    sb = venv.env.callmethod("get_state")[idx]
    env = create_venv(num=1, start_level=0, num_levels=1)
    env.env.callmethod("set_state", [sb])
    return env


def copy_venvs(venv_all):
    "Return a copy of all the venvs in venv_all. WARNING: After level is finished, the copy will be reset."
    sb = venv_all.env.callmethod("get_state")
    env = create_venv(num=len(sb), start_level=0, num_levels=1)
    env.env.callmethod("set_state", sb)
    return env


import pickle as pkl


def load_venv(filename: str):
    "Load a venv from a file."
    state_bytes = pkl.load(open(filename, "rb"))
    num_envs = len(state_bytes)
    venv = create_venv(num=num_envs, start_level=0, num_levels=1)
    venv.env.callmethod("set_state", state_bytes)
    return venv


def save_venv(venv: ProcgenGym3Env, filename: str):
    "Save all envs in venv to a file."
    state_bytes = venv.env.callmethod("get_state")
    pkl.dump(state_bytes, open(filename, "wb"))


def remove_cheese_from_state(state):
    grid = state.full_grid()
    grid[grid == CHEESE] = EMPTY
    state.set_grid(grid)


def move_cheese_in_state(state, new_cheese_pos):
    grid = state.full_grid()
    grid[grid == CHEESE] = EMPTY
    grid[new_cheese_pos] = CHEESE
    state.set_grid(grid)


def get_custom_venv_pair(seed: int, num_envs=2):
    """Allow the user to edit num_envs levels from a seed. Return a venv containing both environments."""
    venv = create_venv(num=num_envs, start_level=seed, num_levels=1)
    display(HBox(venv_editor(venv, check_on_dist=False)))
    return venv


def get_cheese_venv_pair(
    seed: int, has_cheese_tup: Tuple[bool, bool] = (True, False)
):
    "Return a venv of 2 environments from a seed, with cheese in the first environment if has_cheese_tup[0] and in the second environment if has_cheese_tup[1]."
    venv = create_venv(num=2, start_level=seed, num_levels=1)

    for idx in range(2):
        if has_cheese_tup[idx]:
            continue  # Skip if we want cheese in this environment
        remove_cheese(venv, idx=idx)

    return venv


def make_top_right_path(state: EnvState) -> EnvState:
    """Make a new EnvState with an L-shaped path from the original
    top-right corner to the absolute top-right corner (in the outer
    grid)."""
    # Get the original top-right free space
    inner_grid = state.inner_grid()
    padding = get_padding(inner_grid)
    if padding == 0:
        return state  # No padding, so no path to make

    orig_row = (
        WORLD_DIM - 1
    ) - padding  # note that numpy representation is flipped, so it's the "bottom left"
    orig_col = (WORLD_DIM - 1) - padding

    # Sanity check math:
    # padding = 1, WORLD_DIM = 5, orig_row = 3, orig_col = 3

    # Get the top-right corner location using padding
    absolute_row, absolute_col = (
        WORLD_DIM - 1,
        WORLD_DIM - 1,
    )  # (row, col) -- note that numpy representation is flipped, so it's the "bottom left"

    # Make the path
    outer_grid = state.full_grid()
    outer_grid[orig_row + 1 : absolute_row + 1, orig_col] = EMPTY
    outer_grid[absolute_row, orig_col + 1 : absolute_col + 1] = EMPTY

    # Make the new state
    new_state = EnvState(state.state_bytes)
    new_state.set_grid(outer_grid)
    return new_state


def get_top_right_venv_pair(seed: int) -> ProcgenGym3Env:
    """Return a venv of 2 environments from a seed, with the first
    environment having an L-shaped path to original top-right corner to
    the absolute top-right corner, and the second environment normally
    generated from the seed."""
    venv = create_venv(num=2, start_level=seed, num_levels=1)

    # Make the first environment have a path
    state_bytes = venv.env.callmethod("get_state")[0]
    state = EnvState(state_bytes)
    new_state: EnvState = make_top_right_path(state)
    sb_list = [new_state.state_bytes, state_bytes]
    venv.env.callmethod("set_state", sb_list)

    return venv


def get_random_obs_opts(
    num_obs: int = 1,
    on_training: bool = True,
    rand_region: int = 5,
    spawn_cheese: bool = True,
    maze_dim: Optional[int] = None,
    mouse_pos_inner: Optional[Tuple[int, int]] = None,
    cheese_pos_inner: Optional[Tuple[int, int]] = None,
    mouse_pos_outer: Optional[Tuple[int, int]] = None,
    cheese_pos_outer: Optional[Tuple[int, int]] = None,
    must_be_dec_square: bool = False,
    start_level: int = 0,
    return_metadata: bool = False,
    random_seed: Optional[int] = None,
    deterministic_levels: bool = False,
    show_pbar: bool = False,
):
    """Get num_obs observations from the maze environment. If on_training is True, then the observation is
    from a training level where the cheese is in the top-right rand_region corner.
    Can also optionally filter/force the resulting samples in various ways:
    - maze_dim ensures a constant maze inner grid size
    - mouse_pos_inner forces the mouse to a specific inner_grid location (skipping levels that aren't open on this location)
    - cheese_pos_inner forces the cheese to a specific inner_grid location (skipping levels that aren't open on this location)
    - cheese/mouse_pos_outer as above, only one of inner or outer should be provided
    - must_be_dec_square ensures that the next step to cheese and top-right-corner are different at the mouse location
    """
    assert (
        rand_region <= WORLD_DIM
    ), "rand_region must be less than or equal to WORLD_DIM."
    assert rand_region > 0, "rand_region must be greater than 0."
    assert num_obs > 0, "num_obs must be greater than 0."
    assert (
        maze_dim is None or maze_dim <= WORLD_DIM
    ), "maze_dim must be less than or equal to WORLD_DIM."
    assert maze_dim is None or maze_dim > 0, "maze_dim must be greater than 0."
    assert (
        mouse_pos_inner is None or mouse_pos_outer is None
    ), "only specify one of mouse_pos_inner, mouse_pos_outer"
    assert (
        cheese_pos_inner is None or cheese_pos_outer is None
    ), "only specify one of cheese_pos_inner, cheese_pos_outer"

    # venvs = create_venv(num_obs, start_level=0, num_levels=0)
    # TODO ensure that if on_training is True, then the cheese is in the top-right rand_region corner

    def venv_gen_func():
        last_start_level = start_level
        while True:
            yield (
                create_venv(
                    1,
                    start_level=last_start_level,
                    num_levels=1 if deterministic_levels else 0,
                ),
                last_start_level,
            )
            last_start_level += 1

    venv_gen = venv_gen_func()

    # Create N random observations, customizing and skipping as needed based on constraints
    state_bytes_list = []
    metadata_list = []
    rng = np.random.default_rng(random_seed)
    with tqdm(total=num_obs, disable=not show_pbar) as pbar:
        while len(state_bytes_list) < num_obs:
            venv, this_level = next(venv_gen)
            env_state = state_from_venv(venv, idx=0)
            full_grid = env_state.full_grid(with_mouse=False)
            inner_grid = env_state.inner_grid(with_mouse=False)

            # Skip this level if it isn't the right size
            maze_dim_this = inner_grid.shape[0]
            if maze_dim is not None and maze_dim_this != maze_dim:
                continue

            # Calculate mouse/cheese outer positions, if either inner or out is
            # specified.
            padding = (env_state.world_dim - maze_dim_this) // 2
            if mouse_pos_inner is not None:
                mouse_pos_outer = (
                    mouse_pos_inner[0] + padding,
                    mouse_pos_inner[1] + padding,
                )
            if cheese_pos_inner is not None:
                cheese_pos_outer = (
                    cheese_pos_inner[0] + padding,
                    cheese_pos_inner[1] + padding,
                )

            if (
                mouse_pos_outer is not None
                and mouse_pos_outer == cheese_pos_outer
            ):
                warn("mouse and cheese positions must be different")

            # Get legal mouse positions
            legal_mouse_positions = get_legal_mouse_positions(full_grid)

            # If mouse position is specified, set it if legal, otherwise skip
            # If not specified, randomize
            if mouse_pos_outer is not None:
                if mouse_pos_outer in legal_mouse_positions:
                    mr, mc = mouse_pos_outer
                else:
                    continue
            else:
                mr, mc = legal_mouse_positions[
                    rng.integers(len(legal_mouse_positions))
                ]

            # Set the mouse position
            env_state.set_mouse_pos(mr, mc)

            # Remove the cheese if required
            if not spawn_cheese:
                remove_cheese_from_state(env_state)
                cheese_pos_outer_this = None
            # Otherwise, force the cheese to a speciic location if provided,
            # skipping if not valid
            elif cheese_pos_outer is not None:
                cheese_pos_outer_this = cheese_pos_outer
                if (
                    cheese_pos_outer in legal_mouse_positions
                    or full_grid[cheese_pos_outer] == CHEESE
                ):
                    move_cheese_in_state(env_state, cheese_pos_outer)
                else:
                    continue
            else:
                # Store the cheese location for metadata later
                cheese_pos_outer_this = get_cheese_pos(full_grid)

            # Skip if the resulting mouse/cheese locations don't match the
            # decision square requirement, if any
            # print(mr, mc)
            # print(legal_mouse_positions)
            if must_be_dec_square or return_metadata:
                graph = maze_grid_to_graph(inner_grid)
                mr_inner, mc_inner = mr - padding, mc - padding
                path_to_cheese = get_path_to_cheese(
                    inner_grid, graph, (mr_inner, mc_inner)
                )
                path_to_corner = get_path_to_corner(
                    inner_grid, graph, (mr_inner, mc_inner)
                )

                def path_step_inner_to_outer(path):
                    return (
                        (path[1][0] + padding, path[1][1] + padding)
                        if len(path) > 1
                        else (mr, mc)
                    )

                next_pos_cheese_outer = path_step_inner_to_outer(
                    path_to_cheese
                )
                next_pos_corner_outer = path_step_inner_to_outer(
                    path_to_corner
                )
                # print(path_to_cheese[1], path_to_corner[1])
                if must_be_dec_square and (
                    next_pos_cheese_outer == next_pos_corner_outer
                ):
                    continue

            # If we get here, we're ready to use this state
            state_bytes_list.append(env_state.state_bytes)
            if return_metadata:
                metadata_list.append(
                    dict(
                        level_seed=this_level,
                        mouse_pos_outer=(mr, mc),
                        cheese_pos_outer=cheese_pos_outer_this,
                        next_pos_cheese_outer=next_pos_cheese_outer,
                        next_pos_corner_outer=next_pos_corner_outer,
                        path_to_cheese=path_to_cheese,
                        path_to_corner=path_to_corner,
                        maze_dim=maze_dim_this,
                    )
                )

            pbar.update(1)

    venvs = create_venv(num_obs, start_level=0, num_levels=0)
    venvs.env.callmethod("set_state", state_bytes_list)
    if return_metadata:
        return (
            venvs.reset().astype(np.float32),
            metadata_list,
            metadata_list[-1]["level_seed"] + 1,
        )
    else:
        return venvs.reset().astype(np.float32)


def get_random_obs(
    num_obs: int = 1,
    on_training: bool = True,
    rand_region: int = 5,
    spawn_cheese: bool = True,
):
    """Get num_obs observations from the maze environment. If on_training is True, then the observation is from a training level where the cheese is in the top-right rand_region corner."""
    assert (
        rand_region <= WORLD_DIM
    ), "rand_region must be less than or equal to WORLD_DIM."
    assert rand_region > 0, "rand_region must be greater than 0."
    assert num_obs > 0, "num_obs must be greater than 0."

    venvs = create_venv(num_obs, start_level=0, num_levels=0)
    # TODO ensure that if on_training is True, then the cheese is in the top-right rand_region corner

    # Randomly place the mouse in each environment
    state_bytes_list = []
    for i in range(num_obs):
        env_state = state_from_venv(venvs, i)
        grid = env_state.full_grid(with_mouse=False)
        legal_mouse_positions = get_legal_mouse_positions(grid)

        # choose a random legal mouse position
        mx, my = legal_mouse_positions[
            np.random.randint(len(legal_mouse_positions))
        ]

        # set the mouse position
        env_state.set_mouse_pos(mx, my)

        if not spawn_cheese:
            remove_cheese(venvs, i)

        # set the state
        state_bytes_list.append(env_state.state_bytes)

    venvs.env.callmethod("set_state", state_bytes_list)
    return venvs.reset().astype(np.float32)


def venv_with_all_mouse_positions(venv):
    """
    From a venv with a single env, create a new venv with one env for each legal mouse position.

    Returns venv_all, (legal_mouse_positions, inner_grid_without_mouse)
    Typically you'd call this with `venv_all, _ = venv_with_all_mouse_positions(venv)`,
    The extra return values are useful for conciseness sometimes.
    """
    assert (
        venv.num_envs == 1
    ), f"Did you forget to use maze.copy_venv to get a single env?"

    sb_back = venv.env.callmethod("get_state")[0]
    env_state = EnvState(sb_back)

    grid = env_state.inner_grid(with_mouse=False)
    legal_mouse_positions = get_legal_mouse_positions(grid)

    # convert coords from inner to outer grid coordinates
    padding = get_padding(grid)

    # create a venv for each legal mouse position
    state_bytes_list = []
    for mx, my in legal_mouse_positions:
        # we keep a backup of the state bytes for efficiency, as calling set_mouse_pos
        # implicitly calls _parse_state_bytes, which is slow. this is a hack.
        # NOTE: Object orientation hurts us here. It would be better to have functions.
        env_state.set_mouse_pos(mx + padding, my + padding)
        state_bytes_list.append(env_state.state_bytes)
        env_state.state_bytes = sb_back

    threads = 1 if len(legal_mouse_positions) < 100 else os.cpu_count()
    venv_all = create_venv(
        num=len(legal_mouse_positions),
        num_threads=threads,
        num_levels=1,
        start_level=1,
    )
    venv_all.env.callmethod("set_state", state_bytes_list)
    return venv_all, (legal_mouse_positions, grid)
