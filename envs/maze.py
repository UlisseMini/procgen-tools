"""
Code to create, serialize, deserialize and edit mazes through their c++ state.

Source due to monte, edited by uli
https://gist.github.com/montemac/6f9f636507ec92967071bb755f37f17b

!!!!!! WARNING: This code is for the *lauro branch* of procgenAISC, it breaks on master !!!!!!
"""

import struct
import typing
from dataclasses import dataclass
from collections import deque
import numpy as np

# Constants in numeric maze representation
CHEESE = 2
EMPTY = 100 # TODO: Change to OPEN (terminology isn't consistent)
BLOCKED = 51

# Parse the environment state dict

MAZE_STATE_DICT_TEMPLATE = [
    ['int',    'SERIALIZE_VERSION'],
    ['string', 'game_name'],
    ['int',    'options.paint_vel_info'],
    ['int',    'options.use_generated_assets'],
    ['int',    'options.use_monochrome_assets'],
    ['int',    'options.restrict_themes'],
    ['int',    'options.use_backgrounds'],
    ['int',    'options.center_agent'],
    ['int',    'options.debug_mode'],
    ['int',    'options.distribution_mode'],
    ['int',    'options.use_sequential_levels'],
    ['int',    'options.use_easy_jump'],
    ['int',    'options.plain_assets'],
    ['int',    'options.physics_mode'],
    ['int',    'grid_step'],
    ['int',    'level_seed_low'],
    ['int',    'level_seed_high'],
    ['int',    'game_type'],
    ['int',    'game_n'],
    # level_seed_rand_gen.serialize(b'],
    ['int',    'level_seed_rand_gen.is_seeded'],
    ['string', 'level_seed_rand_gen.str'],
    # end level_seed_rand_gen.serialize(b'],
    # rand_gen.serialize(b'],
    ['int',    'rand_gen.is_seeded'],
    ['string', 'rand_gen.str'],
    # end rand_gen.serialize(b'],
    ['float',  'step_data.reward'],
    ['int',    'step_data.done'],
    ['int',    'step_data.level_complete'],
    ['int',    'action'],
    ['int',    'timeout'],
    ['int',    'current_level_seed'],
    ['int',    'prev_level_seed'],
    ['int',    'episodes_remaining'],
    ['int',    'episode_done'],
    ['int',    'last_reward_timer'],
    ['float',  'last_reward'],
    ['int',    'default_action'],
    ['int',    'fixed_asset_seed'],
    ['int',    'cur_time'],
    ['int',    'is_waiting_for_step'],
    # end Game::serialize(b'],
    ['int',    'grid_size'],
    # write_entities(b, entities'],
    ['int',    'ents.size'],
    #for (size_t i = 0; i < ents.size(', i++)
    ['loop',   'ents', 'ents.size', [
        # ents[i]->serialize(b'],
        ['float',  'x'],
        ['float',  'y'],
        ['float',  'vx'],
        ['float',  'vy'],
        ['float',  'rx'],
        ['float',  'ry'],
        ['int',    'type'],
        ['int',    'image_type'],
        ['int',    'image_theme'],
        ['int',    'render_z'],
        ['int',    'will_erase'],
        ['int',    'collides_with_entities'],
        ['float',  'collision_margin'],
        ['float',  'rotation'],
        ['float',  'vrot'],
        ['int',    'is_reflected'],
        ['int',    'fire_time'],
        ['int',    'spawn_time'],
        ['int',    'life_time'],
        ['int',    'expire_time'],
        ['int',    'use_abs_coords'],
        ['float',  'friction'],
        ['int',    'smart_step'],
        ['int',    'avoids_collisions'],
        ['int',    'auto_erase'],
        ['float',  'alpha'],
        ['float',  'health'],
        ['float',  'theta'],
        ['float',  'grow_rate'],
        ['float',  'alpha_decay'],
        ['float',  'climber_spawn_x',]]],
    # end ents[i]->serialize(b'],
    # end write_entities
    ['int',    'use_procgen_background'],
    ['int',    'background_index'],
    ['float',  'bg_tile_ratio'],
    ['float',  'bg_pct_x'],
    ['float',  'char_dim'],
    ['int',    'last_move_action'],
    ['int',    'move_action'],
    ['int',    'special_action'],
    ['float',  'mixrate'],
    ['float',  'maxspeed'],
    ['float',  'max_jump'],
    ['float',  'action_vx'],
    ['float',  'action_vy'],
    ['float',  'action_vrot'],
    ['float',  'center_x'],
    ['float',  'center_y'],
    ['int',    'random_agent_start'],
    ['int',    'has_useful_vel_info'],
    ['int',    'step_rand_int'],
    # asset_rand_gen.serialize(b'],
    ['int',    'asset_rand_gen.is_seeded'],
    ['string', 'asset_rand_gen.str'],
    # end asset_rand_gen.serialize(b'],
    ['int',    'main_width'],
    ['int',    'main_height'],
    ['int',    'out_of_bounds_object'],
    ['float',  'unit'],
    ['float',  'view_dim'],
    ['float',  'x_off'],
    ['float',  'y_off'],
    ['float',  'visibility'],
    ['float',  'min_visibility'],
    # grid.serialize(b'],
    ['int',    'w'],
    ['int',    'h'],
    # b->write_vector_int(data'],
    ['int',    'data.size'],
    # for (auto i : v) {
    ['loop',   'data', 'data.size', [['int',    'i']]],
    # end b->write_vector_int(data'],
    # end grid.serialize(b'],
    # end BasicAbstractGame::serialize(b'],
    ['int',    'maze_dim'],
    ['int',    'world_dim'], 
    ['int',    'END_OF_BUFFER']]

@dataclass
class StateValue:
    val: typing.Any
    idx: int


def parse_maze_state_bytes(state_bytes: bytes, assert_=True) -> typing.Dict[str, StateValue]:
    # Functions to read values of different types
    def read_fixed(sb, idx, fmt):
        sz = struct.calcsize(fmt)
        # print(f'{idx} chomp {sz} got {len(sb[idx:(idx+sz)])} fmt {fmt}')
        val = struct.unpack(fmt, sb[idx:(idx+sz)])[0]
        idx += sz
        return val, idx
    read_int = lambda sb, idx: read_fixed(sb, idx, '@i')
    read_float = lambda sb, idx: read_fixed(sb, idx, '@f')
    def read_string(sb, idx):
        sz, idx = read_int(sb, idx)
        val = sb[idx:(idx+sz)].decode('ascii')
        idx += sz
        return val, idx

    # Function to process a value definition and return a value (called recursively for loops)
    def parse_value(vals, val_def, idx):
        typ = val_def[0]
        name = val_def[1]
        # print((typ, name))
        if typ == 'int':
            val, idx = read_int(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == 'float':
            val, idx = read_float(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == 'string':
            val, idx = read_string(state_bytes, idx)
            vals[name] = StateValue(val, idx)
        elif typ == 'loop':
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
        assert serialize_maze_state(vals, assert_=False) == state_bytes, 'serialize(deserialize(state_bytes)) != state_bytes'
    return vals

def serialize_maze_state(state_vals: typing.Dict[str, StateValue], assert_=True) -> bytes:
    # Serialize any value to a bytes object
    def serialize_val(val):
        if isinstance(val, StateValue):
            val = val.val
        if isinstance(val, int):
            return struct.pack('@i', val)
        elif isinstance(val, float):
            return struct.pack('@f', val)
        elif isinstance(val, str):
            return serialize_val(len(val)) + val.encode('ascii')
        else:
            raise ValueError(f'type(val)={type(val)} not handled')

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

    state_bytes = b''.join([serialize_val(val) for val in flat_vals])

    if assert_:
        assert parse_maze_state_bytes(state_bytes, assert_=False) == state_vals, 'deserialize(serialize(state_vals)) != state_vals'
    return state_bytes


# TODO: Better abstraction would return some kind of "Grid object" which automatically
# updated state_vals when updated, as well as having helper functions for common ops.

def get_grid(state_vals):
    "Get numpy (world_dim, world_dim) grid of the maze. "
    world_dim = state_vals['world_dim'].val
    grid_vals = np.array([dd['i'].val for dd in state_vals['data']]).reshape(world_dim, world_dim)
    return grid_vals

def set_grid(state_vals, grid):
    "Set the grid of the maze. "
    world_dim = state_vals['world_dim'].val
    assert grid.shape == (world_dim, world_dim)
    for i, dd in enumerate(state_vals['data']):
        dd['i'].val = int(grid.ravel()[i])

def get_cheese_pos(grid: np.ndarray):
    "Get (x, y) position of the cheese in the grid"
    num_cheeses = (grid == CHEESE).sum()
    assert num_cheeses == 1, f'num_cheeses={num_cheeses} should be 1'
    return tuple(ix[0] for ix in np.where(grid == CHEESE))

def set_cheese_pos(grid: np.ndarray, x, y):
    "Set the cheese position in the grid"
    grid[grid == CHEESE] = EMPTY
    assert grid[x, y] == EMPTY, f'grid[{x}, {y}]={grid[x, y]} should be EMPTY={EMPTY}'
    grid[x, y] = CHEESE


def inner_grid(grid: np.ndarray) -> np.ndarray:
    "Get the inside of the maze, ie. the stuff within the outermost walls"
    # uses the fact that the mouse always starts in the bottom left.
    bl = next(i for i in range(len(grid)) if grid[i][i] == EMPTY)
    return grid[bl:-bl, bl:-bl]


def is_tree(inner_grid: np.ndarray) -> bool:
    """
    Is there exactly one path between any two open squares in the maze?
    (Also known as, is the set of open squares a spanning tree)
    """

    def get_neighbors(x, y):
        "Get the neighbors of (x, y) in the grid"
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    def ingrid(n):
        "Is (x, y) in the grid?"
        return 0 <= n[0] < inner_grid.shape[0] and 0 <= n[1] < inner_grid.shape[1]

    def get_open_neighbors(x, y):
        "Get the open neighbors of (x, y) in the grid"
        return [n for n in get_neighbors(x, y) if ingrid(n) and inner_grid[n] == EMPTY]

    visited_edges = set()
    visited_nodes = set()
    stack = [(0,0)]
    while stack:
        node = stack.pop()
        if node in visited_nodes:
            print(f'{node} already visited, a cycle!')
            return False
        visited_nodes.add(node)
        for neighbor in get_open_neighbors(*node):
            edge = (node, neighbor)
            if edge not in visited_edges and edge[::-1] not in visited_edges:
                stack.append(neighbor)
                visited_edges.add(edge)

    # There were no cycles, if we visited all the nodes, then it's a tree
    visited_all_nodes = len(visited_nodes) == (inner_grid == EMPTY).sum()
    if not visited_all_nodes:
        # only print debugging on assertion fail
        print(f'visited {len(visited_nodes)} out of {(inner_grid == EMPTY).sum()} nodes')
        return False
    return True

# TODO? Constructor/wrap with docs for options. Or inject into docstring
# from procgen import ProcgenGym3Env

from envs.procgen_wrappers import TransposeFrame, ScaledFloatFrame, VecExtractDictObs
from gym3 import ToBaselinesVecEnv
# TODO: Options to use ViewerWrapper, VideoRecorderWrapper from gym3.

def wrap_venv(venv):
    venv = ToBaselinesVecEnv(venv) # gym3 env to gym env converter
    venv = VecExtractDictObs(venv, "rgb")

    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv

