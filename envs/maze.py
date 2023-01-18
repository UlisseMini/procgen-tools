"""
Code to create, serialize, deserialize and edit mazes through their c++ state.

Source due to monte, edited by uli
https://gist.github.com/montemac/6f9f636507ec92967071bb755f37f17b

!!!!!! WARNING: This code is for the *lauro branch* of procgenAISC, it breaks on master !!!!!!
"""

import struct
import typing
from dataclasses import dataclass
import numpy as np
import heapq

# Constants in numeric maze representation
CHEESE = 2
EMPTY = 100 # TODO: Change to OPEN (terminology isn't consistent)
BLOCKED = 51
MOUSE = 25 # UNOFFICIAL. The mouse isn't in the grid in procgen.

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


# TODO: Rename functions to be clear which take state_vals vs. grid (e.g. sget vs. gget?)

def get_grid(state_vals):
    "Get numpy (world_dim, world_dim) grid of the maze. "
    world_dim = state_vals['world_dim'].val
    grid_vals = np.array([dd['i'].val for dd in state_vals['data']]).reshape(world_dim, world_dim)
    return grid_vals

def get_mouse_grid_pos(state_vals):
    "Get (x, y) position of mouse in grid."
    ents = state_vals['ents'][0]
    # flipped turns out to be oriented right for grid.
    return int(ents['y'].val), int(ents['x'].val)

def get_grid_with_mouse(state_vals):
    "Get grid with mouse position"
    grid = get_grid(state_vals)
    grid[get_mouse_pos(state_vals)] = MOUSE
    return grid

def set_grid_with_mouse(state_vals, grid):
    "Set state_vals <- grid with mouse position. grid must be (world_dim, world_dim)"
    assert grid.shape == (state_vals['world_dim'].val, state_vals['world_dim'].val)
    assert (grid==MOUSE).sum() == 1, f'grid has {(grid==MOUSE).sum()} mice'

    grid = grid.copy()
    x,y = [c[0] for c in np.where(grid==MOUSE)]

    state_vals['ents'][0]['x'].val = float(y) + 0.5 # flip again to get back to original orientation
    state_vals['ents'][0]['y'].val = float(x) + 0.5

    grid[grid==MOUSE] = EMPTY
    set_grid(state_vals, grid)

    return state_vals


def set_grid(state_vals, grid):
    "Set the grid of the maze."
    assert (grid==MOUSE).sum() == 0, 'use set_grid_with_mouse'
    world_dim = state_vals['world_dim'].val
    assert grid.shape == (world_dim, world_dim)
    for i, dd in enumerate(state_vals['data']):
        dd['i'].val = int(grid.ravel()[i])


def get_cheese_pos(grid: np.ndarray) -> typing.Tuple[int, int]:
    "Get (x, y) position of the cheese in the grid"
    num_cheeses = (grid == CHEESE).sum()
    assert num_cheeses == 1, f'num_cheeses={num_cheeses} should be 1'
    return tuple(ix[0] for ix in np.where(grid == CHEESE))


def get_mouse_pos(grid: np.ndarray) -> typing.Tuple[int, int]:
    "Get (x, y) position of the mouse in the grid"
    num_mouses = (grid == MOUSE).sum()
    assert num_mouses == 1, f'{num_mouses} mice, should be 1'
    return tuple(ix[0] for ix in np.where(grid == MOUSE))


def set_cheese_pos(grid: np.ndarray, x, y):
    "Set the cheese position in the grid"
    grid[grid == CHEESE] = EMPTY
    assert grid[x, y] == EMPTY, f'grid[{x}, {y}]={grid[x, y]} should be EMPTY={EMPTY}'
    grid[x, y] = CHEESE


def inner_grid(grid: np.ndarray) -> np.ndarray:
    """
    Get the inside of the maze, ie. the stuff within the outermost walls.
    inner_grid(inner_grid(x)) = inner_grid(x) for all x.
    """
    # uses the fact that the mouse always starts in the bottom left.
    bl = next(i for i in range(len(grid)) if grid[i][i] != BLOCKED)
    if bl == 0: # edgecase! the whole grid is the inner grid.
        return grid
    return grid[bl:-bl, bl:-bl]


def euclidian_dist_to_cheese(grid: np.ndarray) -> float:
    "Compute the *euclidian* distance from the mouse to the cheese"
    mx, my = get_mouse_pos(grid)
    cx, cy = get_cheese_pos(grid)
    return np.sqrt((mx - cx)**2 + (my - cy)**2)




def _get_neighbors(x, y):
    "Get the neighbors of (x, y) in the grid"
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _ingrid(grid, n):
    "Is (x, y) in the grid?"
    return 0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1]

def _get_open_neighbors(grid, x, y):
    "Get the open neighbors of (x, y) in the grid"
    return [n for n in _get_neighbors(x, y) if _ingrid(grid, n) and grid[n] != BLOCKED]


def shortest_path_to_cheese(grid: np.ndarray) -> typing.Tuple[typing.Dict, typing.Dict]:
    "Compute the number of moves for the mouse to get the cheese (using A*)"
    assert (grid==MOUSE).sum() == 1, f'grid has {(grid==MOUSE).sum()} mice'
    assert (grid==CHEESE).sum() == 1, f'grid has {(grid==CHEESE).sum()} cheeses'
    grid = inner_grid(grid).copy()

    # A* search
    start = get_mouse_pos(grid)
    goal = get_cheese_pos(grid)
    frontier = []
    heapq.heappush(frontier, (0, start))

    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while len(frontier) > 0:
        current = heapq.heappop(frontier)[1]
        if current == goal:
            break

        for next in _get_open_neighbors(grid, *current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + euclidian_dist_to_cheese(grid)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    return cost_so_far, came_from


def is_tree(grid: np.ndarray, debug=False) -> bool:
    """
    Is there exactly one path between any two open squares in the maze?
    (Also known as, is the set of open squares a spanning tree)
    """
    grid = inner_grid(grid).copy()
    grid[grid == CHEESE] = EMPTY

    visited_edges = set()
    visited_nodes = set()
    stack = [(0,0)]
    while stack:
        node = stack.pop()
        if node in visited_nodes:
            if debug: print(f'{node} already visited, a cycle!')
            return False
        visited_nodes.add(node)
        for neighbor in _get_open_neighbors(grid, *node):
            edge = (node, neighbor)
            if edge not in visited_edges and edge[::-1] not in visited_edges:
                stack.append(neighbor)
                visited_edges.add(edge)

    # There were no cycles, if we visited all the nodes, then it's a tree
    visited_all_nodes = len(visited_nodes) == (grid == EMPTY).sum()
    if debug:
        print(f'visited {len(visited_nodes)} out of {(grid == EMPTY).sum()} nodes')
    return visited_all_nodes

def on_distribution(grid: np.ndarray, p=print) -> bool:
    "Is the given maze something that could have been generated during training?"

    # Make a copy of the inner grid without the cheese and mouse
    g = inner_grid(grid).copy()
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


def grid_editor(grid: np.ndarray, node_radius='8px'):
    from ipywidgets import GridspecLayout, Button, Layout, HBox, Output

    CELL_TO_COLOR = {EMPTY: '#fde724', BLOCKED: '#24938b', CHEESE: '#440154', MOUSE: '#3c4d8a'}
    CELL_TO_CHAR = {EMPTY: 'Empty', BLOCKED: 'Blocked', CHEESE: '🧀', MOUSE: '🐭'}

    num_mice = (grid==MOUSE).sum()
    assert num_mice in (0,1), f'num_mice {num_mice}'

    # will maintain a pointer into grid
    g = inner_grid(grid)
    rows, cols = g.shape
    wgrid = GridspecLayout(rows, cols, width='min-content')

    output = Output()

    def button_clicked(b: Button):
        i, j = getattr(b, 'coord')
        if (g == CHEESE).sum() == 0:
            g[i, j] = CHEESE
        elif (g == MOUSE).sum() == 0 and num_mice > 0:
            g[i, j] = MOUSE
        else:
            g[i, j] = {EMPTY: BLOCKED, BLOCKED: EMPTY, CHEESE: EMPTY, MOUSE: EMPTY}[g[i,j]]
        b.style.button_color = CELL_TO_COLOR[g[i,j]] # type: ignore
        b.tooltip = CELL_TO_CHAR[g[i,j]]
        with output:
            output.clear_output()
            on_distribution(g)

    for i in range(rows):
        for j in range(cols):
            b = Button(layout=Layout(padding=node_radius, width='0px', height='0px', margin='0px'))
            b.tooltip = CELL_TO_CHAR[g[i,j]]
            setattr(b, 'coord', (i, j)) # monkey patch to pass coords
            b.style.button_color = CELL_TO_COLOR[g[i,j]] # type: ignore
            b.on_click(button_clicked)
            # flip the grid so it's oriented correctly, like origin=lower in matplotlib.
            wgrid[rows-i-1, j] = b
    return HBox([wgrid, output])


from envs.procgen_wrappers import TransposeFrame, ScaledFloatFrame, VecExtractDictObs
from gym3 import ToBaselinesVecEnv

def wrap_venv(venv) -> ToBaselinesVecEnv:
    "Wrap a vectorized env, making it compatible with the gym apis, transposing, scaling, etc."
    # TODO: Move this to another file (same thing is used for coinrun)

    venv = ToBaselinesVecEnv(venv) # gym3 env to gym env converter
    venv = VecExtractDictObs(venv, "rgb")

    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv # type: ignore - ToBaselinesVecEnv gives best type annotations

