# %%
# Imports and initial setup
%reload_ext autoreload
%autoreload 2

from typing import List, Tuple, Dict, Union, Optional, Callable
import random
import itertools

import numpy as np
import numpy.linalg
import pandas as pd
import xarray as xr
import torch as t
import torch.nn.functional as F
from torch.distributions import Categorical
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm import tqdm
from einops import rearrange
from IPython.display import Video, display, clear_output
from ipywidgets import *
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt 
plt.ioff() # disable interactive plotting, so that we can control where figures show up when refreshed by an ipywidget

import circrl.module_hook as cmh
import procgen_tools.models as models
import procgen_tools.maze as maze
import procgen_tools.patch_utils as patch_utils
import procgen_tools.vfield as vfield
from procgen import ProcgenGym3Env

path_prefix = '../'

# Load model and hook it
rand_region = 5
policy = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, t.device('cpu'))
hook = cmh.ModuleHook(policy)

# %%
# Run stats on cheese patching

# Pick a seed for randomizing the level_seeds,
# and a number of levels to generate
random_seed = 43
num_levels = 30

# Pick direction of patch (patch cheese to make less cheese, 
# or patch no-cheese to make cheesier)
patch_more_cheese = False

# Pick value to test patching on
value_label = 'embedder.block2.res1.resadd_out'

# Pick a maze size so that this variable is controlled (makes later 
# stuff much simpler)
maze_dim = 15

# Pick some cheese positions to test (e.g. corners, and points in a 
# square closer to middle?)  (In inner_grid coords)
cheese_poss = [(2, 2), (2, 12), (12, 2), (12, 12)]
#cheese_poss = [(2, 12)]
num_cheese_pos = len(cheese_poss)

# Pick some mouse positions to test (all true cells that are known 
# to be open, and which could be branch squares?)
mouse_poss = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10)]
#mouse_poss = [(8, 8)]
num_mouse_pos = len(mouse_poss)

def remove_cheese_from_state(state):
    grid = state.full_grid()
    grid[grid == maze.CHEESE] = maze.EMPTY
    state.set_grid(grid)
    return state

def move_cheese_in_state(state, new_cheese_pos):
    grid = state.full_grid()
    grid[grid == maze.CHEESE] = maze.EMPTY
    grid[new_cheese_pos] = maze.CHEESE
    state.set_grid(grid)
    return state

# Iterate through level_seeds, skipping all that don't match maze size
random.seed(random_seed)
obs_no_cheese_list = []
obs_cheese_list = []
level_seeds = []
with tqdm(total=num_levels) as pbar:
    while len(level_seeds) < num_levels:
        # Get the level seed
        level_seed = random.randint(0, int(1e6))
        
        # Create a venv to be the "no cheese" envs, which needs to have the same number
        # as mouse positions we want to evaluate
        venv_no_cheese = patch_utils.create_venv(num=num_mouse_pos, start_level=level_seed, 
            num_levels=1)
        state_bytes_list = venv_no_cheese.env.callmethod("get_state")
        states = [maze.EnvState(sb) for sb in state_bytes_list]
        # Check the size, skipping if not correct
        maze_dim_this = states[0].inner_grid().shape[0]
        if maze_dim_this != maze_dim:
            continue
        level_seeds.append(level_seed)
        padding = (states[0].world_dim - maze_dim_this) // 2
        # Remove the cheese from all the envs
        states = [remove_cheese_from_state(state) for state in states]
        state_bytes_list = [state.state_bytes for state in states]
        venv_no_cheese.env.callmethod("set_state", state_bytes_list)
        # Get the "no cheese" observations
        obs_no_cheese_list.append(venv_no_cheese.reset().astype(np.float32))

        # Get the "with cheese" envs; same process, but need to have
        # one env for every cheese and mouse pos combo
        venv_cheese = patch_utils.create_venv(num=num_cheese_pos*num_mouse_pos, 
            start_level=level_seed, num_levels=1)
        state_bytes_list = venv_cheese.env.callmethod("get_state")

        # Position the cheese and mouse as needed in each venv
        state_bytes_new = []
        for idx_mouse, mouse_pos in enumerate(mouse_poss):
            for idx_cheese, cheese_pos in enumerate(cheese_poss):
                idx = idx_mouse * num_cheese_pos + idx_cheese
                state = maze.EnvState(state_bytes_list[idx])
                # Place objects
                move_cheese_in_state(state, 
                    (cheese_pos[0]+padding, cheese_pos[1]+padding))
                state.set_mouse_pos(mouse_pos[1]+padding, mouse_pos[0]+padding)
                # Update state bytes
                state_bytes_new.append(state.state_bytes)
        venv_cheese.env.callmethod("set_state", state_bytes_new)

        # Get the "with cheese" observations
        obs_cheese_list.append(venv_cheese.reset().astype(np.float32))

        pbar.update(1)

# %%
# Get all the values and action logits for all the observations

mouse_pos_coords = np.array(mouse_poss, dtype='i,i')
cheese_pos_coords = np.array(cheese_poss, dtype='i,i')

# Rearrange lists into xarrays with appropriate non-flat dims
obs_no_cheese = xr.DataArray(
    data = rearrange(obs_no_cheese_list, 'lev mse ... -> lev mse ...'),
    dims = ['level_seed', 'mouse_pos', 'rgb', 'y', 'x'],
    coords = dict(level_seed=level_seeds, mouse_pos=mouse_pos_coords))
obs_cheese = xr.DataArray(
    data = rearrange(obs_cheese_list, 'lev (mse chs) ... ->  lev mse chs ...', 
        mse=num_mouse_pos),
    dims = ['level_seed', 'mouse_pos', 'cheese_pos', 'rgb', 'y', 'x'],
    coords = dict(level_seed=level_seeds, mouse_pos=mouse_pos_coords,
        cheese_pos=cheese_pos_coords))

def stack(array):
    dims_to_stack = [dim for dim in 
        ['level_seed', 'mouse_pos', 'cheese_pos']
        if dim in array.dims]
    return array.stack(dict(sbatch=dims_to_stack)).transpose('sbatch',...)

def unstack(array):
    existing_dims = [dim for dim in array.dims if dim != 'sbatch']
    unstacked = array.unstack('sbatch')
    new_dims = [dim for dim in unstacked.dims if not dim in existing_dims]
    return unstacked.transpose(*new_dims, ...)

def add_cheese_dim(array):
    return array.expand_dims(dict(cheese_pos=cheese_pos_coords), axis=2)

# Run both obs through the hooked network and extract values, unstacked
# (Force the no-cheese arrays to have the same shape as the cheese arrays,
# for simpler indexing later)
action_logits_label = 'fc_policy_out'
hook.run_with_input(stack(obs_no_cheese))
value_no_cheese = add_cheese_dim(unstack(hook.get_value_by_label(value_label)))
action_logits_no_cheese = \
    add_cheese_dim(unstack(hook.get_value_by_label(action_logits_label)))
hook.run_with_input(stack(obs_cheese))
value_cheese = unstack(hook.get_value_by_label(value_label))
action_logits_cheese = unstack(hook.get_value_by_label(action_logits_label))

# %%
# Do the patching

num_src_levels = 1

# # Function to map (idx_level, idx_cheese, idx_mouse) arrays to flattened batch idx array
# def idxs_to_flat(idx_arrays):
#     return np.ravel_multi_index(idx_arrays, (num_levels, num_mouse_pos, num_cheese_pos))

def logits_to_prob_arrows(logits):
    log_probs = F.log_softmax(t.from_numpy(logits.copy()), dim=1)
    probs = Categorical(logits=log_probs).probs.detach().numpy()
    return np.einsum('ba,ad->ba', probs, models.MAZE_ACTION_DELTAS_BY_INDEX)

def score_vect_cosim(action_logits_targ, action_logits_patched):
    # Turn logits into probs
    arrows_targ = logits_to_prob_arrows(action_logits_targ)
    arrows_patched = logits_to_prob_arrows(action_logits_patched)
    return np.einsum('bd,bd->b', arrows_targ, arrows_patched) / \
        (np.linalg.norm(arrows_targ, axis=1) * np.linalg.norm(arrows_patched, axis=1))

# Calculate all the cheese - (no cheese) diffs as patch sources
cheese_diff = value_cheese - value_no_cheese

# Test patching between different levels, same mouse and cheese pos
# Iterate over source levels
scores_np = np.zeros((num_src_levels, num_levels, num_mouse_pos, num_cheese_pos))
scores_nopatch_np = np.zeros_like(scores_np)
for src_level_idx in range(num_src_levels):
    # Grab the cheese diffs for this source level
    cheese_diff_this = cheese_diff.isel(level_seed=src_level_idx)
    # Iterate over levels-to-patch (could vectorize)
    for patch_level_idx in range(num_levels):
        # if src_level_idx == patch_level_idx:
        #     continue # Don't patch the source level
        # Make a patch function that will apply the appropriate
        # diff at all mouse/cheese locations
        cheese_diff_this_stk_t = t.from_numpy(
            stack(cheese_diff_this).values)
        def patch_func(outp):
            #return t.from_numpy(
            #    stack(value_cheese.isel(level_seed=src_level_idx)).values)
            #return t.zeros_like(outp)
            #return outp
            return outp + (1. if patch_more_cheese else -1.)*cheese_diff_this_stk_t
        # Apply the patch and run the network
        if patch_more_cheese:
            obs_to_use = obs_no_cheese.isel(level_seed=patch_level_idx)
            action_logits_orig = action_logits_no_cheese.isel(level_seed=patch_level_idx)
            action_logits_targ = action_logits_cheese.isel(level_seed=patch_level_idx)
        else:
            obs_to_use = obs_cheese.isel(level_seed=patch_level_idx)
            action_logits_orig = action_logits_cheese.isel(level_seed=patch_level_idx)
            action_logits_targ = action_logits_no_cheese.isel(level_seed=patch_level_idx)
        hook.run_with_input(stack(obs_to_use), 
            patches={value_label: patch_func})
        # Get the (still stacked) patched logits
        action_logits_patched_stk = hook.get_value_by_label(action_logits_label)
        # Get the metric score
        scores = score_vect_cosim(
            stack(action_logits_targ).values,
            action_logits_patched_stk.values)
        scores_np[src_level_idx, patch_level_idx, ...] = rearrange(scores,
            '(mse chs) -> mse chs', mse=num_mouse_pos)
        # Save the baseline "no patch" score also
        scores_nopatch_np[src_level_idx, patch_level_idx, ...] = rearrange(
            score_vect_cosim(
                stack(action_logits_targ).values,
                stack(action_logits_orig).values),
            '(mse chs) -> mse chs', mse=num_mouse_pos)

scores = xr.DataArray(
    data = scores_np,
    dims = ['src_level_seed', 'patch_level_seed', 'mouse_pos', 'cheese_pos'],
    coords = dict(
        src_level_seed=obs_cheese.indexes['level_seed'].values[:num_src_levels],
        patch_level_seed=obs_cheese.indexes['level_seed'].values,
        mouse_pos=mouse_pos_coords,
        cheese_pos=cheese_pos_coords))
scores_nopatch = scores.copy(data=scores_nopatch_np)

display((scores - scores_nopatch).mean().item())

px.histogram((scores - scores_nopatch).values.flatten()).show()

# display('No patch baseline', scores_nopatch_np.squeeze())
# display('With patch', scores_np.squeeze())
#score_improvement

# Temp: visualize some stuff for sanity checking
# px.imshow(rearrange(obs_cheese.values.squeeze(), 
#     '(lh lw) c h w -> (lh h) (lw w) c', lh=2))


# Test patching within the same level, same cheese pos, different mouse pos



# Randomly or deterministically pick a small number of "source data points", 
# which will be used to calculate cheese diff tensors for patching into other mazes.  
# These are (seed, cheese pos, mouse pos) tuples.

# For each source point, calculate the cheese diff tensor at the 
# provided value.

# For each source, for each target, patch in the cheese diff and 
# apply metrics to the resulting pre and post logits
# %%
