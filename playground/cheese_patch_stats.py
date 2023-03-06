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
# Functions for running stats on cheese patching

def stack(array, dims_can_stack=['level_seed', 'mouse_pos', 'cheese_pos']):
    dims_to_stack = [dim for dim in dims_can_stack
        if dim in array.dims]
    return array.stack(dict(sbatch=dims_to_stack)).transpose('sbatch',...)

def unstack(array):
    existing_dims = [dim for dim in array.dims if dim != 'sbatch']
    unstacked = array.unstack('sbatch')
    new_dims = [dim for dim in unstacked.dims if not dim in existing_dims]
    return unstacked.transpose(*new_dims, ...)

def make_full_dims(array, coords):
    dims_to_add = {dd: cc for dd, cc in coords.items() if not dd in array.dims}
    return array.expand_dims(dims_to_add).transpose(
        *(tuple(coords.keys()) + (...,)))

def get_obs_batch_on_grid(
        levels = 10,         # Pass int to generate N random levels, pass iterable of level_seeds for specific levels
        cheese_poss = None,    # Iterable of fixed cheese positions to use in every level seed, None to use position in level
        mouse_poss = None,     # Iterable of fixed mouse postions, or None to place 
        maze_dim = 15,    # Int to filter for specific maze dimensions, None for no limit
        random_seed = 43):
    
    # Setup
    random.seed(random_seed)
    obs_no_cheese_list = []
    obs_cheese_list = []
    level_seeds = []
    num_mouse_pos = 1 if mouse_poss is None else len(mouse_poss)
    num_cheese_pos = 1 if cheese_poss is None else len(cheese_poss)
    try:
        level_seed_it = iter(levels)
        num_levels = len(levels)  # We may end up with less if they don't meet filter criteria
    except TypeError:
        def random_level_seed():
            while True:
                yield random.randint(0, int(1e6))
        level_seed_it = random_level_seed()
        num_levels = levels
    
    # Iterate through level_seeds, skipping all that don't match maze size
    venv_no_cheese_list = []
    venv_cheese_list = []
    with tqdm(total=num_levels) as pbar:
        while len(level_seeds) < num_levels:
    
            # Get the level seed, breaking out of the loop if we run out of levels
            try:
                level_seed = next(level_seed_it)
            except StopIteration:
                break
            
            # Create a venv to be the "no cheese" envs, which needs to have the same number
            # as mouse positions we want to evaluate
            venv_no_cheese = patch_utils.create_venv(num=num_mouse_pos, start_level=level_seed, 
                num_levels=1)
            state_bytes_list = venv_no_cheese.env.callmethod("get_state")
            states = [maze.EnvState(sb) for sb in state_bytes_list]
            
            # Check the size, skipping if not correct
            maze_dim_this = states[0].inner_grid().shape[0]
            if maze_dim is not None and maze_dim_this != maze_dim:
                continue

            # Check the presence of a decision square, skipping if required and not present
            # dec_pos = maze.get_decision_square_from_maze_state

            level_seeds.append(level_seed)
            padding = (states[0].world_dim - maze_dim_this) // 2
                        
            # TODO: speed this way up!  Should be able to just have one or two EnvStates and
            # just save out bytes from these?
                    
            # Position the mouse as needed in all no-cheese envs, and remove the cheese
            state_bytes_new = []
            for idx, mouse_pos in enumerate(mouse_poss):
                state = maze.EnvState(state_bytes_list[idx])
                # Place/remove objects
                remove_cheese_from_state(state)
                state.set_mouse_pos(mouse_pos[1]+padding, mouse_pos[0]+padding)
                # Update state bytes
                state_bytes_new.append(state.state_bytes)
            venv_no_cheese.env.callmethod("set_state", state_bytes_new)

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

            # Store the venvs in case we want them later
            venv_no_cheese_list.append(venv_no_cheese)
            venv_cheese_list.append(venv_cheese)

            pbar.update(1)

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

    # # Same for venvs (with dims stacked as appropriate)
    # venvs_no_cheese = stack(obs_no_cheese, []
    
    return xr.Dataset(dict(no_cheese=obs_no_cheese, cheese=obs_cheese))

def get_logits_and_values_from_obs(obs, hook,
        value_label='embedder.block2.res1.resadd_out',
        action_logits_label='fc_policy_out'):
    # Run all obs through the hooked network and extract values, unstacked
    # (Force all arrays to have all the dims in the obs dataset,
    # for simpler indexing later)
    values = {}
    action_logits = {}
    coords = obs.coords
    for obs_var in obs:
        hook.run_with_input(stack(obs[obs_var]))
        values[obs_var] = \
            make_full_dims(unstack(hook.get_value_by_label(value_label)), coords)
        action_logits[obs_var] = \
            make_full_dims(unstack(hook.get_value_by_label(action_logits_label)), coords)
    return xr.Dataset(action_logits), xr.Dataset(values)
    # hook.run_with_input(stack(obs_cheese))
    # value_cheese = unstack(hook.get_value_by_label(value_label))
    # action_logits_cheese = unstack(hook.get_value_by_label(action_logits_label))

def logits_to_prob_arrows(logits):
    log_probs = F.log_softmax(t.from_numpy(logits.copy()), dim=1)
    probs = Categorical(logits=log_probs).probs.detach().numpy()
    return np.einsum('ba,ad->ba', probs, models.MAZE_ACTION_DELTAS_BY_INDEX)

COSIM_MIN_MATCH = np.cos(np.pi/4.)
def score_vect_cosim(action_logits_targ, action_logits_patched):
    # Stack arrays
    action_logits_targ_stk = stack(action_logits_targ)
    action_logits_patched_stk = stack(action_logits_patched)
    # Turn logits into probs
    arrows_targ = logits_to_prob_arrows(action_logits_targ_stk.values)
    arrows_patched = logits_to_prob_arrows(action_logits_patched_stk.values)
    cosim = np.einsum('bd,bd->b', arrows_targ, arrows_patched) / \
        (np.linalg.norm(arrows_targ, axis=1) * np.linalg.norm(arrows_patched, axis=1))
    cosim_match = cosim >= COSIM_MIN_MATCH
    return unstack(xr.DataArray(
        data=cosim_match, coords={'sbatch': action_logits_targ_stk.coords['sbatch']}))

def run_cheese_patch_test(
        obs, action_logits, values, hook, patch_func, 
        value_label = 'embedder.block2.res1.resadd_out',
        action_logits_label = 'fc_policy_out',
        orig_var = 'cheese',
        targ_var = 'no_cheese',
        score_func = score_vect_cosim):
    num_levels = obs.sizes['level_seed']
    num_mouse_pos = obs.sizes['mouse_pos']
    # Pull out the relevant quantities for patching and evaluation
    obs_to_use = obs[orig_var]
    action_logits_orig = action_logits[orig_var]
    action_logits_targ = action_logits[targ_var]
    # Run the network, with patch function
    hook.run_with_input(stack(obs_to_use), 
        patches={value_label: patch_func})
    # Get the (still stacked) patched logits
    action_logits_patched = unstack(hook.get_value_by_label(action_logits_label))
    # Get the scores
    scores = score_vect_cosim(
        action_logits_targ,
        action_logits_patched)
    # Get the baselines scores
    scores_unpatched = score_vect_cosim(
        action_logits_targ,
        action_logits_orig)
    # scores_np[src_level_idx, patch_level_idx, ...] = rearrange(scores,
    #     '(mse chs) -> mse chs', mse=num_mouse_pos)
    # # Save the baseline "no patch" score also
    # scores_nopatch_np[src_level_idx, patch_level_idx, ...] = rearrange(
    #     score_vect_cosim(
    #         stack(action_logits_targ).values,
    #         stack(action_logits_orig).values),
    #     '(mse chs) -> mse chs', mse=num_mouse_pos)

    return xr.Dataset(dict(
        patched =   scores,
        unpatched = scores_unpatched))
    
    # scores = xr.DataArray(
    #     data = scores_np,
    #     dims = ['src_level_seed', 'patch_level_seed', 'mouse_pos', 'cheese_pos'],
    #     coords = dict(
    #         src_level_seed=obs_cheese.indexes['level_seed'].values[:num_src_levels],
    #         patch_level_seed=obs_cheese.indexes['level_seed'].values,
    #         mouse_pos=mouse_pos_coords,
    #         cheese_pos=cheese_pos_coords))
    # scores_nopatch = scores.copy(data=scores_nopatch_np)

def binary_scores_table(scores):
    df = pd.DataFrame(dict(
        unpatched_score = scores.unpatched.values.flatten(),
        patched_score = scores.patched.values.flatten()))
    df['dummy'] = 1
    df = df.groupby(['unpatched_score','patched_score']).count().reset_index().pivot(
        index='unpatched_score', columns='patched_score', values='dummy')
    array = xr.Dataset.from_dataframe(df).to_array().rename(variable='patched_score').T
    return df, array


# %%
# Get the observations to use for patching

obs = get_obs_batch_on_grid(
    levels =  30,
    cheese_poss = [(2, 2), (2, 12), (12, 2), (12, 12)],
    mouse_poss =  [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10)],
    maze_dim =    15,
    random_seed=  43
)

action_logits, values = get_logits_and_values_from_obs(obs, hook,
    value_label = 'embedder.block2.res1.resadd_out')

# Variables that are useful for multiple experiments
patch_more_cheese = False
cheese_diff = values['cheese'] - values['no_cheese']

# # Pick some cheese positions to test (e.g. corners, and points in a 
# # square closer to middle?)  (In inner_grid coords)
# cheese_poss = [(2, 2), (2, 12), (12, 2), (12, 12)]
# #cheese_poss = [(2, 12)]
# num_cheese_pos = len(cheese_poss)

# # Pick some mouse positions to test (all true cells that are known 
# # to be open, and which could be branch squares?)
# mouse_poss = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10)]
# #mouse_poss = [(8, 8)]
# num_mouse_pos = len(mouse_poss)


# %%
# Do "different level, same mouse/cheese pos" patching

num_src_levels = 1

# Iterate over source levels
scores_array_list = []
for src_level_idx in range(num_src_levels):
    # Create the patch function based on the cheese diffs for this level
    cheese_diff_this = cheese_diff.isel(level_seed=[src_level_idx])
    cheese_diff_this_t = t.from_numpy(cheese_diff_this.values)
    def patch_func(outp):
        outp_unstack = rearrange(outp, '(lev mse chs) ... -> lev mse chs ...', 
            lev = cheese_diff.sizes['level_seed'],
            mse = cheese_diff.sizes['mouse_pos']) \
                + (1. if patch_more_cheese else -1.)*cheese_diff_this_t
        return rearrange(outp_unstack, 'lev mse chs ... -> (lev mse chs) ...')

        #return outp 
    # def patch_func(outp):
    #     #return t.from_numpy(
    #     #    stack(value_cheese.isel(level_seed=src_level_idx)).values)
    #     #return t.zeros_like(outp)
    #     #return outp
    #     return outp + (1. if patch_more_cheese else -1.)*cheese_diff_this_stk_t

    scores = run_cheese_patch_test(obs, action_logits, values, hook, patch_func=patch_func,
        orig_var = 'no_cheese' if patch_more_cheese else 'cheese',
        targ_var = 'cheese' if patch_more_cheese else 'no_cheese')
    scores_table, scores_array = binary_scores_table(scores)
    scores_array_list.append(scores_array)

    print('Source level seed: {}'.format(obs.indexes['level_seed'][src_level_idx]))
    display(scores_table/scores.patched.size)
    # display(scores)
    # px.histogram((scores.patched > scores.unpatched).values.flatten()).show()

scores_all = xr.concat(scores_array_list, dim='src_level_seed').assign_coords(
    src_level_seed = cheese_diff.indexes['level_seed'].values[:num_src_levels])
print(f'Average fractions of score combinations over {num_src_levels} source levels')
display(scores_all.mean(dim='src_level_seed').to_dataframe(name='scores')/scores.patched.size)


# %%
# Try mouse position (0, 0), patch to other positions within same level

# Create the patch function based on the cheese diffs for this level
cheese_diff_this = cheese_diff.isel(mouse_pos=[0])
cheese_diff_this_t = t.from_numpy(cheese_diff_this.values)
def patch_func(outp):
    outp_unstack = rearrange(outp, '(lev mse chs) ... -> lev mse chs ...', 
        lev = cheese_diff.sizes['level_seed'],
        mse = cheese_diff.sizes['mouse_pos']) \
            + (1. if patch_more_cheese else -1.)*cheese_diff_this_t
    return rearrange(outp_unstack, 'lev mse chs ... -> (lev mse chs) ...') 

scores = run_cheese_patch_test(obs, action_logits, values, hook, patch_func=patch_func,
    orig_var = 'no_cheese' if patch_more_cheese else 'cheese',
    targ_var = 'cheese' if patch_more_cheese else 'no_cheese')
scores_table, scores_array = binary_scores_table(scores)

display(scores_table/scores.patched.size)

# %%
# Test this with some vfield viz
level_seed = scores.indexes['level_seed'][0]
cheese_pos = scores.indexes['cheese_pos'][1]
























# # Iterate over source levels
# scores_np = np.zeros((num_src_levels, num_levels, num_mouse_pos, num_cheese_pos))
# scores_nopatch_np = np.zeros_like(scores_np)
# for src_level_idx in range(num_src_levels):
#     # Grab the cheese diffs for this source level
#     cheese_diff_this = cheese_diff.isel(level_seed=src_level_idx)

# cheese_diff_this_stk_t = t.from_numpy(
#     stack(cheese_diff_this).values)
# def patch_func(outp):
#     #return t.from_numpy(
#     #    stack(value_cheese.isel(level_seed=src_level_idx)).values)
#     #return t.zeros_like(outp)
#     #return outp
#     return outp + (1. if patch_more_cheese else -1.)*cheese_diff_this_stk_t




# display((scores - scores_nopatch).mean().item())

# px.histogram((scores - scores_nopatch).values.flatten()).show()

# # display('No patch baseline', scores_nopatch_np.squeeze())
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
