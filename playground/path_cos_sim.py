# %%
# Imports
%reload_ext autoreload
%autoreload 2

import os
import pickle

import numpy as np
import pandas as pd
import scipy as sp
import torch as t
import torch.nn.functional as f
import xarray as xr
import plotly.express as px
import plotly as py
import plotly.subplots
import plotly.graph_objects as go
from einops import rearrange
from IPython.display import Video, display
from tqdm.auto import tqdm

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr

import procgen_tools.models as models
import procgen_tools.maze as maze

# %%
# Functions

def get_paths_as_world_grids(data_all, state_bytes_key):
    '''Extract paths from provided list of preprocessed maze data objects
    and turn them into boolean grids of the size of world-dims, which
    can be resampled and compared to convolutional layer activations.'''
    path_grids = {'cheese': [], 'corner': []}
    for dd in tqdm(data_all):
        for goal in path_grids.keys():
            path = dd[f'path_to_{goal}']
            env_state = maze.EnvState(dd[state_bytes_key])
            path_grid = np.zeros_like(env_state.full_grid(), dtype=bool)
            offset = tuple(
                [int((sf-si)/2) for sf, si in 
                    zip(path_grid.shape, env_state.inner_grid().shape)])
            for node in path:
                path_grid[node[0]+offset[0], node[1]+offset[1]] = True
            # Flip to match y-axis convention of observation and activations
            path_grids[goal].append(path_grid[::-1,:])
    return xr.Dataset(
        {goal: xr.DataArray(
            data = rearrange(grids, 'b ... -> b ...'),
            dims = ['batch', 'row', 'col'])
                for goal, grids in path_grids.items()}).assign_coords(
                    dict(batch = np.arange(len(data_all))))

def resample_path_grids(path_grids, new_size, mode='bilinear'):
    new_arrays = {}
    for goal in path_grids:
        values_tensor = t.from_numpy(
            rearrange(path_grids[goal].values.astype(np.float32), 'b h w -> b 1 h w'))
        new_arrays[goal] = xr.DataArray(
            data = rearrange(f.interpolate(values_tensor, new_size, 
                mode=mode).detach().numpy(), 'b 1 h w -> b h w'),
            dims = path_grids.dims,
            coords = path_grids.coords)
    return xr.Dataset(new_arrays)


# %%
# Setup / params

# This limits the number of mazes to look at, 
# likely needed to avoid running out of RAM
num_batch = 1000


# %%
# Load postprocessed data and convert to required form, including probe targets

# 10k run, only mazes with dec square, obs saved on dec square
# dr = '../episode_data/20230131T224127/'
# fn = 'postproc_probe_data.pkl'

# 10k run, only mazes with dec square, obs saved on initial square
dr = '../episode_data/20230131T224127/'
fn = 'postproc_probe_data_initial.pkl'

# 1k run, normal full rollouts, cheese/mouse pos already processed out
# dr = '../episode_data/20230131T183918/' 
# fn = 'postproc_batch_of_obs.pkl'

# Load data from post-processed pickled file
with open(os.path.join(dr, fn), 'rb') as fl:
    data_all = pickle.load(fl)['data']
    num_batch_to_use = min(num_batch, len(data_all))
    data_all = data_all[:num_batch_to_use]

# Pull out the observations into a single batch
batch_coords = np.arange(len(data_all))
obs_all = xr.concat([dd['obs'] for dd in data_all], 
    dim='batch').assign_coords(dict(batch=batch_coords))

state_bytes_key = 'init_state_bytes'

path_grids = get_paths_as_world_grids(data_all, state_bytes_key)


# %%
# Set up model and hook it
model_file = '../trained_models/maze_I/model_rand_region_5.pth'
policy = models.load_policy(model_file, action_size=15, device=t.device('cpu'))
model_name = os.path.basename(model_file)
hook = cmh.ModuleHook(policy)

    
# %%
# Run obs through model to get all the activations
_ = hook.run_with_input(obs_all)


# %%
# Pick values to use

# Some resadd outs
# value_labels = ['embedder.block1.conv_in0',
#                 'embedder.block1.res1.resadd_out',
#                 'embedder.block1.res2.resadd_out',
#                 'embedder.block2.res1.resadd_out',
#                 'embedder.block2.res2.resadd_out',
#                 'embedder.block3.res1.resadd_out',
#                 'embedder.block3.res2.resadd_out']

# The layer Peli highlighted on 2023-02-03, and it's relu 
value_labels = ['embedder.block2.res1.conv1_out', 'embedder.block2.res1.relu2_out']

# # All the conv layers!
value_labels = [
    'embedder.block1.conv_out',
    'embedder.block1.maxpool_out',
    'embedder.block1.res1.relu1_out',
    'embedder.block1.res1.conv1_out',
    'embedder.block1.res1.relu2_out',
    'embedder.block1.res1.conv2_out',
    'embedder.block1.res1.resadd_out',
    'embedder.block1.res2.relu1_out',
    'embedder.block1.res2.conv1_out',
    'embedder.block1.res2.relu2_out',
    'embedder.block1.res2.conv2_out',
    'embedder.block1.res2.resadd_out',
    'embedder.block2.conv_out',
    'embedder.block2.maxpool_out',
    'embedder.block2.res1.relu1_out',
    'embedder.block2.res1.conv1_out',
    'embedder.block2.res1.relu2_out',
    'embedder.block2.res1.conv2_out',
    'embedder.block2.res1.resadd_out',
    'embedder.block2.res2.relu1_out',
    'embedder.block2.res2.conv1_out',
    'embedder.block2.res2.relu2_out',
    'embedder.block2.res2.conv2_out',
    'embedder.block2.res2.resadd_out',
    'embedder.block3.conv_out',
    'embedder.block3.maxpool_out',
    'embedder.block3.res1.relu1_out',
    'embedder.block3.res1.conv1_out',
    'embedder.block3.res1.relu2_out',
    'embedder.block3.res1.conv2_out',
    'embedder.block3.res1.resadd_out',
    'embedder.block3.res2.relu1_out',
    'embedder.block3.res2.conv1_out',
    'embedder.block3.res2.relu2_out',
    'embedder.block3.res2.conv2_out',
    'embedder.block3.res2.resadd_out',
    'embedder.relu3_out',
]

# The input observation
# value_labels = ['embedder.block1.conv_in0']

# Final conv layer output
# value_labels = ['embedder.relu3_out']


# %%
# Test similarity between images of paths and convolutional channels

avg_cos_sims_by_value = {}
for value_label in tqdm(value_labels):
    value = hook.get_value_by_label(value_label)
    path_grids_resized = resample_path_grids(path_grids, value.shape[2:], mode='bilinear')
    avg_cos_sims_by_value[value_label] = {}
    for goal in path_grids_resized:
        path_grid_resized_flat_tensor = rearrange(t.from_numpy(
            path_grids_resized[goal].values), 'b h w -> b (h w)')
        avg_cos_sims_this_value = []
        for ch_ind in range(value.shape[1]):
            ch_flat_tensor = rearrange(t.from_numpy(value[:,ch_ind,:,:].values),
                'b h w -> b (h w)')
            cos_sim = f.cosine_similarity(path_grid_resized_flat_tensor,
                ch_flat_tensor).detach().numpy()
            avg_cos_sims_this_value.append(cos_sim.mean())
        avg_cos_sims_by_value[value_label][goal] = np.array(avg_cos_sims_this_value)
        
# %% 
# Process results
results = []
for value_label, avg_cos_sims_by_goal in avg_cos_sims_by_value.items():
    for goal, avg_cos_sims in avg_cos_sims_by_goal.items():
        ind_abs_max = np.abs(avg_cos_sims).argmax()
        results.append(dict(
            value_label=value_label,
            goal=goal,
            ind_abs_max=ind_abs_max,
            avg_cos_sim_abs_max=np.abs(avg_cos_sims)[ind_abs_max],
        ))
results_df = pd.DataFrame(results).set_index(['value_label', 'goal']).sort_values(
    'avg_cos_sim_abs_max', ascending=False)
display(results_df)

# %%
# Look into a specific interesting-looking channel
value_label, ch_ind = ('embedder.block1.res1.resadd_out', 40)

value = hook.get_value_by_label(value_label)

# Look at this channel vs path for a few mazes
for ii, obs in enumerate(obs_all[:3]):
    px.imshow(rearrange(obs.values, 'c h w -> h w c')).show()
    px.imshow(value[ii,ch_ind,:,:]).show()


