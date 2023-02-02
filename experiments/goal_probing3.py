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

def get_node_probe_targets(data_all, world_loc):
    node_data = []
    for dd in tqdm(data_all):
        node_data.append(
            maze.get_node_type_by_world_loc(dd['dec_state_bytes'], world_loc))
    node_types = [tt for tt, ngb in node_data]
    node_ngb = [ngb for tt, ngb in node_data]
    return xr.Dataset(dict(
        node_type = xr.DataArray(node_types, dims=['batch']),
        node_ngb = xr.DataArray(node_ngb, dims=['batch', 'dir']),
    )).assign_coords(dict(
        batch = np.arange(len(data_all)),
        dir = ['L', 'R', 'D', 'U']
    ))


# %%
# Setup / params

# This limits the number of mazes to look at, 
# likely needed to avoid running out of RAM
num_batch = 1000

# The cell we'll be checking neighbours of as an example probe
maze_loc_to_probe = (12, 12)  # Center cell, inside mazes of any size


# %%
# Load postprocessed data and convert to required form

# Load data as list of dicts
dr = '../episode_data/20230131T224127/' # 10k run
with open(os.path.join(dr, 'postproc_probe_data.pkl'), 'rb') as fl:
    data_all = pickle.load(fl)['data'][:num_batch]

# Pull out the observations into a single batch
batch_coords = np.arange(len(data_all))
obs_all = xr.concat([dd['obs'] for dd in data_all], 
    dim='batch').assign_coords(dict(batch=batch_coords))


# %%
# Pull out / create probe targets of interest
# Can take a while for a large dataset)

# Did we get the cheese?  
# probe_targets = xr.Dataset(dict(
#     did_get_cheese = xr.DataArray([dd['did_get_cheese'] for dd in data_all]),
# ))

# Neighbour open status of a particular cell
probe_targets = get_node_probe_targets(data_all, maze_loc_to_probe)


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
# Try the circrl probing, probe for left, right, down, up neighbour status

value_labels = ['embedder.block1.conv_in0',
                'embedder.block1.res1.resadd_out',
                'embedder.block1.res2.resadd_out',
                'embedder.block2.res1.resadd_out',
                'embedder.block2.res2.resadd_out',
                'embedder.block3.res1.resadd_out',
                'embedder.block3.res2.resadd_out']

index_nums = np.array([1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50])

probe_results = {}
for dir_ in probe_targets['node_ngb'].coords['dir'].values:
    print(f'Probing to predict whether square to {dir_} is open')
    y = probe_targets['node_ngb'].sel(dir=dir_).values
    probe_results[dir_] = cpr.run_probe(hook, y,
        value_labels = value_labels,
        index_nums = index_nums)

# %% 
# Show results

# Reference baseline for classifier scores
target_probs = probe_targets['node_ngb'].mean(dim='batch')

clr = plotly.colors.DEFAULT_PLOTLY_COLORS[0]

# Plot scores for each direction and layer, at each number of acts
num_values = probe_results['L'].sizes['value_label']
num_dirs = len(probe_results)
# Make the figure
fig = py.subplots.make_subplots(rows=num_values, cols=num_dirs, 
    shared_yaxes='all', shared_xaxes='all',
    subplot_titles=[f'{label}<br>dir={dir_}' for label in value_labels 
        for dir_ in probe_results.keys()])
# Plot each set of scores
for rr, label in enumerate(value_labels):
    for cc, (dir_, results) in enumerate(probe_results.items()):
        row = rr + 1
        col = cc + 1
        # Score
        fig.add_trace(go.Scatter(
            y=results['score'].sel(value_label=label).values,
            x=index_nums,
            line=dict(color=clr)), row=row, col=col)
        # Baseline
        fig.add_hline(y=target_probs.sel(dir=dir_).values[()], 
            line_dash="dot", row=row, col=col,
            annotation_text="baseline", 
            annotation_position="bottom right")
# Tweak figure and show
fig.add_hline(y=1., 
    line_dash="dot", row='all', col='all',
    annotation_text="perfect", 
    annotation_position="bottom right")
fig.update_layout(dict(
    font_size=10,
    showlegend=False,
    height=1000))
fig.update_xaxes(title_text="num acts used")
fig.update_yaxes(title_text="predict score")
fig.update_annotations(font_size=8)
fig.show()
