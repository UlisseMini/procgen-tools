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
import warnings

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr

import procgen_tools.models as models
import procgen_tools.maze as maze

warnings.filterwarnings("ignore", message=r'.*labels with no predicted samples.*')

# %%
# Functions

def get_node_probe_targets(data_all, world_loc, state_bytes_key):
    '''Get potential probe targets based on maze node information for 
    the maze square located at world_loc, which should be (row, col).
    Targets returned are node type (wall, path, branch, etc.) and
    neighbour "is open" status for left, right, down, up.'''
    node_data = []
    for dd in tqdm(data_all):
        node_data.append(
            maze.get_node_type_by_world_loc(dd[state_bytes_key], world_loc))
    node_types = [tt for tt, ngb in node_data]
    node_ngb = [ngb for tt, ngb in node_data]
    return xr.Dataset(dict(
        node_type = xr.DataArray(node_types, dims=['batch']),
        node_ngb = xr.DataArray(node_ngb, dims=['batch', 'dir']),
    )).assign_coords(dict(
        batch = np.arange(len(data_all)),
        dir = ['L', 'R', 'D', 'U']))

def get_obj_loc_targets(data_all, obj_value, state_bytes_key):
    '''Get potential probe targets for y,x (row, col) location of an
    object, where the object is specified by obj_value as the value
    to match on in the maze grid array.'''
    pos_arr = maze.get_object_pos_from_seq_of_states(
        [dd[state_bytes_key] for dd in data_all], obj_value)
    pos = xr.DataArray(
        data = pos_arr,
        dims = ['batch', 'pos_axis'],
        coords = {'batch': np.arange(len(data_all)), 'pos_axis': ['y', 'x']})
    return pos

# This won't be as clean as it seemed since initial node changes based on maze dim,
# need to filter data to constant maze size
# def get_first_path_step_targets(data_all, state_bytes_key):
#     next_node_is_up = np.array((len(data_all), 2), dtype=bool)
#     for bb, dd in enumerate(tqdm(data_all)):
#         for ii, node in enumerate([dd['path_to_cheese'][1],
#                                    dd['path_to_corner'][1]]):
#             next_node_is_up[bb,ii] = node == 


# %%
# Setup / params

# This limits the number of mazes to look at, 
# likely needed to avoid running out of RAM
num_batch = 1000

# The cell we'll be checking neighbours of as an example probe
maze_loc_to_probe = (12, 12)  # Center cell, inside mazes of any size


# %%
# Load postprocessed data and convert to required form, including probe targets

# 10k run, only mazes with dec square, obs saved on dec square
dr = '../episode_data/20230131T224127/'
fn = 'postproc_probe_data.pkl'

# 10k run, only mazes with dec square, obs saved on initial square
# dr = '../episode_data/20230131T224127/'
# fn = 'postproc_probe_data_initial.pkl'

# 1k run, normal full rollouts, cheese/mouse pos already processed out
# dr = '../episode_data/20230131T183918/' 
# fn = 'postproc_batch_of_obs.pkl'

# Load data from post-processed pickled file
if 'postproc_probe_data' in fn:
    with open(os.path.join(dr, fn), 'rb') as fl:
        data_all = pickle.load(fl)['data']
        num_batch_to_use = min(num_batch, len(data_all))
        data_all = data_all[:num_batch_to_use]

    # Pull out the observations into a single batch
    batch_coords = np.arange(len(data_all))
    obs_all = xr.concat([dd['obs'] for dd in data_all], 
        dim='batch').assign_coords(dict(batch=batch_coords))

    # Did we get the cheese?  
    # probe_targets = xr.Dataset(dict(
    #     did_get_cheese = xr.DataArray([dd['did_get_cheese'] for dd in data_all]),
    # ))

    state_bytes_key = 'init_state_bytes' if 'initial' in fn else 'dec_state_bytes'

    # Neighbour open status of a particular cell
    probe_targets = get_node_probe_targets(data_all, maze_loc_to_probe,
        state_bytes_key)

    # Cheese location
    probe_targets['cheese'] = get_obj_loc_targets(data_all, maze.CHEESE,
        state_bytes_key)

    # Mouse location
    probe_targets['mouse'] = get_obj_loc_targets(data_all, maze.MOUSE,
        state_bytes_key)

    # Next square on path to cheese/corner 
    # ...
    

elif 'batch_of_obs' in fn:
    pass
    # TODO: fix this to work with big batch of obs, likely need to re-think
    # this as the pkl file is 4Gb!!!
    # with open(os.path.join(dr, fn), 'rb') as fl:
    #     data_all = pickle.load(fl)['data']

    # # Pull out data into a single batch, with (obs, cheese_pos, mouse_pos, level_seed)
    # # at each data point
    # batch_coords = np.arange(len(data_all))
    # obs_all = xr.concat([dd['obs'] for dd in data_all], 
    #     dim='batch').assign_coords(dict(batch=batch_coords))

    # # Did we get the cheese?  
    # # probe_targets = xr.Dataset(dict(
    # #     did_get_cheese = xr.DataArray([dd['did_get_cheese'] for dd in data_all]),
    # # ))

    # # Neighbour open status of a particular cell
    # probe_targets = get_node_probe_targets(data_all, maze_loc_to_probe)

    # # Cheese location
    # probe_targets = probe_targets.merge(get_cheese_loc_targets(data_all))

    # # Mouse location
    # probe_targets = probe_targets.merge(get_mouse_loc_targets(data_all))



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
# Pick values to use for probes
# (Adjust to specific values to speed up work when only certain layers
# are of interest)

# Some resadd outs
# value_labels = ['embedder.block1.conv_in0',
#                 'embedder.block1.res1.resadd_out',
#                 'embedder.block1.res2.resadd_out',
#                 'embedder.block2.res1.resadd_out',
#                 'embedder.block2.res2.resadd_out',
#                 'embedder.block3.res1.resadd_out',
#                 'embedder.block3.res2.resadd_out']

# The layer Peli highlighted on 2023-02-03, and it's relu 
# value_labels = ['embedder.block2.res1.conv1_out', 'embedder.block2.res1.relu2_out']

# All the conv layers!
value_labels = [
    'embedder.block1.conv_in0',
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
# Try the circrl probing, probe for left, right, down, up neighbour status
# as a demo of functionality

value_labels_to_plot = [
    'embedder.block1.conv_in0',
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
    probe_results[dir_], _ = cpr.sparse_linear_probe(hook, value_labels, y,
        index_nums = index_nums, max_iter=1000)

# %% 
# Plot scores for each direction and layer, at each number of acts

# Reference baseline for classifier scores
target_probs = probe_targets['node_ngb'].mean(dim='batch')

def get_color(ii):
    rep_ii = len(plotly.colors.DEFAULT_PLOTLY_COLORS)
    return plotly.colors.DEFAULT_PLOTLY_COLORS[ii%rep_ii]

num_values = len(value_labels_to_plot)
num_dirs = len(probe_results)
# Make the figure
fig = py.subplots.make_subplots(rows=num_values, cols=num_dirs, 
    shared_yaxes='all', shared_xaxes='all',
    subplot_titles=[f'{label}<br>dir={dir_}' for label in value_labels_to_plot 
        for dir_ in probe_results.keys()])
# Plot each set of scores
for rr, label in enumerate(value_labels_to_plot):
    for cc, (dir_, results) in enumerate(probe_results.items()):
        row = rr + 1
        col = cc + 1
        # Score
        fig.add_trace(go.Scatter(
            y=results['score'].sel(value_label=label).values,
            x=index_nums,
            line=dict(color=get_color(0))), row=row, col=col)
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

# %%
# Now plot test score over layers in order

# Make the figure
num_cols = 2
num_rows = int(np.ceil(num_dirs/2))
fig = py.subplots.make_subplots(rows=num_rows, cols=num_cols, 
    shared_yaxes='all', shared_xaxes='all',
    subplot_titles=[f'dir={dir_}' for dir_ in probe_results.keys()])
# Plot each set of scores
for ii, (dir_, results) in enumerate(probe_results.items()):
    row = ii // num_cols + 1
    col = ii % num_cols + 1
    for nn, index_num in enumerate(index_nums):
        # Score
        fig.add_trace(go.Scatter(
            y=results['score'].sel(value_label=value_labels, 
                index_num_step=nn).values,
            line=dict(color=get_color(nn))), row=row, col=col)
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
    height=600))
fig.update_xaxes(title_text="values", tickmode = 'array',
        tickvals = np.arange(len(value_labels)),
        ticktext = value_labels)
fig.update_yaxes(title_text="predict score")
fig.show()


# %%
# Probe for location as scalar value (mouse or cheese)

#y = probe_targets['cheese'].sel(pos_axis='x').values.astype(float)
y = probe_targets['mouse'].sel(pos_axis='x').values.astype(float)

results_df = cpr.linear_probe_single_channels(hook, value_labels, y, 
    model_type='ridge', alpha=30)
print(results_df[['train_score', 'test_score']])


# %%
# Plot results of object location probing

# best_mdl = results_df.iloc[0]['model']
# best_X = results_df.iloc[0]['X']
# y_pred_best_all = best_mdl.predict(best_X)
# px.scatter(x=y, y=y_pred_best_all)

# Explore the best regression from a PCA perspective
# px.imshow(rearrange(best_mdl.coef_, '(h w) -> h w', h=16)).show()

# Plot test score over layers in order, max over channels
test_score_max_by_value = results_df['test_score'].groupby('value_label').max()[
    value_labels]
display(test_score_max_by_value)
fig = px.line(test_score_max_by_value)
fig.update_layout(height=600)
fig.update_xaxes(title_text="values", tickmode='array',
    tickvals = np.arange(len(value_labels)),
    ticktext = value_labels)
fig.show()


# %%
# Try with ranked sparse probing, for specific location true/false in region

# obj_x = probe_targets['cheese'].sel(pos_axis='x')
# y = (obj_x >= 9) & (obj_x <= 13)

# # probe_results, scaler = cpr.sparse_linear_probe(hook, value_labels, y,
# #     model_type = 'classifier',
# #     index_nums = [10, 50, 200, 400],
# #     regression_kwargs = dict(max_iter=1000))
# # probe_results['conf_matrix']

# results_df = cpr.linear_probe_single_channels(hook, value_labels, y, 
#     model_type='classifier', max_iter=300, test_size=0.4)
# display(results_df[['train_score', 'test_score']])
# display(results_df.iloc[0]['conf_matrix'])
# y_pred_best_all = results_df.iloc[0]['model'].predict(results_df.iloc[0]['X'])



        


