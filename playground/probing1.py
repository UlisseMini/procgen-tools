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

    # Pull out level seeds for later reference
    level_seeds = np.array([maze.EnvState(dd[state_bytes_key]).state_vals['current_level_seed'].val for
        dd in data_all])

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
# Just plot scores for a single dir for simplicy

dir_ = 'L'
results = probe_results[dir_]
fig = go.Figure()
for nn, index_num in enumerate(index_nums):
    fig.add_trace(go.Scatter(
        y=results['score'].sel(value_label=value_labels, 
            index_num_step=nn).values,
        line=dict(color=get_color(nn)),
        name=f'K={index_num}'))
fig.add_hline(y=1., 
    line_dash="dot", row='all', col='all',
    annotation_text="perfect", 
    annotation_position="bottom right")
fig.add_hline(y=target_probs.sel(dir=dir_).values[()], 
        line_dash="dot", row=row, col=col,
        annotation_text="baseline", 
        annotation_position="bottom right")
fig.update_layout(dict(
    title='Square open/closed prediction accuracy for K-sparse linear probe over all layers',
    font_size=14,
    height=600))
fig.update_xaxes(title_text="layers", tickmode = 'array',
        tickvals = np.arange(len(value_labels)),
        ticktext = value_labels)
fig.update_yaxes(title_text="predict score")
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

# Preprocessing func that zeros out all but the maximum element in a channel
# at each data point in the batch
# This seems to improve cheese probing capability a bit, maybe?  Need more testing
def zero_all_but_max_pixel(value):
    value_flat = rearrange(value.values, 'b h w -> b (h w)')
    max_only_flat = np.zeros_like(value_flat)
    max_inds = np.expand_dims(value_flat.argmax(axis=1), axis=1)
    np.put_along_axis(max_only_flat, max_inds, np.take_along_axis(value_flat, max_inds, 1), 1)
    return value.copy(data=rearrange(max_only_flat, 'b (h w) -> b h w', h=value.shape[1]))

results_df = cpr.linear_probe_single_channels(hook, value_labels, y, 
    #value_preproc_func = zero_all_but_max_pixel,
    model_type='ridge', alpha=30)
print(results_df[['train_score', 'test_score']])


# %%
# Plot results of object location probing

best_mdl = results_df.iloc[0]['model']
best_X = results_df.iloc[0]['X']
y_pred_best_all = best_mdl.predict(best_X)
px.scatter(x=y, y=y_pred_best_all)

# Explore the best regression from a PCA perspective
# px.imshow(rearrange(best_mdl.coef_, '(h w) -> h w', h=16)).show()

# # Plot test score over layers in order, max over channels
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
# Specifically probe manually discovered "channel 55"
ch_ind = 55
X_normal = hook.get_value_by_label('embedder.block2.res1.resadd_out')[:,ch_ind,...].values
X_argmax_only = zero_all_but_max_pixel(hook.get_value_by_label(
    'embedder.block2.res1.resadd_out')[:,ch_ind,...]).values
y = probe_targets['cheese'].sel(pos_axis='x').values.astype(float)
results_normal = cpr.linear_probe(X_normal, y, model_type='ridge', alpha=30)
results_argmax_only = cpr.linear_probe(X_argmax_only, y, model_type='ridge', alpha=0.1)
print('Normal obs', results_normal['train_score'], results_normal['test_score'])
print('Argmax only', results_argmax_only['train_score'], 
    results_argmax_only['test_score'])

X_flat_argmax_only = rearrange(X_argmax_only, 'b ... -> b (...)')
y_pred_argmax_only = results_argmax_only['model'].predict(
    X_flat_argmax_only)
px.scatter(x=y, y=y_pred_argmax_only,
    hover_data={'index': np.arange(len(y))})

# %%
# Try a hackier way for now, just to check this channel: find the col argmax, plot that against
# cheese x-pos, see if they are linearly related and identify any outlier.
ch_ind = 55
value_label = 'embedder.block2.res1.resadd_out'
ch_value = hook.get_value_by_label(value_label)[:,ch_ind,...].values
argmax_title = f'{value_label}:{ch_ind} argmax'
df = pd.concat([
    pd.DataFrame({
        'cheese pos (row/col)': probe_targets['cheese'].sel(pos_axis='x').values.astype(float),
        argmax_title: ch_value.max(axis=1).argmax(-1),
        'batch_index': np.arange(ch_value.shape[0]),
        'axis': 'x'}),
    pd.DataFrame({
        'cheese pos (row/col)': 
            maze.WORLD_DIM-1 - probe_targets['cheese'].sel(pos_axis='y').values.astype(float),
        argmax_title: ch_value.max(axis=2).argmax(-1),
        'batch_index': np.arange(ch_value.shape[0]),
        'axis': 'y'})], axis='index')
px.scatter(df, x='cheese pos (row/col)', y=argmax_title, opacity=0.2, facet_col='axis',
    hover_data=['batch_index']).show()

# %%
# Check a couple specific indices (outliers, etc.)
bis = [409, 583]
for bi in bis:
    env_state = maze.EnvState(data_all[bi][state_bytes_key])
    level_seed = env_state.state_vals['current_level_seed'].val
    fig = py.subplots.make_subplots(rows=1, cols=2,
        subplot_titles = [f'{argmax_title},<br>level_seed:{level_seed}',
                          f'obs, level_seed:{level_seed}'])
    fig.add_trace(go.Heatmap(z=ch_value[bi,...]), row=1, col=1)
    fig.update_yaxes(autorange="reversed")
    fig.add_trace(go.Image(z=rearrange(obs_all[bi,...].values, 'c h w -> h w c')*255.), row=1, col=2)
    # px.imshow(, title=).show()
    # px.imshow(rearrange(obs_all[bi,...].values, 'c h w -> h w c'),
    #     title=).show()
    fig.show()


# %%
# Test for cheesy/mousey channels using f-statistic?

def grid_coord_to_value_ind(full_grid_coord, value_size):
    return np.round(full_grid_coord * value_size / maze.WORLD_DIM).astype(int)

def get_obj_pos_data(value_label, object_name):
    rng = np.random.default_rng(15)
    # Grab a bunch of cheese pixels locations
    # TODO: vectoriez this!    
    value = hook.get_value_by_label(value_label).values
    value_size = value.shape[-1]
    num_pixels = num_batch * 2
    pixels = np.zeros((num_pixels, value.shape[1]))
    is_obj = np.zeros(num_pixels, dtype=bool)
    rows_in_value = np.zeros(num_pixels, dtype=int)
    cols_in_value = np.zeros(num_pixels, dtype=int)
    for bb in tqdm(range(obs_all.shape[0])):
        # Cheese location (transform from full grid row/col to row/col in this value)
        obj_pos_value = (grid_coord_to_value_ind(
                maze.WORLD_DIM-1 - probe_targets[object_name].sel(pos_axis='y')[bb].item(), value_size),
            grid_coord_to_value_ind(probe_targets[object_name].sel(pos_axis='x')[bb].item(), value_size))
        pixels[bb,:] = value[bb,:,obj_pos_value[0],obj_pos_value[1]]
        is_obj[bb] = True
        rows_in_value[bb] = obj_pos_value[0]
        cols_in_value[bb] = obj_pos_value[1]
        # Random pixel that isn't the obect location
        bb_rand = bb + num_batch
        random_pos = obj_pos_value
        while random_pos == obj_pos_value:
            random_pos = (rng.integers(value_size), rng.integers(value_size))
        pixels[bb_rand,:] = value[bb,:,random_pos[0],random_pos[1]]
        is_obj[bb_rand] = False
        rows_in_value[bb_rand] = random_pos[0]
        cols_in_value[bb_rand] = random_pos[1]
    return pixels, is_obj, rows_in_value, cols_in_value

def show_pixel_histogram(target, target_name, ch_ind):
    df = pd.DataFrame({'pixel_value': pixels[:,ch_ind], target_name: target, 
        'level_seed': np.concatenate([level_seeds, level_seeds]),
        'row_in_value': rows_in_value,
        'col_in_value': cols_in_value,})
    px.histogram(df, title=f'{value_label}:{ch_ind}<br>Pixel values at {target_name} and not-{target_name} locations',
        x='pixel_value', color=target_name, opacity=0.5, 
        barmode='overlay',
        histnorm='probability', marginal='box', 
        hover_data=list(df.columns)).show()

def show_f_test_results(pixels, target, target_name):
    f_test, _ = cpr.f_classif_fixed(pixels, target)
    f_test_df = pd.Series(f_test).sort_values(ascending=False)

    fig = px.line(y=f_test_df, title=f'Sorted {target_name} f-test scores for channels of<br>{value_label}',
        hover_data={'channel': f_test_df.index})
    fig.update_layout(
        xaxis_title="channel rank",
        yaxis_title="f-test score",)
    fig.show()

    print(list(f_test_df.index[:20]))

    for ch_ind in f_test_df.index[:2]:
        show_pixel_histogram(target, target_name, ch_ind)

value_label = 'embedder.block2.res1.resadd_out'

for object_name in ['cheese']: #['cheese', 'mouse']:
    pixels, is_obj, rows_in_value, cols_in_value = get_obj_pos_data(value_label,
        object_name)
    show_f_test_results(pixels, is_obj, object_name)


# %% 
# Linear probe!
# results = cpr.linear_probe(pixels[:,f_test_df.index[0]], is_cheese)
# print(results['train_score'])
# print(results['test_score'])


# %%
# What about cheese diff vector, does the scale of these diffs concentrate 
# within the most "cheesy" channels according to the f-test?



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



        


