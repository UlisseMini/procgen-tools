# %%[markdown]
# 
# Tool to experiment with visualizing presums (i.e. contributions through a linear or convolutional layer from the previous layer before the final summation) to understand how a particular activation or channel is formed from the prior layer for a given input

# %%
# Imports
%reload_ext autoreload
%autoreload 2
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy as sp
import torch as t
import torch.nn as nn
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

def check_and_run_with_input(hook, obs, module_label, module_cls):
    # Get module and check type
    module = hook.modules_by_label[module_label]
    assert any([isinstance(module, cls) for cls in module_cls]), \
        f'Expecting module to be instance of one of {str(module_cls)}'
    
    # Execute forward call on obs, making sure it has a singleton leading dim if needed (batch of 1)
    if obs.shape[0] > 1:
        obs = rearrange(obs, '... -> 1 ...')
    hook.run_with_input(obs.astype(np.float32))

    return hook.get_module_and_data_by_label(module_label)


def viz_presums(hook, obs, module_label, channel_index, K=4,
        score_func=lambda ps: ps.max(axis=-1).max(axis=-1)):
    # Get the module and module data
    module, module_data = check_and_run_with_input(hook, obs, module_label, [nn.Conv2d])

    # Grab required activation values:
    #  The specified channel from the module output
    out_ch = module_data.output[0,channel_index,...].detach().numpy()
    #  The full prior-layer values
    inp = module_data.inputs[0][0,...].detach().numpy()
    #  The conv weights and biases for this channel
    weight = module_data.custom_data['weight'][channel_index,...].detach().numpy()
    bias = module_data.custom_data['bias'][channel_index].detach().numpy()
    #  The presums for this channel (already calculated in hook if batch size is 1)
    presum = module_data.custom_data['presum'][0,channel_index,...].detach().numpy()
    
    # Score each input channel using the presum and provided score func, so we can sort,
    # and pick the top-K
    score = score_func(presum)
    best_score_inds = score.argsort()[-K:] #[::-1][:K]
    
    # Visualize the final output, the sum of the top-K presums
    topK_sum = presum[best_score_inds].sum(axis=0) + bias
    fig = px.imshow(np.concatenate([out_ch[...,np.newaxis], topK_sum[...,np.newaxis]], axis=-1),
        facet_col=2)
    fig.layout.annotations[0].update(text='Channel value')
    fig.layout.annotations[1].update(text=f'Sum of top-{K} pre-sums + bias')
    fig.show()

    # Visualize the top-K presums inputs, their weights, and the presums
    subplot_titles = []
    for ch_ind in best_score_inds:
        subplot_titles.extend([f'Channel {ch_ind} value', f'Channel {ch_ind} kernel',
            f'Channel {ch_ind} pre-sum'])
    fig = py.subplots.make_subplots(rows=K, cols=3, subplot_titles=subplot_titles)
    # Plot all the lines
    OPACITY = 0.1
    for kk, ch_ind in enumerate(best_score_inds):
        row = kk+1
        fig.add_trace(go.Heatmap(z=inp[ch_ind,...]), row=row, col=1)
        fig.add_trace(go.Heatmap(z=weight[ch_ind,...]), row=row, col=2)
        fig.add_trace(go.Heatmap(z=presum[ch_ind,...]), row=row, col=3)
        #fig.update_layout({f'yaxis{row_str}_scaleanchor':"x2"})
    fig.update_layout(height=200*K)
    fig.show()


    


def viz_resadd(hook, obs, module_label, channel_index, include_relu=True):
    # Get the module and module data
    module, module_data = check_and_run_with_input(hook, obs, module_label, [models.ResidualAdd])

    # Stack the various quatities to show in a list for convenience
    # (The res stream (second input in the InterpretableResidualBlock), 
    # the new input, the output, and the following Relu output, if requested)
    all_values_list = [
        module_data.inputs[1][0,[channel_index],...], 
        module_data.inputs[0][0,[channel_index],...],
        module_data.output[0,[channel_index],...]]
    value_names = ['res stream', 'new input', 'new res stream']

    # Get the output of the succeeding ReLU, if requested
    if include_relu:
        successor_value_labels = hook.get_successors(module_label)
        successor_module_labels = []
        for value_label in successor_value_labels:
            successor_module_labels.extend(hook.get_successors(value_label))
        successor_modules_and_data = [hook.get_module_and_data_by_label(label) 
            for label in successor_module_labels 
                if label in hook.modules_by_label]
        successor_relus = [(mod, data) for mod, data in successor_modules_and_data 
            if isinstance(mod, nn.ReLU)]
        if len(successor_relus) == 0:
            warnings.warn(f'No ReLU successors found for module {module_label}')
        elif len(successor_relus) > 1:
            warnings.warn(f'Multiple ReLU successors found for module {module_label}')
        else:
            all_values_list.append(successor_relus[0][1].output[0,[channel_index],...])
            value_names.append('new stream ReLU')

    # Visualize the res stream (second input in the InterpretableResidualBlock), 
    # the new input, the output, and the following Relu output, if requested
    all_values = np.concatenate(all_values_list)
    all_values_da = xr.DataArray(data = all_values,
        dims=['v', 'h', 'w']).transpose('h', 'w', 'v')
    fig = px.imshow(all_values_da, facet_col='v', aspect='equal')
    new_titles =["Metrics: a", "Metrics: b", "Metrics: c"]
    for ii, name in enumerate(value_names):
        fig.layout.annotations[ii].update(text=name)
    fig.show()

# %%    
# Test!
if __name__ == "__main__":
    # Load an env and get an observation
    venv = maze.create_venv(num=1, start_level=10, num_levels=0)
    obs = venv.reset()

    # Load and hook a model
    model_file = '../trained_models/maze_I/model_rand_region_5.pth'
    policy = models.load_policy(model_file, action_size=15, device=t.device('cpu'))
    hook = cmh.ModuleHook(policy)

    # Test the viz functions
    viz_presums(hook, obs, 'embedder.block2.conv', 10)
    viz_resadd(hook, obs, 'embedder.block2.res1.resadd', 100)