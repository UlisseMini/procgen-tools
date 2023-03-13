# %%[markdown]
# # Training Probes to Predict Next Cheese Action
# 
# What would happen if we essentially re-trained the final fc-to-logits weights using a linear probe to predict next "cheese-direction" action?  Would this work?  
# 
# Start with the usual imports...


# %%
# Imports and initial setup
%reload_ext autoreload
%autoreload 2

from typing import List, Tuple, Dict, Union, Optional, Callable
import random
import itertools
import copy
import pickle
import os
import glob
from math import prod

import numpy as np
import numpy.linalg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, NMF
import xarray as xr
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm.auto import tqdm
from einops import rearrange, repeat
from IPython.display import Video, display, clear_output
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array, vfx
import matplotlib.pyplot as plt 
plt.ioff() # disable interactive plotting, so that we can control where figures show up when refreshed by an ipywidget

import lovely_tensors as lt
lt.monkey_patch()

import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr
import procgen_tools.models as models
import procgen_tools.maze as maze
import procgen_tools.patch_utils as patch_utils
import procgen_tools.vfield as vfield
import procgen_tools.rollout_utils as rollout_utils
from procgen import ProcgenGym3Env

from action_probing_obsproc import load_value

# Hack to make sure cwd is the script folder
os.chdir(globals()['_dh'][0])

path_prefix = '../'

# %%
# Load data and hook module

obs_cache_fn = 'action_probing_obs.pkl'
value_cache_dr = 'action_probing_proc_20230309T074726'

logits_label = 'fc_policy_out'

# A hack because storing activations in middle layers took too much RAM with full dataset
num_obs_to_ignore = 15000

with open(obs_cache_fn, 'rb') as fl:
    obs, obs_meta, next_action_cheese, next_action_corner = pickle.load(fl)
    obs = obs[num_obs_to_ignore:]
    obs_meta = np.array(obs_meta[num_obs_to_ignore:])
    next_action_cheese = next_action_cheese[num_obs_to_ignore:]
    next_action_corner = next_action_corner[num_obs_to_ignore:]

# rand_region = 5
# policy = models.load_policy(path_prefix + 
#         f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 
#     15, t.device('cpu'))
# hook = cmh.ModuleHook(policy)

value_labels = [
    'embedder.block1.conv_in0',
    # Block 1 activations are really big?
    # 'embedder.block1.res1.resadd_out',
    # 'embedder.block1.res2.resadd_out',
    'embedder.block2.res1.resadd_out',
    'embedder.block2.res2.resadd_out',
    'embedder.block3.res1.resadd_out',
    'embedder.block3.res2.resadd_out',
    #'embedder.relufc_out',
]

logits = load_value(logits_label, value_cache_dr)

logits_argmax = logits.argmax(axis=1)
next_action_logits = models.MAZE_ACTIONS_BY_INDEX[logits_argmax].astype('<U1')


# %%
# Train sparse probes on different layers

#index_nums = np.array([10, 100, 300, 600, 1000])
index_nums = np.array([10, 100, 1000])

target = next_action_cheese

scores_list = []
for value_label in tqdm(value_labels):
    value_flat = rearrange(load_value(value_label, value_cache_dr), 'b ... -> b (...)')
    f_test, _ = cpr.f_classif_fixed(value_flat, target)
    sort_inds = np.argsort(f_test)[::-1]
    for K in tqdm(index_nums):
        results = cpr.linear_probe(value_flat[:,sort_inds[:K]], target, C=0.01, 
                                class_weight='balanced', random_state=42)
        scores_list.append({'value': value_label, 'K': K, 'train_score': 
            results['train_score'], 'test_score': results['test_score']})
scores_df = pd.DataFrame(scores_list)
scores_df

# %%
# What about using sparse channels?
chan_nums = np.array([5, 10, 20]) #np.array([5, 10, 15, 20])

target = next_action_cheese

scores_list = []
for value_label in tqdm(value_labels):
    value = load_value(value_label, value_cache_dr)
    value_flat = value.view()
    value_flat.shape = (value.shape[0], prod(value.shape[1:]))
    f_test, _ = cpr.f_classif_fixed(value_flat, target)
    f_test.shape = value.shape[1:]
    f_test_sum_by_chan = f_test.sum(axis=-1).sum(axis=-1)
    chan_sort_inds = np.argsort(f_test_sum_by_chan)[::-1]
    for ch_num in tqdm(chan_nums):
        results = cpr.linear_probe(value[:,chan_sort_inds[:ch_num],:,:], target, C=0.001, 
                                class_weight='balanced', random_state=42)
        scores_list.append({'value': value_label, 'ch_num': ch_num, 'train_score': 
            results['train_score'], 'test_score': results['test_score']})
scores_df = pd.DataFrame(scores_list)
scores_df

# %%
# What about PCA/NMF as an interp tool to autoencode conv channels into a smaller set of channel features?
value_label = 'embedder.block1.res2.resadd_out'
value = load_value(value_label, value_cache_dr)

# Turn value into reasonably-sized data set by flattening a random sample of the value batch inds
num_batch_samples = 100
random_seed = 42
rng = np.random.default_rng(random_seed)
batch_inds = rng.integers(0, value.shape[0], num_batch_samples)
X = rearrange(value[batch_inds,...], 
    'b c h w -> (b h w) c')

# Try a decomposition or two
pca = PCA(n_components=20)
pca.fit(X)
print(pca.explained_variance_ratio_)

nmf_p = NMF(n_components=20, random_state=random_seed)
nmf_p.fit(np.maximum(X,0.))
print(nmf_p.reconstruction_err_/X.size)

nmf_n = NMF(n_components=20, random_state=random_seed)
nmf_n.fit(np.maximum(-X,0.))
print(nmf_n.reconstruction_err_/X.size)


# %%
# Show results on an example data point
def show_rdu_res(obs, value, components, bi, title):
    obs_mono = np.zeros_like(obs[bi])
    obs_mono[1,...] = obs[bi].mean(axis=0)
    exp_factor = obs.shape[-1]//value.shape[-1]
    value_rdu_exp = repeat(np.einsum('chw,pc->hwp', value[bi,...], components),
        'h w p -> (h h2) (w w2) p', h2=exp_factor, w2=exp_factor)
    value_rdu_exp = (value_rdu_exp - value_rdu_exp.min())/(value_rdu_exp.max()-value_rdu_exp.min())
    obs_value_rdu = repeat(obs_mono, 'c h w -> c h w p', p=components.shape[0])
    obs_value_rdu[0,...] = value_rdu_exp
    obs_value_rdu = rearrange(obs_value_rdu, 'c h w p -> h w c p')
    # #px.imshow(rearrange(obs[bi], 'c h w -> h w c')).show()
    fig = px.imshow(obs_value_rdu, facet_col=3, facet_col_wrap=5, title=title)
    fig.update_layout(height=1000)
    fig.show()
    #return obs_value_rdu

show_rdu_res(obs, value, pca.components_, batch_inds[0], 'PCA')
show_rdu_res(obs, value, nmf_p.components_, batch_inds[0], 'NMF(p)')
show_rdu_res(obs, value, nmf_n.components_, batch_inds[0], 'NMF(n)')