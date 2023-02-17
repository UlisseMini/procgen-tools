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

import numpy as np
import numpy.linalg
import pandas as pd
import xarray as xr
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm import tqdm
from einops import rearrange
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
from procgen import ProcgenGym3Env

path_prefix = '../'

# %%
# Generate a large batch of observations, run through hooked network to get fc activations,
# cache these as dataset along with "next cheese action" and "next corner action".

num_obs_normal = 1000
num_obs_dec = 1000

# Get a bunch of obs not necessarily on dec square to get decent navigational basis
obs_normal, obs_meta_normal, next_level_seed = maze.get_random_obs_opts(num_obs_normal, 
    start_level=0, return_metadata=True, random_seed=42, deterministic_levels=True, 
    show_pbar=True)
# Also get a bunch on dec squares to show diversity between cheese/corner actions
obs_dec, obs_meta_dec, next_level_seed = maze.get_random_obs_opts(num_obs_normal, 
    start_level=next_level_seed, return_metadata=True, random_seed=43, deterministic_levels=True, show_pbar=True)
# Merge into a single batch of observations
obs = np.concatenate([obs_normal, obs_dec], axis=0)
obs_meta = obs_meta_normal + obs_meta_dec

# Extract best action for cheese and corner paths
def get_action(curr_pos, next_pos):
    if next_pos[0] < curr_pos[0]: return 'D'
    if next_pos[0] > curr_pos[0]: return 'U'
    if next_pos[1] < curr_pos[1]: return 'L'
    if next_pos[1] > curr_pos[1]: return 'R'
    return 'N'
next_action_cheese = np.array([get_action(md['mouse_pos_outer'], md['next_pos_cheese_outer'])
    for md in obs_meta])
next_action_corner = np.array([get_action(md['mouse_pos_outer'], md['next_pos_corner_outer'])
    for md in obs_meta])

# cache_fn = 'action_probing_obs.pkl'
# if cache_fn

# %%
# Run observations through a hooked network, extract the fc layer activations as the training/test data
value_label = 'embedder.fc_out'

rand_region = 5
policy = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, 
    t.device('cpu'))
hook = cmh.ModuleHook(policy)

hook.run_with_input(obs, values_to_store=[value_label])
value = hook.get_value_by_label(value_label)

# %%
# Train a probe!
probe_result = cpr.linear_probe(value, next_action_cheese, model_type='classifier', C=0.1, test_size=0.3)

print(probe_result['train_score'], probe_result['test_score'])