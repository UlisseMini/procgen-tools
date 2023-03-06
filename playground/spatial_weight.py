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
import procgen_tools.models as models
import procgen_tools.maze as maze
import procgen_tools.patch_utils as patch_utils
import procgen_tools.vfield as vfield
from procgen import ProcgenGym3Env

path_prefix = '../'

# %%
# Load policy
rand_region = 5
policy_normal = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, t.device('cpu'))


# %% 
# Get a bunch of random observations to test various things on
obs = maze.get_random_obs(200, show_pbar=True)
obs_t = t.from_numpy(obs.astype(np.float32))


# %%
# Viz flatten-to-fc weights
weight = policy_normal.embedder.fc.weight.detach().clone()
weight_unflat = rearrange(weight, 'd1 (c h w) -> d1 c h w', c=128, h=8)
weight_unflat_scl = weight_unflat / (2*2*weight_unflat.std())

weight_unflat_scl[:4,:8].chans(frame_px=1, gutter_px=0, scale=10)

# Try to reduce dimensionality by taking mean abs value across channels
#weight_absmean = t.abs(weight_unflat).mean(axis=1)


# %%
# Do any of the first layer conv kernels look like flipped versions of eachother?
# RESULT: doesn't look promising!

weight = policy_normal.embedder.block1.conv.weight.detach().clone()
weight.shape

ch_pairs = []
for c in range(weight.shape[0]):
    knl_flip = t.flip(weight[c,0,:,:], dims=(1,))
    c_best = np.abs(knl_flip - weight[:,0,:,:]).mean(axis=-1).mean(axis=-1).argmin()
    print(c, c_best)
    ch_pairs.append(weight[[c,c_best],0,:,:])
px.imshow(rearrange(ch_pairs, '(c1 c2) ob h w -> (c1 h) (c2 ob w)', c1=8))


# %%
# Try digitizing first conv layer kernels to see how that impacts logits
# RESULT: doesn't seem to work, lots of large changes to logits result from discretizing input weights,
# which is an update towards weights being pretty subtle...

# def digitize_weights(weight, num_bins=9, range_std=3.):
#     # Get bins that cover a certain std-dev range
#     with t.no_grad():
#         std = weight.std()
#         bins = t.linspace(-range_std*std, range_std*std, num_bins+1)
#         bin_inds = t.clip(t.bucketize(weight, bins), 1, num_bins) - 1
#         bin_centers = bins[:-1] + np.diff(bins)/2
#         weight_digi = bin_centers[bin_inds]
#         return weight_digi

# weight = policy_normal.embedder.block1.conv.weight.detach().clone()
# weight_digi = digitize_weights(weight)
# print(t.abs(weight - weight_digi).max())

# policy_disc = copy.deepcopy(policy_normal)
# policy_disc.embedder.block1.conv.weight = nn.Parameter(weight_digi)

# def forward_func_policy(network, inp):
#     hidden = network.embedder(inp)
#     logits = network.fc_policy(hidden)
#     log_probs = F.log_softmax(logits, dim=1)
#     return logits, Categorical(logits=log_probs).probs

# with t.no_grad():
#     logits_normal, probs_normal = forward_func_policy(policy_normal, obs_t)
#     logits_disc, probs_disc = forward_func_policy(policy_disc, obs_t)

# (t.abs(probs_normal - probs_disc).mean(), t.abs(probs_normal - probs_disc).max())
