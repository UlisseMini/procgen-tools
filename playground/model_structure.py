
# %%
# Imports
%reload_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import scipy as sp
import torch as t
import xarray as xr
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm.auto import tqdm
from einops import rearrange
from IPython.display import Video, display

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro

import procgen_tools.models as models
import procgen_tools.maze as maze

# %%

# Create a policy with weights=1, bias=0
model = models.InterpretableImpalaModel(in_channels=3)
policy = models.CategoricalPolicy(model, action_size=15)
# Set all weights to 1, biases to 0
with t.no_grad():
    for name, W in policy.named_parameters():
        if 'weight' in name:
            W[...] = 1.
        elif 'bias' in name:
            W[...] = 0.

# Hook the network so we can extract the layer outputs
hook = cmh.ModuleHook(policy)

# Values to check for results
values_to_check = ['embedder.block1.res2.resadd_out',
    'embedder.block2.res2.resadd_out',
    'embedder.block3.res2.resadd_out']
 
# Iterate through rows and cols of the observation input (just one channel), 
# creating a large observation array
# TODO: could vectorize this creation logic!
obs_shape = (3, 64, 64)
n_ch, n_rows, n_cols = obs_shape
obs_full = np.zeros((n_rows, n_cols, n_ch, n_rows, n_cols), dtype=np.float32)
for rr in range(n_rows):
    for cc in range(n_cols):
        obs_full[rr,cc,:,rr,cc] = 1.

obs = rearrange(obs_full, 'r1 c1 ch r2 c2 -> (r1 c1) ch r2 c2')
hook.run_with_input(obs)
results_by_value = {}
for label in values_to_check:
    value = hook.get_value_by_label(label)
    value_shape = value.shape
    results_by_value[label] = rearrange(value[:,0,:,:], 
        '(r1 c1) r3 c3 -> r1 c1 r3 c3', r1=obs_full.shape[0])
