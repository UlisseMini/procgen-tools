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
from tqdm.auto import tqdm
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

num_obs_normal = 5000
num_obs_dec = 5000

batch_size = 100
value_label = 'embedder.fc_out'
logits_value_label = 'fc_policy_out'

REDO_OBS = False
cache_fn = 'action_probing_obs.pkl'

rand_region = 5
policy = models.load_policy(path_prefix + 
        f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 
    15, t.device('cpu'))
hook = cmh.ModuleHook(policy)

if not os.path.isfile(cache_fn) or REDO_OBS:
    # Get a bunch of obs not necessarily on dec square to get decent navigational basis
    obs_normal, obs_meta_normal, next_level_seed = maze.get_random_obs_opts(
        num_obs_normal, 
        start_level=0, return_metadata=True, random_seed=42, deterministic_levels=True, 
        show_pbar=True)
    # Also get a bunch on dec squares to show diversity between cheese/corner actions
    obs_dec, obs_meta_dec, next_level_seed = maze.get_random_obs_opts(
        num_obs_normal, 
        start_level=next_level_seed, return_metadata=True, random_seed=43, deterministic_levels=True, show_pbar=True,
        must_be_dec_square=True)
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
    next_action_cheese = np.array([get_action(md['mouse_pos_outer'], 
            md['next_pos_cheese_outer'])
        for md in obs_meta])
    next_action_corner = np.array([get_action(md['mouse_pos_outer'], 
            md['next_pos_corner_outer'])
        for md in obs_meta])

    # Run observations through a hooked network, extract the fc layer activations 
    # as the training/test data.  Do it batches to avoid running out of RAM!
    value_list = []
    logits_list = []
    for batch_start_ind in tqdm(range(0, obs.shape[0], batch_size)):
        hook.run_with_input(obs[batch_start_ind:(batch_start_ind+batch_size)], 
            values_to_store=[value_label, logits_value_label])
        value_list.append(hook.get_value_by_label(value_label))
        logits_list.append(hook.get_value_by_label(logits_value_label))
    value = np.concatenate(value_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)
    
    with open(cache_fn, 'wb') as fl:
        pickle.dump((obs, value, logits, next_action_cheese, next_action_corner), fl)

else:
    with open(cache_fn, 'rb') as fl:
        obs, value, logits, next_action_cheese, next_action_corner = pickle.load(fl)

# %%
# Train a probe!
probe_result = cpr.linear_probe(value, next_action_cheese, model_type='classifier', 
    C=1, test_size=0.3)

print(probe_result['train_score'], probe_result['test_score'])
print(probe_result['conf_matrix'])

# %%
# See how the probe compares with the actual best actions chosen by the real network logits
logits_argmax = logits.argmax(axis=1)
next_action_logits = models.MAZE_ACTIONS_BY_INDEX[logits_argmax].astype('<U1')
logits_cheese_score = (next_action_logits == next_action_cheese).mean()
print(logits_cheese_score)

# %%
# Test an agent with the trained cheese-action weights?
# RESULT: so far, the resulting policy performs quite badly, which is suprising as
# it predicts the correct "next action towards cheese" better than the actual policy!
# I think this is worth some debugging...

level = 5
random_seed = 42
rng = np.random.default_rng(random_seed)

model = probe_result['model']

def predict(obs, deterministic):
    obs = t.FloatTensor(obs)
    with hook.store_specific_values([value_label]):
        hook.network(obs)
        fc = hook.get_value_by_label(value_label)
    probs = np.squeeze(model.predict_proba(fc))
    if deterministic:
        act_sh_ind = probs.argmax()
    else:
        act_sh_ind = rng.choice(4, 1, p=probs)
    act_sh = model.classes_[act_sh_ind][0]
    for act_name, inds in models.MAZE_ACTION_INDICES.items():
        if act_name[0] == act_sh:
            act = np.array([inds[0]])
            break
    return act, None

venv = maze.create_venv(1, start_level=level, num_levels=1)
seq, _, _ = cro.run_rollout(predict, venv, max_episodes=1, max_steps=256)
vid_fn, fps = cro.make_video_from_renders(seq.renders)
display(Video(vid_fn, embed=True))