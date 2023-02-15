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

import circrl.module_hook as cmh
import circrl.rollouts as cro
import procgen_tools.models as models
import procgen_tools.maze as maze
import procgen_tools.patch_utils as patch_utils
import procgen_tools.vfield as vfield
from procgen import ProcgenGym3Env

path_prefix = '../'

# %%
# Load model
rand_region = 5
policy_normal = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, t.device('cpu'))
policy_top_left = copy.deepcopy(policy_normal)

# Modify weights!

# First, flip all conv kernels left-to-right
for label, mod in policy_top_left.named_modules():
    if isinstance(mod, nn.Conv2d):
        with t.no_grad():
            mod.weight = nn.Parameter(t.flip(mod.weight, dims=(-1,)))

# Then, the flatten-to-fc weight matrix: flip the pixels that each fc activation uses left-to-right
weight_unflat = rearrange(policy_top_left.embedder.fc.weight, 'd1 (c h w) -> d1 c h w', c=128, h=8)
weight_unflat_flip = t.flip(weight_unflat, dims=(3,))
weight_flip = rearrange(weight_unflat_flip, 'd1 c h w -> d1 (c h w)')
with t.no_grad():
    policy_top_left.embedder.fc.weight = nn.Parameter(weight_flip)

# Next, the weights and biases of the final logits, to replace the left actions with the right actions
def swap_left_right(tens):
    left = tens[models.MAZE_ACTION_INDICES['LEFT']]
    tens[models.MAZE_ACTION_INDICES['LEFT']] = tens[models.MAZE_ACTION_INDICES['RIGHT']]
    tens[models.MAZE_ACTION_INDICES['RIGHT']] = left    
weight = policy_top_left.fc_policy.weight.detach().clone()
swap_left_right(weight)
bias = policy_top_left.fc_policy.bias.detach().clone()
swap_left_right(bias)
with t.no_grad():
    policy_top_left.fc_policy.weight = nn.Parameter(weight)
    policy_top_left.fc_policy.bias = nn.Parameter(bias)


# %%
# Predict func for rollouts
def get_predict(plcy):
    def predict(obs, deterministic):
        #obs = t.flip(t.FloatTensor(obs), dims=(-1,))
        obs = t.FloatTensor(obs)
        last_obs = obs
        dist, value = plcy(obs)
        if deterministic:
            act = dist.mode.numpy()
        else:
            act = dist.sample().numpy()
        return act, None, dist.logits.detach().numpy()
    return predict
predict_normal = get_predict(policy_normal)
predict_top_left = get_predict(policy_top_left)

# Run rollouts with normal and modified networks
def rollout_video_clip(predict, level, remove_cheese=False):
    venv = maze.create_venv(1, start_level=level, num_levels=1)
    # Remove cheese
    if remove_cheese:
        maze.remove_cheese(venv)
    # Rollout
    seq, _, _ = cro.run_rollout(predict, venv, max_episodes=1, max_steps=256)
    vid_fn, fps = cro.make_video_from_renders(seq.renders)
    rollout_clip = VideoFileClip(vid_fn)
    # try:
    #     txt_clip = TextClip("GeeksforGeeks", fontsize = 75, color = 'black') 
    #     txt_clip = txt_clip.set_pos('center').set_duration(10) 
    #     final_clip = CompositeVideoClip([rollout_clip, txt_clip]) 
    # except OSError as e:
    #     print('Cannot add text overlays, maybe ImageMagick is missing?  Try sudo apt install imagemagick')
    #     final_clip = rollout_clip
    final_clip = rollout_clip
    return seq, final_clip

def side_by_side_rollout(level, remove_cheese=False):
    print(f'Level {level}, cheese:{not remove_cheese}, normal policy and "top left" policy')
    seq_normal, clip_normal = rollout_video_clip(predict_normal, level, remove_cheese)
    seq_top_left, clip_top_left = rollout_video_clip(predict_top_left, level, remove_cheese)
    final_clip = clips_array([[clip_normal, clip_top_left]])
    stacked_fn = 'stacked.mp4'
    final_clip.resize(width=600).write_videofile(stacked_fn, logger=None)
    display(Video(stacked_fn, embed=True))

side_by_side_rollout(13, False)
side_by_side_rollout(13, True)