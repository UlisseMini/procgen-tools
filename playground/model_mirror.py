# %%[markdown]
# # Modifying the Agent's "Top Right Corner" Drive
# 
# Based on the spatial symmetry of the conv net used in the maze environment goal misgeneralizing results, it is possible to make simple modifications to the weights of the policy network to change the "corner seeking" behavior of the agent.  This notebook demonstrates this .
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

import circrl.module_hook as cmh
import circrl.rollouts as cro
import procgen_tools.models as models
import procgen_tools.maze as maze
import procgen_tools.patch_utils as patch_utils
import procgen_tools.vfield as vfield
from procgen import ProcgenGym3Env

path_prefix = '../'

# %%[markdown]
# Define some functions to copy and modify a policy object to change the corner-seeking behavior.  This function makes the following changes to the weights of the network:
# - Flip all conv layer kernels left-right and/or up-down
# - Rearrange order of weights from flatten layer to fully-connected fc layer such that the pixels each weight refers to are flipped left-right and/or up-down
# - Rearrange order of weights and biases in the final linear layer from fc to logits, such that the weight vectors and biases corresponding actions resulting in left/right actions are swapped, and/or same for up/down actions.
#
# The net effect of all this is that the network is fully spatially flipped, and should behave as if it is living in a "mirror world" where it's vision and actions are all flipped across one or both axes, even though in fact the changes are made to the weights of the network instead.

# %%
# Define weight-flipping function

def swap_actions(tens, actions_to_swap):
    act1, act2 = actions_to_swap
    tens_act1 = tens[models.MAZE_ACTION_INDICES[act1]]
    tens[models.MAZE_ACTION_INDICES[act1]] = tens[models.MAZE_ACTION_INDICES[act2]]
    tens[models.MAZE_ACTION_INDICES[act2]] = tens_act1    

def flip_weights_and_actions(policy, flip_x=False, flip_y=False):
    # Copy network
    policy_flipped = copy.deepcopy(policy)

    # Set up dimensions to flip
    dims = ((-2,) if flip_y else ()) + ((-1,) if flip_x else ())

    # First, flip all conv kernels
    for label, mod in policy_flipped.named_modules():
        if isinstance(mod, nn.Conv2d):
            with t.no_grad():
                mod.weight = nn.Parameter(t.flip(mod.weight, dims=dims))

    # Then, the flatten-to-fc weight matrix: flip the pixels that each fc activation uses
    weight_unflat = rearrange(policy_flipped.embedder.fc.weight, 'd1 (c h w) -> d1 c h w',
         c=128, h=8)
    weight_unflat_flip = t.flip(weight_unflat, dims=dims)
    weight_flip = rearrange(weight_unflat_flip, 'd1 c h w -> d1 (c h w)')
    with t.no_grad():
        policy_flipped.embedder.fc.weight = nn.Parameter(weight_flip)

    # Next, the weights and biases of the final logits, to replace the left actions with the right actions
    weight = policy_flipped.fc_policy.weight.detach().clone()
    bias = policy_flipped.fc_policy.bias.detach().clone()
    if flip_x:
        swap_actions(weight, ['LEFT', 'RIGHT'])
        swap_actions(bias, ['LEFT', 'RIGHT'])
    if flip_y:
        swap_actions(weight, ['UP', 'DOWN'])
        swap_actions(bias, ['UP', 'DOWN'])
    with t.no_grad():
        policy_flipped.fc_policy.weight = nn.Parameter(weight)
        policy_flipped.fc_policy.bias = nn.Parameter(bias)
    
    return policy_flipped

# %%[markdown]
# Now we can load a policy (the usual 5x5 rand-region trained model), apply the flipping, and test it out on a maze.
#
# Let's flip left-right, right-left, and both, testing on a normal maze, and the same maze without cheese and starting close to the middle to see the full effect of the corner-seeking mod.

# %%
# Define helper functions and load model

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

# Run rollout and return a video clip
def rollout_video_clip(predict, level, remove_cheese=False, 
        mouse_inner_pos=None,
        mouse_outer_pos=None):
    venv = maze.create_venv(1, start_level=level, num_levels=1)
    # Remove cheese
    if remove_cheese:
        maze.remove_cheese(venv)
    # Place mouse if specified (no error checking)
    env_state = maze.EnvState(venv.env.callmethod('get_state')[0])
    if mouse_inner_pos is not None:
        padding = (env_state.world_dim - env_state.inner_grid().shape[0]) // 2
        mouse_outer_pos = (mouse_pos_inner[0] + padding,
            mouse_pos_inner[1] + padding)
    if mouse_outer_pos is not None:
        env_state.set_mouse_pos(mouse_outer_pos[1], mouse_outer_pos[0])
        venv.env.callmethod('set_state', [env_state.state_bytes])
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

# Run rollouts with multiple predict functions, stack the videos side-by-side and return
def side_by_side_rollout(predicts_dict, level, remove_cheese=False, num_cols=2,
            mouse_inner_pos=None,
            mouse_outer_pos=None):
        policy_descs = list(predicts_dict.keys())
        policy_descs_grid = [policy_descs[x:x+num_cols] for x in 
            range(0, len(policy_descs), num_cols)]
        print(f'Level:{level}, cheese:{not remove_cheese}, policies:{policy_descs_grid}')
        clips = []
        seqs = []
        for desc, predict in predicts_dict.items():
            seq, clip = rollout_video_clip(predict, level, remove_cheese, mouse_inner_pos,
                mouse_outer_pos)
            clips.append(clip)
            seqs.append(seq)
        clips_grid = [clips[x:x+num_cols] for x in range(0, len(clips), num_cols)]
        final_clip = clips_array(clips_grid)
        stacked_fn = 'stacked.mp4'
        final_clip.resize(width=600).write_videofile(stacked_fn, logger=None)
        return(Video(stacked_fn, embed=True))

rand_region = 5
policy_normal = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, t.device('cpu'))
predict_normal = get_predict(policy_normal)

# %%
# Top-left agent (flip on X axis)
level = 13

# Apply flipping to get new model and predict func
policy_dict = {
    'top-left':     flip_weights_and_actions(policy_normal, flip_x=True, flip_y=False),
    'top-right':    flip_weights_and_actions(policy_normal, flip_x=False, flip_y=False),
    'bottom-left':  flip_weights_and_actions(policy_normal, flip_x=True,  flip_y=True),
    'bottom-right': flip_weights_and_actions(policy_normal, flip_x=False, flip_y=True),
}
predict_dict = {desc: get_predict(policy) for desc, policy in policy_dict.items()}

# Test with and without cheese on a specific level, setting the mouse pos in the last run
display(side_by_side_rollout(predict_dict, level, False))
display(side_by_side_rollout(predict_dict, level, True,  mouse_outer_pos=(11, 11)))

# %%[markdown]
# Interestingly, the bottom-seeking agents, especially the bottom-right seeking agent, seems to be generally less capable than the other agents.  I hypothesize that this may be caused by some sensitivity to the mouse/cheese icon flipping (the only non-visually-symmetrical objects in the environment as far as I can tell, with the exception of some minor asymmetries in the background pattern.)
#
# To see if this is driven by visual stuff only and not some mistaken assumption in the weight modification process, we can test the normal agent, with flipped observations and flipped actions only...
