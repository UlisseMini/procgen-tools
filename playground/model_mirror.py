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

import lovely_tensors as lt
lt.monkey_patch()

import circrl.module_hook as cmh
import circrl.rollouts as cro
import procgen_tools.models as models
import procgen_tools.maze as maze
import procgen_tools.patch_utils as patch_utils
import procgen_tools.vfield as vfield
import procgen_tools.rollout_utils as rollout_utils
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
# Load model
rand_region = 5
policy_normal = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, t.device('cpu'))
predict_normal = rollout_utils.get_predict(policy_normal)

# %%
# All agents
level = 13

# Apply flipping to get new model and predict func
policy_dict = {
    'top-left':     flip_weights_and_actions(policy_normal, flip_x=True, flip_y=False),
    'top-right':    flip_weights_and_actions(policy_normal, flip_x=False, flip_y=False),
    'bottom-left':  flip_weights_and_actions(policy_normal, flip_x=True,  flip_y=True),
    'bottom-right': flip_weights_and_actions(policy_normal, flip_x=False, flip_y=True),
}
predict_dict = {desc: rollout_utils.get_predict(policy) for desc, policy in policy_dict.items()}

# Test with and without cheese on a specific level, setting the mouse pos in the last run
display(rollout_utils.side_by_side_rollout(predict_dict, level, False))
display(rollout_utils.side_by_side_rollout(predict_dict, level, True,  mouse_outer_pos=(11, 11)))

# %%[markdown]
# Interestingly, the bottom-seeking agents, especially the bottom-right seeking agent, seems to be generally less capable than the other agents.  I hypothesize that this may be caused by some sensitivity to the mouse/cheese icon flipping (the only non-visually-symmetrical objects in the environment as far as I can tell, with the exception of some minor asymmetries in the background pattern.)
#
# To see if this is driven by visual stuff only and not some mistaken assumption in the weight modification process, we can test the normal agent, with flipped observations and flipped actions only...
#
# TODO: do this!

# %%[markdown]
#
# Let's think about how this corner-seeking could be implemented.
#
# One possibility is that it all happens in the final fully-connected layers, and everything in the conv layers is just detecting maze-solving-relate stuff?  I doubt this is true (<20% credence) but worth starting here?
#
# Let's first visualize the flatten-to-fc weights, reshaped to return the spatial context... TODO

# %%[markdown]
#
# What would happen if we created new agent, which is the combination of multiple agents with a summation at the final logits?  Would such an agent be capable at all, or would it just get stuck if the models disagreed, or be broken in some other way?  Let's try it...
#
# Okay, 

# %%
# Try combining two agents at the logit level

def get_dist_from_policies(policies, obs_t):
    logits_list = []
    for policy in policies:
        hidden = policy.embedder(obs_t)
        logits_list.append(policy.fc_policy(hidden))
    # Sum the logits
    logits_comb = rearrange(logits_list, 'p b a -> p b a').mean(axis=0)
    log_probs = F.log_softmax(logits_comb, dim=1)
    return Categorical(logits=log_probs)

def get_predict_multi(policies):
    def predict(obs, deterministic):
        obs_t = t.FloatTensor(obs)
        dist = get_dist_from_policies(policies, obs_t)
        if deterministic:
            act = dist.mode.numpy()
        else:
            act = dist.sample().numpy()
        return act, None, dist.logits.detach().numpy()
    return predict

def get_forward_multi(policies):
    def forward(obs_t):
        dist = get_dist_from_policies(policies, obs_t)
        return dist, None
    return forward

def rollout_and_vfield_diff(level, policies_original, policies_patched, desc):
    print(f'{desc} | level:{level}')
    # Rollout
    predict_dict = {'original': get_predict_multi(policies_original),
        'patched': get_predict_multi(policies_patched)}
    display(rollout_utils.side_by_side_rollout(predict_dict, level))
    # Vfield
    venv = maze.create_venv(1, start_level=level, num_levels=1)
    policy_hacked = copy.deepcopy(policy_normal)
    policy_hacked.forward = get_forward_multi(policies_original)
    vf_original = vfield.vector_field(venv, policy_hacked)
    policy_hacked.forward = get_forward_multi(policies_patched)
    vf_patched = vfield.vector_field(venv, policy_hacked)
    vfield.plot_vfs(vf_original, vf_patched)
    plt.show()

desc = 'Original vs all-dirs-combined'
policies_all_dirs = [
    flip_weights_and_actions(policy_normal, flip_x=True, flip_y=False),
    flip_weights_and_actions(policy_normal, flip_x=False, flip_y=False),
    flip_weights_and_actions(policy_normal, flip_x=True,  flip_y=True),
    flip_weights_and_actions(policy_normal, flip_x=False, flip_y=True)]

rollout_and_vfield_diff(13, [policy_normal], policies_all_dirs, desc)
rollout_and_vfield_diff(2, [policy_normal], policies_all_dirs, desc)
