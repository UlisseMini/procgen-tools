# %%
# Imports
# %reload_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm import tqdm
from einops import rearrange
from IPython.display import Video, display

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro

from procgen_tools import maze
from procgen_tools.models import load_policy
import procgen_tools.models as models

import gatherdata

# %% 
# Load two levels and get values
import pickle as pkl
from procgen import ProcgenGym3Env
import lovely_tensors as lt
lt.monkey_patch()

# Check whether we're in jupyter
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False

path_prefix = '../' if in_jupyter else ''

def create_venv(num: int):
    venv = ProcgenGym3Env(
        num=num, env_name='maze', num_levels=1, start_level=0,
        distribution_mode='hard', num_threads=1, render_mode="rgb_array",
    )
    venv = maze.wrap_venv(venv)
    return venv


def copy_venv(venv, idx: int):
    "Return a copy of venv number idx. WARNING: After reset, env will be a new maze."
    sb = venv.env.callmethod("get_state")[idx]
    env = create_venv(num=1)
    env.env.callmethod("set_state", [sb])
    return env

def load_venv_from_file(path: str):
    venv = create_venv(num=2)
    path_prefix = '../' if in_jupyter else ''
    with open(path_prefix + path, 'rb') as f:
        state_bytes = pkl.load(f) 
    venv.env.callmethod('set_state', state_bytes)
    def _step(*_, **__):
        raise NotImplementedError('This venv is only used as a template for copy_env')
    venv.step = _step
    return venv

venv = load_venv_from_file('mazes/2.pkl')

# Get initial observation, and show maze rendering
obs = venv.reset().astype(np.float32)  # Not sure why the venv is returning a float64 object?

# Load model

policy = models.load_policy(path_prefix + 'trained_models/maze_I/model_rand_region_15.pth', 15,
    t.device('cpu'))


hook = cmh.ModuleHook(policy)
def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)
hook.probe_with_input(obs, func=forward_func_policy)

# print(hook.values_by_label.keys())

# %% 
# Get patching function 
activation_label = 'embedder.block3.res2.resadd_out'
value = hook.get_value_by_label(activation_label)
cheese_diff = value[0,...] - value[1,...] # Add this to activation_label's activations during forward passes
# mask = t.from_numpy(np.ones(1,dtype=bool))

patches = {activation_label: lambda outp: outp - cheese_diff}

# Custom predict function to match rollout expected interface, uses
# the hooked network so it is patchable
def predict(obs, deterministic):
    obs = t.FloatTensor(obs)
    dist, value = hook.network(obs)
    if deterministic:
        act = dist.mode.numpy() # Take most likely action
    else:
        act = dist.sample().numpy() # Sample from distribution
    return act, None


# %% 
# Reload a single env; can't do two at once

env = copy_venv(venv, 0)
# use env, a copy of the first venv env

# %%
# Make a video of the renders so we can see what it looks like
def logits_to_action_plot(logits):
    """
    Plot the action logits as a heatmap, ignoring bogus repeat actions. Use px.imshow. Assumes logits is a DataArray of shape (n_steps, n_actions).
    """
    logits_np = logits.to_numpy()
    prob = t.softmax(t.from_numpy(logits_np), dim=-1)
    action_indices = models.MAZE_ACTION_INDICES
    prob_dict = models.human_readable_actions(t.distributions.categorical.Categorical(probs=prob))
    prob_dist = t.stack(list(prob_dict.values()))
    px.imshow(prob_dist, y=[k.title() for k in prob_dict.keys()]).show()

DETERMINISTIC = False
MAX_STEPS = 100
# Run a normal, unpatched roll-out
seq, _, _ = cro.run_rollout(predict, env, max_steps=MAX_STEPS, deterministic=DETERMINISTIC)

# Probe to get all the activations for this rollout, then show the action logits
hook.probe_with_input(seq.obs.astype(np.float32))
action_logits_label = 'fc_policy_out'
action_logits_orig = hook.get_value_by_label(action_logits_label)
logits_to_action_plot(action_logits_orig)

# Make a video of the renders so we can see what it looks like
vid_fn, fps = cro.make_video_from_renders(seq.renders, fps=10)
display(Video(vid_fn, embed=True))

# %% 
# Run a patched rollout
env = copy_venv(venv, 0) 

with hook.use_patches(patches):
    seq_patched, _, _ = cro.run_rollout(predict, env, max_steps=MAX_STEPS, deterministic=DETERMINISTIC)
# Probe to get all the activations for this rollout, then show the action logits
hook.probe_with_input(seq_patched.obs.astype(np.float32), patches=patches)
action_logits_patched = hook.get_value_by_label(action_logits_label)
logits_to_action_plot(action_logits_patched)


vid_fn, fps = cro.make_video_from_renders(seq_patched.renders, fps=10)
display(Video(vid_fn, embed=True))
# Do px.imshow, but with the action names as the x-axis

# %%
