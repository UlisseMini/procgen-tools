# %%
# Imports
import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm import tqdm
from einops import rearrange
from IPython.display import Video, display
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

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

# Load model
policy = models.load_policy(path_prefix + 'trained_models/maze_I/model_rand_region_15.pth', 15,
    t.device('cpu'))




# %% 
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

def logits_to_action_plot(logits, title=''):
    """
    Plot the action logits as a heatmap, ignoring bogus repeat actions. Use px.imshow. Assumes logits is a DataArray of shape (n_steps, n_actions).
    """
    logits_np = logits.to_numpy()
    prob = t.softmax(t.from_numpy(logits_np), dim=-1)
    action_indices = models.MAZE_ACTION_INDICES
    prob_dict = models.human_readable_actions(t.distributions.categorical.Categorical(probs=prob))
    prob_dist = t.stack(list(prob_dict.values()))
    px.imshow(prob_dist, y=[k.title() for k in prob_dict.keys()],title=title).show()
    # Get px imshow of the logits, with the action labels, showing the title

diff_coeff = 100

# Get patching function 
def patch_layer(hook, values, activation_label: str, venv, level: str = ''):
    """
    Subtract (values[0, ...] - values[1, ...]) from the activations at label given by activation_label. Plot using logits_to_action_plot and video of rollout in the first environment specified by venv. 
    """
    assert hasattr(venv, 'num_envs'), "Environment must be vectorized"

    cheese = values[0,...]
    no_cheese = values[1,...]
    assert np.any(cheese != no_cheese), "Cheese and no cheese values are the same"

    cheese_diff = cheese - no_cheese # Subtract this from activation_label's activations during forward passes

    patches = {activation_label: lambda outp: outp - diff_coeff*cheese_diff}

    DETERMINISTIC = False
    MAX_STEPS = 400
    action_logits_label = 'fc_policy_out'

    for mode in ('original', 'patched'):
        env = copy_venv(venv, 0)
        if mode == 'patched':
            with hook.use_patches(patches):
                seq, _, _ = cro.run_rollout(predict, env, max_steps=MAX_STEPS, deterministic=DETERMINISTIC)
        else:
            seq, _, _ = cro.run_rollout(predict, env, max_steps=MAX_STEPS, deterministic=DETERMINISTIC)

        hook.probe_with_input(seq.obs.astype(np.float32))
        action_logits = hook.get_value_by_label(action_logits_label)

        if mode == 'patched':
            logits_to_action_plot(action_logits, title=activation_label)
            vid_fn, fps = cro.make_video_from_renders(seq.renders, fps=10)
            display(Video(vid_fn, embed=True))
            
            vidpath = path_prefix + f'videos/lvl-{level}-{mode}{"-" + str(diff_coeff) if mode == "patched" else ""}.mp4'
            clip = ImageSequenceClip([aa.to_numpy() for aa in seq.renders], fps=fps)
            clip.write_videofile(vidpath, logger=None)

# %%
hook = cmh.ModuleHook(policy)
def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)
label = 'embedder.block2.res1.resadd_out'

for diff_coeff in (1, 2, 3, 5, 10, 20, 50, 100, 1000, 10000):
    for mazename in ('0', '0-rev', '2'): # Compare across both files
        venv = load_venv_from_file('mazes/lvl-num-'+mazename+'.pkl')
        obs = venv.reset().astype(np.float32)

        hook.probe_with_input(obs, func=forward_func_policy)
        values = hook.get_value_by_label(label)
        patch_layer(hook, values, label, venv, level=mazename)

# %% 
# Try all labels 
labels = list(hook.values_by_label.keys())
for label in labels: # block2 res2 resadoutt seems promising somehow?
    # if label == 'embedder.block1.maxpool_out': break 
    values = hook.get_value_by_label(label)
    patch_layer(hook, values, label, venv)
    hook.probe_with_input(obs, func=forward_func_policy)

# %%
