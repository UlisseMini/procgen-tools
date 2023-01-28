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

def create_venv(num: int, start_level: int = 0, num_levels: int = 1):
    venv = ProcgenGym3Env(
        num=num, env_name='maze', num_levels=num_levels, start_level=start_level,
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


def venv_patch_pair(seed: int):
    "Return a venv of 2 environments from a seed, one with cheese, one without cheese"
    venv = create_venv(num=2, start_level=seed)
    state_bytes_list = venv.env.callmethod("get_state")[1]
    state = maze.EnvState(state_bytes_list[1])

    # TODO(uli): The multiple sources of truth here suck. Ideally one object linked to venv auto-updates(?)
    grid = state.full_grid()
    grid[grid == maze.CHEESE] = 0
    state.set_grid(grid)
    state_bytes_list[1] = state.state_bytes
    venv.env.callmethod("set_state", state_bytes_list)

    return venv


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

# Get patching function 
def patch_layer(hook, values, coeff:float, activation_label: str, venv, levelpath: str = '', display_bl: bool = True):
    """
    Subtract coeff*(values[0, ...] - values[1, ...]) from the activations at label given by activation_label.  If display_bl is True, plot using logits_to_action_plot and video of rollout in the first environment specified by venv. Saves movie at "videos/lvl-{seed}-{coeff}.mp4".
    """
    assert hasattr(venv, 'num_envs'), "Environment must be vectorized"

    cheese = values[0,...]
    no_cheese = values[1,...]
    assert np.any(cheese != no_cheese), "Cheese and no cheese values are the same"

    cheese_diff = cheese - no_cheese # Subtract this from activation_label's activations during forward passes

    patches = {activation_label: lambda outp: outp - coeff*cheese_diff}

    mode = 'patched' if coeff != 0 else 'unpatched'
    env = copy_venv(venv, 0)
    with hook.use_patches(patches):
        seq, _, _ = cro.run_rollout(predict, env, max_steps=250, deterministic=False)

    hook.probe_with_input(seq.obs.astype(np.float32))
    action_logits = hook.get_value_by_label('fc_policy_out')

    if display_bl:
        logits_to_action_plot(action_logits, title=activation_label)
        
        vidpath = path_prefix + f'videos/lvl-{levelpath}-{coeff}.mp4'
        clip = ImageSequenceClip([aa.to_numpy() for aa in seq.renders], fps=10.)
        clip.write_videofile(vidpath, logger=None)
        display(Video(vidpath, embed=True))

label = 'embedder.block2.res1.resadd_out'
diff_coeffs = (0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 1000)

# %%
# Sweep all levels using patches gained from each level
hook = cmh.ModuleHook(policy)
def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)

for diff_coeff in diff_coeffs:
    for seed in range(num):
        venv = load_venv_from_file('mazes/lvl-num-'+seed+'.pkl')
        obs = venv.reset().astype(np.float32)

        hook.probe_with_input(obs, func=forward_func_policy)
        values = hook.get_value_by_label(label)
        patch_layer(hook, values, diff_coeff, label, venv, levelpath=seed, display_bl=True)
        # Wait for input from jupyter notebook
        print(f"Finished {seed} {diff_coeff}")
        if in_jupyter:
            input("Press Enter to continue...")

# %% 
# Try using one patch for many levels at different strengths
hook = cmh.ModuleHook(policy)
def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)

fixed_value_source = '0-rev'
venv = load_venv_from_file(f'mazes/lvl-num-{fixed_value_source}.pkl')
obs = venv.reset().astype(np.float32)

hook.probe_with_input(obs, func=forward_func_policy)
values = hook.get_value_by_label(label)

for diff_coeff in (1, 2, 3, 5, 10, 20):
    for seed in ('0',):  
        venv = load_venv_from_file(f'mazes/lvl-num-'+seed+'.pkl')
        # hook.probe_with_input(obs, func=forward_func_policy)
        patch_layer(hook, values, diff_coeff, label, venv, levelpath=f'{seed}-fixed-{fixed_value_source}')

# %% 
# Try all labels 
labels = list(hook.values_by_label.keys())
for label in labels: # block2 res2 resadoutt seems promising somehow?
    # if label == 'embedder.block1.maxpool_out': break 
    values = hook.get_value_by_label(label)
    patch_layer(hook, values, label, venv)
    hook.probe_with_input(obs, func=forward_func_policy)

# %%
