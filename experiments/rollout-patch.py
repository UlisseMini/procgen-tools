# %%
# Imports
from typing import List, Tuple, Dict, Union, Optional, Callable

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

rand_region = 5
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
    state_bytes_list = venv.env.callmethod("get_state")
    state = maze.EnvState(state_bytes_list[1])

    # TODO(uli): The multiple sources of truth here suck. Ideally one object linked to venv auto-updates(?)
    grid = state.full_grid()
    grid[grid == maze.CHEESE] = maze.EMPTY
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

# %%
# Load model
policy = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15,
    t.device('cpu'))

def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)


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
def patch_layer(hook, values, coeff:float, activation_label: str, venv, levelpath: str = '', display_bl: bool = True, vanished=False, steps: int = 1000):
    """
    Subtract coeff*(values[0, ...] - values[1, ...]) from the activations at label given by activation_label.  If display_bl is True, plot using logits_to_action_plot and video of rollout in the first environment specified by venv. Saves movie at "videos/lvl-{seed}-{coeff}.mp4".
    """
    assert hasattr(venv, 'num_envs'), "Environment must be vectorized"

    cheese = values[0,...]
    no_cheese = values[1,...]
    assert np.any(cheese != no_cheese), "Cheese and no cheese values are the same"

    cheese_diff = cheese - no_cheese # Subtract this from activation_label's activations during forward passes
    patches = {activation_label: lambda outp: outp - coeff*cheese_diff}

    env = copy_venv(venv, 1 if vanished else 0)
    with hook.use_patches(patches):
        seq, _, _ = cro.run_rollout(predict, env, max_steps=steps, deterministic=False)

    hook.probe_with_input(seq.obs.astype(np.float32))
    action_logits = hook.get_value_by_label('fc_policy_out')

    if display_bl:
        logits_to_action_plot(action_logits, title=activation_label)
        
        vidpath = path_prefix + f'videos/{rand_region}/lvl:{levelpath}_coeff:{coeff}.mp4'
        clip = ImageSequenceClip([aa.to_numpy() for aa in seq.renders], fps=10.)
        clip.write_videofile(vidpath, logger=None)
        display(Video(vidpath, embed=True))


# %%
label = 'embedder.block2.res1.resadd_out'
diff_coeffs = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 1000]
hook = cmh.ModuleHook(policy)

# %% 
# Infrastructure for running different kinds of seeds
def get_values(seed:int, label:str, hook: cmh.ModuleHook):
    """ Get the cheese/no-cheese activations for the given seed. """
    venv = venv_patch_pair(seed) 
    obs = venv.reset().astype(np.float32)
    hook.probe_with_input(obs, func=forward_func_policy)
    return hook.get_value_by_label(label)

def run_seed(seed:int, hook: cmh.ModuleHook, diff_coeffs: List[float], display_bl: bool = True, values:Optional[Union[np.ndarray, str]]=None, label='embedder.block2.res1.resadd_out', steps:int=150):
    """ Run a single seed, with the given hook, diff_coeffs, and display_bl. If values is provided, use those values for the patching. Otherwise, generate them via a cheese/no-cheese activation diff.""" 
    venv = venv_patch_pair(seed) 

    # Get values if not provided
    if values is None:
        values = get_values(seed, label, hook)
        value_src = seed 
    else:
        values, value_src = values

    # Show behavior on the level without cheese
    patch_layer(hook, values, 0, label, venv, levelpath=f'{seed}-vanished', display_bl=display_bl, vanished=True, steps=steps)

    for coeff in diff_coeffs:
        # hook.probe_with_input(obs, func=forward_func_policy) # TODO does this have to be reset?
        patch_layer(hook, values, coeff, label, venv, levelpath=f'{seed}_vals:{value_src}', display_bl=display_bl, vanished=False, steps=steps)

        # Wait for input from jupyter notebook
        # print(f"Finished {seed} {diff_coeff}")
        # if in_jupyter:
        #     input("Press Enter to continue...")

# %%
# Sweep all levels using patches gained from each level
for seed in range(50):
    run_seed(seed, hook, diff_coeffs)

# %% 
# Try using one patch for many levels at different strengths
venv = load_venv_from_file(f'mazes/lvl-num-{fixed_value_source}.pkl')
value_seed = 0
values = get_values(value_seed, label, hook) 

for seed in range(20):  
    run_seed(seed, hook, [1,5,10], values=(values, str(value_seed)))

# %% 
# Average diff over a bunch of seeds
values = np.zeros_like(get_values(0, label, hook))
seeds = slice(10e5,10e5+50)
for seed in range(seeds):
    values += get_values(seed, label, hook)
values /= len(seeds)

for seed in range(20):
    run_seed(seed, hook, [1,5,10], values=(values, f'avg from {seeds.start} to {seeds.stop}'))

# %% 
# Try all labels for a fixed seed and diff_coeff
labels = list(hook.values_by_label.keys()) # TODO this dict was changing in size during the loop, but why?
for label in labels: 
    run_seed(0, hook, [1], label=label)

# %%
