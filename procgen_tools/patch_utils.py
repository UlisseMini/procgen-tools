# %%
# Imports
from typing import List, Tuple, Dict, Union, Optional, Callable

import numpy as np
import torch as t
import plotly.express as px
import matplotlib.pyplot as plt
from IPython.display import Video, display, clear_output
from ipywidgets import Text, interact, HBox
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from einops import *

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro

from procgen_tools import maze, vfield
import procgen_tools.models as models
from procgen_tools.maze import copy_venv, create_venv

# %% 
# Load two levels and get values
import pickle as pkl
from procgen import ProcgenGym3Env

# Check whether we're in jupyter
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False

path_prefix = '../' if in_jupyter else ''
rand_region = 5


def get_cheese_venv_pair(seed: int, has_cheese_tup : Tuple[bool, bool] = (True, False)):
    "Return a venv of 2 environments from a seed, with cheese in the first environment if has_cheese_tup[0] and in the second environment if has_cheese_tup[1]."
    venv = create_venv(num=2, start_level=seed, num_levels=1)
    state_bytes_list = venv.env.callmethod("get_state")

    for idx in range(2):
        if has_cheese_tup[idx]: continue # Skip if we want cheese in this environment
        state = maze.EnvState(state_bytes_list[idx])

        # TODO(uli): The multiple sources of truth here suck. Ideally one object linked to venv auto-updates(?)
        grid = state.full_grid()
        grid[grid == maze.CHEESE] = maze.EMPTY
        state.set_grid(grid)
        state_bytes_list[idx] = state.state_bytes
        venv.env.callmethod("set_state", state_bytes_list)

    return venv

def get_custom_venv_pair(seed: int, num_envs=2):
    """ Allow the user to edit num_envs levels from a seed. Return a venv containing both environments. """
    venv = create_venv(num=num_envs, start_level=seed, num_levels=1)
    display(HBox(maze.venv_editor(venv, check_on_dist=False)))
    return venv

def load_venv_pair(path: str):
    """ Load a venv pair from a file. """
    venv = create_venv(num=2, start_level=1, num_levels=1)
    with open(path_prefix + path, 'rb') as f:
        state_bytes = pkl.load(f) 
    venv.env.callmethod('set_state', state_bytes)
    def _step(*_, **__):
        raise NotImplementedError('This venv is only used as a template for copy_env')
    venv.step = _step
    return venv

# %%
# Load model

def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)


# %% 
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

def get_values_diff_patch(values: np.ndarray, coeff: float, label: str):
    """ Get a patch function that patches the activations at label with coeff*(values[0, ...] - values[1, ...]). TODO generalize """
    cheese = values[0,...]
    no_cheese = values[1,...]
    assert np.any(cheese != no_cheese), "Cheese and no cheese values are the same"

    cheese_diff = cheese - no_cheese # Add this to activation_label's activations during forward passes
    return {label: lambda outp: outp + coeff*cheese_diff}

def get_zero_patch(label: str):
    """ Get a patch function that patches the activations at label with 0. """
    return {label: lambda outp: t.zeros_like(outp)}

def get_mean_patch(values: np.ndarray, label: str):
    """ Get a patch that replaces the activations at label with the mean of values, taken across the batch (first) dimension. """
    mean_vals = reduce(t.from_numpy(values), 'batch ... -> ...', 'mean')
    return {label: lambda _: mean_vals}

def patch_layer(hook, values, coeff:float, activation_label: str, venv, seed_str: str = '', show_video: bool = False, show_vfield: bool = True, vanished=False, steps: int = 150):
    """
    Add coeff*(values[0, ...] - values[1, ...]) to the activations at label given by activation_label.  If display_bl is True, plot using logits_to_action_plot and video of rollout in the first environment specified by venv. Saves movie at "videos/{rand_region}/lvl-{seed_str}-{coeff}.mp4", where rand_region is a global int.
    """
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

    assert hasattr(venv, 'num_envs'), "Environment must be vectorized"
    
    patches = get_values_diff_patch(values, coeff, label=activation_label)

    if show_video:
        env = copy_venv(venv, 1 if vanished else 0)
        with hook.use_patches(patches):
            seq, _, _ = cro.run_rollout(predict, env, max_steps=steps, deterministic=False)
        hook.run_with_input(seq.obs.astype(np.float32))
        action_logits = hook.get_value_by_label('fc_policy_out')
        logits_to_action_plot(action_logits, title=activation_label)
        
        vidpath = path_prefix + f'videos/{rand_region}/lvl:{seed_str}_{"no_cheese" if vanished else "coeff:" + str(coeff)}.mp4'
        clip = ImageSequenceClip([aa.to_numpy() for aa in seq.renders], fps=10.)
        clip.write_videofile(vidpath, logger=None)
        display(Video(vidpath, embed=True))

    if show_vfield:
        # Make a side-by-side subplot of the two vector fields
        fig = plt.figure(figsize=(10, 5))

        # Make the subplots 
        plt.subplot(1, 2, 1)
        plt.gca().set_title("Original")
        vf1 = vfield.vector_field(venv, hook.network)
        vfield.plot_vf(vf1)

        plt.subplot(1, 2, 2)
        with hook.use_patches(patches):
            plt.gca().set_title("Patched")
            vf2 = vfield.vector_field(venv, hook.network)
            vfield.plot_vf(vf2)
        # Make a figure title above the two subplots
        fig.suptitle(f"Vector fields for layer {activation_label} with coeff={coeff:.2f} and level={seed_str}") 
        plt.show()

# %% 
# Infrastructure for running different kinds of seeds
def values_from_venv(venv: ProcgenGym3Env, hook: cmh.ModuleHook, label: str):
    """ Get the values of the activations at label for the given venv. """
    obs = venv.reset().astype(np.float32) # TODO why reset?
    hook.run_with_input(obs, func=forward_func_policy)
    return hook.get_value_by_label(label)

def cheese_diff_values(seed:int, label:str, hook: cmh.ModuleHook):
    """ Get the cheese/no-cheese activations at the label for the given seed. """
    venv = get_cheese_venv_pair(seed) 
    return values_from_venv(venv, hook, label)

def run_seed(seed:int, hook: cmh.ModuleHook, diff_coeffs: List[float], show_video: bool = False, show_vfield: bool = True, values_tup:Optional[Union[np.ndarray, str]]=None, label='embedder.block2.res1.resadd_out', steps:int=150, render_padding : bool = False):
    """ Run a single seed, with the given hook and diff_coeffs. If values_tup is provided, use those values for the patching. Otherwise, generate them via a cheese/no-cheese activation diff.""" 
    venv = get_cheese_venv_pair(seed) 

    # Get values if not provided
    values, value_src = (cheese_diff_values(seed, label, hook), seed) if values_tup is None else values_tup

    # Show behavior on the level without cheese
    # patch_layer(hook, values, 0, label, venv, seed=seed, show_video=show_video, show_vfield=show_vfield, vanished=True, steps=steps)

    for coeff in diff_coeffs:
        # display(Text(f'Patching with coeff {coeff} seed {seed}'))
        patch_layer(hook, values, coeff, label, venv, seed_str=f'{seed}_vals:{value_src}', show_video=show_video, show_vfield=show_vfield,steps=steps)

def compare_patched_vfields(venv : ProcgenGym3Env, patches : dict, hook: cmh.ModuleHook, render_padding: bool = False, ax_size : int = 4, reuse_first : bool = True):
    """ Takes as input a venv with one or two maze environments. If one and reuse_first is true, we compare vfields for original/patched on that fixed venv. If two, we show the vfield for the original on the first venv environment, and the patched on the second, and the difference between the two. """

    assert 1 <= venv.num_envs <= 2, "Needs one or environments to compare the vector fields"
    venv1, venv2 = copy_venv(venv, 0), copy_venv(venv, 0 if venv.num_envs == 1 or reuse_first else 1)

    original_vfield = vfield.vector_field(venv, hook.network)
    with hook.use_patches(patches):
        patched_vfield = vfield.vector_field(venv2, hook.network)
    fig, axs = vfield.plot_vfs_with_diff(original_vfield, patched_vfield, render_padding=render_padding, ax_size=ax_size)

    obj = {
        'patches': patches,
        'original_vfield': original_vfield,
        'patched_vfield': patched_vfield,
    }

    return fig, axs, obj


def plot_patched_vfields(seed: int, coeff: float, label: str, hook: cmh.ModuleHook, values: Optional[np.ndarray] = None, venv: Optional[ProcgenGym3Env] = None, show_title: bool = False, title:str = '', render_padding: bool = False, ax_size : int = 5):
    """ Plot the original and patched vector fields for the given seed, coeff, and label. If values is provided, use those values for the patching. Otherwise, generate them via a cheese/no-cheese activation diff. """
    values = cheese_diff_values(seed, label, hook) if values is None else values
    patches = get_values_diff_patch(values, coeff, label) 
    venv = copy_venv(get_cheese_venv_pair(seed) if venv is None else venv, 0) # Get env with cheese present / first env in the pair

    fig, axs, obj = compare_patched_vfields(venv, patches, hook, render_padding=render_padding, ax_size=ax_size)
    obj.update({
        'seed': seed,
        'coeff': coeff,
        'patch_label': label,
        })

    if show_title:
        fig.suptitle(title if title != '' else f"Level {seed} coeff {coeff} layer {label}")

    return fig, axs, obj