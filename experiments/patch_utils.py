# %%
# Imports
from typing import List, Tuple, Dict, Union, Optional, Callable

import numpy as np
import torch as t
import plotly.express as px
import matplotlib.pyplot as plt
from IPython.display import Video, display, clear_output
from ipywidgets import Text, interact
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

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


def get_cheese_venv_pair(seed: int):
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

def get_custom_venv_pair(seed: int):
    """ Allow the user to edit two levels from a seed. Return a venv containing both environments. """
    venv = create_venv(num=2, start_level=seed)
    display(maze.venv_editor(venv))
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

def get_patches(values: np.ndarray, coeff: float, label: str):
    """ Get a patch function that patches the activations at label with coeff*(values[0, ...] - values[1, ...]). TODO generalize """
    cheese = values[0,...]
    no_cheese = values[1,...]
    assert np.any(cheese != no_cheese), "Cheese and no cheese values are the same"

    cheese_diff = cheese - no_cheese # Add this to activation_label's activations during forward passes
    return {label: lambda outp: outp + coeff*cheese_diff}

def patch_layer(hook, values, coeff:float, activation_label: str, venv, seed: str = '', show_video: bool = False, show_vfield: bool = True, vanished=False, steps: int = 1000):
    """
    Add coeff*(values[0, ...] - values[1, ...]) to the activations at label given by activation_label.  If display_bl is True, plot using logits_to_action_plot and video of rollout in the first environment specified by venv. Saves movie at "videos/lvl-{seed}-{coeff}.mp4".
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
    
    patches = get_patches(values, coeff, label=activation_label)

    if show_video:
        env = copy_venv(venv, 1 if vanished else 0)
        with hook.use_patches(patches):
            seq, _, _ = cro.run_rollout(predict, env, max_steps=steps, deterministic=False)
        hook.probe_with_input(seq.obs.astype(np.float32))
        action_logits = hook.get_value_by_label('fc_policy_out')
        logits_to_action_plot(action_logits, title=activation_label)
        
        vidpath = path_prefix + f'videos/{rand_region}/lvl:{seed}_{"no_cheese" if vanished else "coeff:" + str(coeff)}.mp4'
        clip = ImageSequenceClip([aa.to_numpy() for aa in seq.renders], fps=10.)
        clip.write_videofile(vidpath, logger=None)
        display(Video(vidpath, embed=True))

    if show_vfield:
        # Make a side-by-side subplot of the two vector fields
        plt.figure(figsize=(10, 5))
        plt.title(f"{activation_label} activations at coefficient {coeff}")
        # Make the subplots 
        plt.subplot(1, 2, 1)
        plt.gca().set_title("Vector field of original network")
        vfield.plot_vector_field(venv, hook.network)

        plt.subplot(1, 2, 2)
        with hook.use_patches(patches):
            plt.gca().set_title("Vector field of patched network")
            vfield.plot_vector_field(venv, hook.network)
        plt.show()

# %% 
# Infrastructure for running different kinds of seeds
def values_from_venv(venv: ProcgenGym3Env, hook: cmh.ModuleHook, label: str):
    """ Get the values of the activations at label for the given venv. """
    obs = venv.reset().astype(np.float32) # TODO why reset?
    hook.probe_with_input(obs, func=forward_func_policy)
    return hook.get_value_by_label(label)

def cheese_diff_values(seed:int, label:str, hook: cmh.ModuleHook):
    """ Get the cheese/no-cheese activations at the label for the given seed. """
    venv = get_cheese_venv_pair(seed) 
    return values_from_venv(venv, hook, label)

def run_seed(seed:int, hook: cmh.ModuleHook, diff_coeffs: List[float], display_bl: bool = True, values_tup:Optional[Union[np.ndarray, str]]=None, label='embedder.block2.res1.resadd_out', steps:int=150):
    """ Run a single seed, with the given hook, diff_coeffs, and display_bl. If values_tup is provided, use those values for the patching. Otherwise, generate them via a cheese/no-cheese activation diff.""" 
    venv = get_cheese_venv_pair(seed) 

    # Get values if not provided
    values, value_src = cheese_diff_values(seed, label, hook), seed if values_tup is None else values_tup

    # Show behavior on the level without cheese
    # patch_layer(hook, values, 0, label, venv, seed=seed, display_bl=display_bl, vanished=True, steps=steps)

    for coeff in diff_coeffs:
        display(Text(f'Patching with coeff {coeff} seed {seed}'))
        patch_layer(hook, values, coeff, label, venv, seed=f'{seed}_vals:{value_src}', show_video=display_bl, vanished=False, steps=steps)


def plot_patched_vfield(seed: int, coeff: float, label: str, hook: cmh.ModuleHook):
    values = cheese_diff_values(seed, label, hook)
    patches = get_patches(values, coeff, label) 

    venv = copy_venv(get_cheese_venv_pair(seed), 0) # Get env with cheese present

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    # remove axis ticks from images
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    ax[0].set_xlabel("Original vfield")
    original_vfield = vfield.plot_vector_field(venv, hook.network, ax=ax[0])
    with hook.use_patches(patches):
        ax[1].set_xlabel("Patched vfield")
        patched_vfield = vfield.plot_vector_field(venv, hook.network, ax=ax[1])

    obj = {
        'seed': seed,
        'coeff': coeff,
        'patch_label': label,
        'original_vfield': original_vfield,
        'patched_vfield': patched_vfield,
    }
    fig.suptitle(f"Level {seed} coeff {coeff}")

    return fig, ax, obj

