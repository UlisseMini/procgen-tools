# %% [markdown]
# # Visualizing details of the goal misgeneralization nets
# Let's understand lots of details about [the goal misgeneralization paper](https://arxiv.org/abs/2105.14111). In particular, we'll be looking at the cheese-maze task from the goal misgeneralization task, for which cheese was spawned in the 5x5 top-right corner of the maze. 
# 
# Key conclusions:
# 1. Convolutional layers limit speed of information propagation (_locality_). More precisely, ignoring the effect of `maxpool2D` layers, any portions of the state separated by `n` pixels take at least `ceil(n/2)` convolutional layers to interact.
# 2. In the maximal maze size of 64 x 64, there is at most **two** steps of computation involving information from e.g. opposite corners of the maze. 

# %%
# %% Don't have to restart kernel and reimport each time you modify a dependency
%reload_ext autoreload
%autoreload 2

# %%
# Imports
from typing import List, Tuple, Dict, Union, Optional, Callable
import re 

import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from einops import rearrange
from IPython.display import *
from ipywidgets import *
import itertools
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt

# Install procgen tools if needed
try:
  import procgen_tools
except ImportError:
  get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

# %%
# Download data and create directory structure

import os, sys
from pathlib import Path
from procgen_tools.utils import setup

setup() # create directory structure and download data

# path this notebook expects to be in
if 'experiments' not in os.getcwd():
    Path('experiments').mkdir(exist_ok=True)
    os.chdir('experiments')


# %%
import circrl.module_hook as cmh
import procgen_tools.models as models
from procgen_tools.patch_utils import *
from procgen_tools.visualization import *

from procgen import ProcgenGym3Env
from ipywidgets import Text # Import this later because otherwise Text gets cast as str?

RAND_REGION = 5
NUM_ACTIONS = 15
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False
PATH_PREFIX = '../' if in_jupyter else ''

# %%
# Load model
model_path = PATH_PREFIX + f'trained_models/maze_I/model_rand_region_{RAND_REGION}.pth'
policy = models.load_policy(model_path, NUM_ACTIONS, t.device('cpu'))
hook = cmh.ModuleHook(policy)

# %% [markdown]
# Let's visualize the network structure. Here's a Mermaid diagram. 
# 
# ![](https://i.imgur.com/acsV4aD.png) 

# %% [markdown]
# And here's a more dynamic view; small nodes are activations, and large nodes are `nn.Module`s.

# %%
hook.run_with_input(np.zeros((1,3, 64, 64), dtype=np.float32))
hook.get_graph(include_parent_modules=False)

# %%
def dummy_obs_pair(color: str, location: Tuple[int, int]=(32,32)):
    """ Returns a mostly-black image pair, the first of which contains a red/green/blue pixel in the center. Returns obs of shape (2, 3, 64, 64). """
    
    assert color in ['R', 'G', 'B'], f'Color must be one of R, G, B, not {color}'
    assert len(location) == 2, 'Location must be a tuple of length 2'
    assert all(0 <= col < 64 for col in location), 'Location must be in [0, 64)'

    channel = {'R': 0, 'G': 1, 'B': 2}[color]
    obs = np.zeros((2, 3, 64, 64), dtype=np.float32)
    obs[0, channel, location[0], location[1]] = 1 # Have one pixel in the middle, in the given channel
    return obs
    
# Let's load a dummy observation with only one nonzero value
plt.imshow(dummy_obs_pair("G")[0].transpose(1,2,0))

# %%
# Get the available labels
hook.run_with_input(dummy_obs_pair('R'))
labels = list(hook.values_by_label.keys())[:-1] # Skip the "_out" layer and remove "embedder." prefixes
assert labels == list(map(expand_label, format_labels(labels)))

# %%
# Let's visualize the activations at each layer using plotly, using an interactive interface that lets us slide the R/G/B pixel around
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def get_activations(label: str, obs: np.ndarray):
    hook.run_with_input(obs) # Run the model with the given obs
    return hook.get_value_by_label(label) # Shape is (b, c, h, w) at conv layers, (b, activations) at linear layers

def activation_diff(label: str, obs: np.ndarray):
    assert obs.shape[0] == 2 # Must be a pair of observations
    
    activations = get_activations(label, obs)
    return activations[0] - activations[1] # Subtract in order to cancel out bias terms which don't behave differently in the presence of the differing inputs 

def plot_activations(activations: np.ndarray, fig: go.FigureWidget):
    """ Plot the activations given a single (non-batched) activation tensor. """
    # If there's a single-batch element, remove it with squeeze
    activations = np.squeeze(activations)
    fig.update(data=[go.Heatmap(z=activations)])

# %%
# We can begin to see how information propagates across the net for a dummy single-pixel input
def activ_gen_px(col: int, row: int, label: str, color: str, pair: bool = False):
    """ Get the activations for running a forward pass on a dummy observation, in the given color. Returns shape of (batch, channels, rows, cols), where size batch=1 if pair=False, and batch=2 if pair=True. """
    activations = get_activations(label, dummy_obs_pair(color, (row, col)))
    return np.expand_dims(activations[0], axis=0) if not pair else activations

# Instantiate the plotter
activ_plotter = ActivationsPlotter(labels, plotter=plot_activations, activ_gen=activ_gen_px, hook=hook, coords_enabled=True, color="R") # TODO for some reason block2.maxpool_out has a different fig width?
activ_plotter.display()

# %%
def plot_nonzero_activations(activations: np.ndarray, fig: go.FigureWidget): 
    """ Plot the nonzero activations in a heatmap. """
    # Find nonzero activations and cast to floats
    nz = (activations != 0).astype(np.float32)
    plot_activations(nz, fig=fig)

def plot_nonzero_diffs(activations: np.ndarray, fig: go.FigureWidget):
    """ Plot the nonzero activation diffs in a heatmap. """
    diffs = activations[0] - activations[1]
    plot_nonzero_activations(diffs, fig)

# Instantiate the plotter
nonzero_plotter = ActivationsPlotter(labels, plotter=plot_nonzero_diffs, activ_gen=activ_gen_px, hook=hook, coords_enabled=True, color="R", pair=True)
nonzero_plotter.display()

# %% [markdown]
# # 1: Locality
# Consider `n` convolutional layers (3x3 kernel, stride=1, padding=1) which each preserve the height col width of the previous feature maps. The above demonstrates that after these layers, information can only propagate `n` L1 pixels. The network itself is composed of # TODO 

# %%
# Load up a cheese/no-cheese maze pair 
from procgen_tools.patch_utils import *

seed = 0
venv = get_cheese_venv_pair(seed=seed)
obs = venv.reset()
obs = np.array(obs, dtype=np.float32)

# Show the diff of the RGB renders
print("The difference between the two images is:")
plt.imshow(rearrange(obs[0]-obs[1], 'c h w -> h w c'))

# %%
# Visualize the activations for this pair TODO doesn't work
cheese_diff_plotter = ActivationsPlotter(labels, plot_activations, lambda label: activation_diff(label, obs), hook)
cheese_diff_plotter.display()

# %%
cheese_pixels = obs[0][(obs[0]-obs[1]) != 0] # Get the nonzero pixels
cheese_pixels = [(cheese_pixels[i], cheese_pixels[i+1], cheese_pixels[i+2]) for i in range(0, len(cheese_pixels), 3)] # Split into (R,G,B) tuples

def show_cheese_at(col: int, row: int):
    """ Make an all-black image with cheese at the given location. """
    obs = np.zeros((3,64,64), dtype=np.float32)
    obs[:, row, col] = cheese_pixels[0]
    obs[:, row, col+1] = cheese_pixels[1]
    return obs

plt.imshow(rearrange(show_cheese_at(0,0), 'c h w -> h w c')) # Show the cheese at the top left corner

# %%
def activ_gen_cheese(col: int, row: int, label: str):
    """ Return the activations for an observation with cheese at the given location. """
    cheese_obs = show_cheese_at(col,row)
    cheese_obs = rearrange(cheese_obs, 'c h w -> 1 c h w')
    activations = get_activations(label, cheese_obs)
    return activations[0]

moving_cheese_plotter = ActivationsPlotter(labels, plot_activations, activ_gen_cheese, hook, coords_enabled=True)
moving_cheese_plotter.display()

# %% [markdown]
# # Visualizing actual observation activations

# %%
SEED = 2
venv = create_venv(num=1, start_level=SEED, num_levels=1)

def activ_gen_obs(label: str):
    """ Returns a tensor of shape (1, channels, rows, cols)."""
    obs = venv.reset() # TODO pretty sure this needs to update -- maybe via a class?
    obs = np.array(obs, dtype=np.float32)
    return get_activations(label, obs)

# Show a maze editor side-by-side with the interactive plotter
custom_maze_plotter = ActivationsPlotter(labels, plotter=plot_activations, activ_gen=activ_gen_obs, hook=hook)

widget_box = custom_vfield(policy, seed=SEED, callback=lambda _: custom_maze_plotter.update_plotter())

display(widget_box)
    
custom_maze_plotter.display() 
# TODO make it so that it's easy to attach notes to files, load  



# %%    
single_venv = maze.create_venv(num=1, start_level=1, num_levels=1)
editors = maze.venv_editors(single_venv, check_on_dist=False, env_nums=range(1))
display(editors)
# %%
