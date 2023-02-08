# %%
%reload_ext autoreload
%autoreload 2

# Install procgen tools if needed
try:
  import procgen_tools
except ImportError:
  get_ipython().run_line_magic(magic_name='pip', line='install git+https://github.com/ulissemini/procgen-tools')

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
from IPython.display import Video, display, clear_output
from ipywidgets import *
import itertools
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt

import circrl.module_hook as cmh
import procgen_tools.models as models
from procgen_tools.patch_utils import *
from procgen_tools.vfield import *
from procgen import ProcgenGym3Env

# %%
# Check whether we're in jupyter
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False
path_prefix = '../' if in_jupyter else ''

# %%
# Load model
rand_region = 5
policy = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, t.device('cpu'))

# %%
label = 'embedder.block2.res1.resadd_out'
interesting_coeffs = np.linspace(-2/3,2/3,10) 
hook = cmh.ModuleHook(policy)

# RUN ABOVE here; the rest are one-off experiments which don't have to be run in sequence
# %% Vfields on each maze
""" The vector field is a plot of the action probabilities for each state in the maze. Let's see what the vector field looks like for a given seed. We'll compare the vector field for the original and patched networks. 
"""
@interact
def interactive_patching(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-3, max=3, step=0.1, value=-1)):
    fig, _, _ = plot_patched_vfields(seed, coeff, label, hook)
    plt.show()

# %% Patching from a fixed seed
""" Let's see what happens when we patch the network from a fixed seed. We'll compare the vector field for the original and patched networks.
"""
value_seed = 0
values_tup = cheese_diff_values(value_seed, label, hook), value_seed
for seed in range(10):  
    run_seed(seed, hook, [-1], values_tup=values_tup)


# %% We can patch a range of coefficients and seeds, saving figures from each one for later reference. This is somewhat deprecated due to the interactive plotting above.
seeds = range(10)
coeffs = [-2, -1, -0.5, 0.5, 1, 2]
for seed, coeff in tqdm(list(itertools.product(seeds, coeffs))):
    fig, _, _ = plot_patched_vfields(seed, coeff, label=label, hook=hook)
    fig.savefig(f"../figures/patched_vfield_seed{seed}_coeff{coeff}.png", dpi=300)
    plt.clf()
    plt.close()

# %% Live vfield probability visualization
""" Edit a maze and see how that changes the vector field representing the action probabilities. """
vbox = custom_vfield(policy, seed=2)
display(vbox)

# %% We can construct a patch which averages over a range of seeds, and see if that generalizes better (it doesn't)
values = np.zeros_like(cheese_diff_values(0, label, hook))
seeds = slice(int(10e5),int(10e5+100))

# Iterate over range specified by slice
for seed in range(seeds.start, seeds.stop):
    # Make values be rolling average of values from seeds
    values = (seed-seeds.start)/(seed-seeds.start+1)*values + cheese_diff_values(seed, label, hook)/(seed-seeds.start+1)

# Assumes a fixed venv, hook, values, and label
@interact
def interactive_patching(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-10, max=10, step=0.1, value=-1)):
    fig, _, _ = plot_patched_vfields(seed, coeff, label, hook, values=values)
    plt.show()


# %% Patching with a random vector 
""" Are we just seeing noise? Let's try patching with a random vector and see if that works. First, let's find appropriate-magnitude random vectors."""
rand_magnitude = .25
for mode in ['random', 'cheese']:
    vectors = []
    for value_seed in range(100):
        if mode == 'random':
            vectors.append(np.random.randn(*cheese_diff_values(0, label, hook).shape, ) * rand_magnitude)
        else:
            vectors.append(cheese_diff_values(value_seed, label, hook))
        
    norms = [np.linalg.norm(v) for v in vectors]
    print(f'For {mode}-vectors, the norm is {np.mean(norms):.2f} with std {np.std(norms):.2f}. Max absolute-value difference of {np.max(np.abs(vectors)):.2f}.')

# %% Run the patches
values = np.random.randn(*cheese_diff_values(0, label, hook).shape) * rand_magnitude
# Cast this to float32
values = values.astype(np.float32)
print(np.max(values).max())
for seed in range(5):
    run_seed(seed, hook, [-1], values_tup=(values, 'garbage'))

# It doesn't work, and destroys performance. In contrast, the cheese vector has a targeted and constrained effect on the network (when not transferring to other mazes), and does little when attempting transfer. This seems intriguing.

# %% Patching different layers
""" We chose the layer block2.res1.resadd_out because it seemed to have a strong effect on the vector field. Let's see what happens when we patch other layers. """

labels = list(hook.values_by_label.keys()) # TODO this dict was changing in size during the loop, but why?
@interact
def run_all_labels(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-3, max=3, step=0.1, value=-1), label=labels):
    fig, _, _ = plot_patched_vfields(seed, coeff, label, hook)
    plt.show()

# %% Transfer to same cheese locations
""" Let's see what happens when we transfer the patches to the same cheese locations. """
value_seed = 0
values = cheese_diff_values(value_seed, label, hook)
cheese_location = maze.get_cheese_pos_from_seed(value_seed)
print(f'The cheese vector is taken from a seed with cheese at {cheese_location}.')

maze.mazes_with_cheese_at_location = get_mazes_with_cheese_at_location(cheese_location, 5, skip_seed = value_seed) 

# %% Now we can see how well the patch transfers to mazes with cheese at the same location.
assert len(mazes_with_cheese_at_location) > 0, "No mazes with cheese at the specified location."
@interact
def run_all_mazes_with_cheese_at_location(seed=Dropdown(options=mazes_with_cheese_at_location)):
    fig, _, _ = plot_patched_vfields(seed, -1, label, hook, values=values, render_padding=True)
    plt.show()