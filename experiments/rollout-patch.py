# %% Don't have to restart kernel and reimport each time you modify a dependency
%reload_ext autoreload
%autoreload 2

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
from ipywidgets import interact
from ipywidgets import Text, interact, IntSlider, FloatSlider, Dropdown
import itertools
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import procgen_tools.models as models
from patch_utils import *

# %% 
# Load two levels and get values
import pickle as pkl
from procgen import ProcgenGym3Env

rand_region = 5
# Check whether we're in jupyter
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False

path_prefix = '../' if in_jupyter else ''

# %%
# Load model

policy = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15,
    t.device('cpu'))

# %% Experiment parameters
label = 'embedder.block2.res1.resadd_out'
interesting_coeffs = np.linspace(-2/3,2/3,10) 
hook = cmh.ModuleHook(policy)

# RUN ABOVE here
# %% Interactive mode for taking cheese-diffs on one seed
@interact
def interactive_patching(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-3, max=3, step=0.1, value=-1)):
    fig, _, _ = plot_patched_vfield(seed, coeff, label, hook)
    plt.show()


# %% Try using one patch for many levels at different strengths
value_seed = 0
values_tup = cheese_diff_values(value_seed, label, hook), value_seed

for seed in range(10):  
    run_seed(seed, hook, [-1], values_tup=values_tup)

# %% Save figures for a bunch of (seed, coeff) pairs
seeds = range(10)
coeffs = [-2, -1, -0.5, 0.5, 1, 2]
for seed, coeff in tqdm(list(itertools.product(seeds, coeffs))):
    fig, _ = plot_patched_vfield(seed, coeff)
    fig.savefig(f"../figures/patched_vfield_seed{seed}_coeff{coeff}.png", dpi=300)
    plt.clf()
    plt.close()
# %% Custom value source via hand-edited maze
@interact 
def custom_values(seed=IntSlider(min=0, max=100, step=1, value=0)):
    global v_env # TODO this seems to not play nicely if you change original seed? Other mazes are negligibly affected
    v_env = get_custom_venvs(seed=seed)
# %% Use these values in desired mazes
# Assumes a fixed venv, hook, values, and label
@interact
def interactive_patching(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-3, max=3, step=0.05, value=-1)):
    values = values_from_venv(v_env, hook, label)
    fig, _, _ = plot_patched_vfield(seed, coeff, label, hook, values=values)
    plt.show()

# %% Check behavior in custom target maze
values = values_from_venv(v_env, hook, label)
target_env = get_custom_venvs(seed=0)
# %%
fig, _, _ = plot_patched_vfield(0, -1, label, hook, values=values, venv=target_env)
plt.show()

# %%
fig, _, _ = plot_patched_vfield(0, -1, label, hook, values=values, venv=v_env)


# %% Try various off-distribution levels (e.g. with just cheese)

# %% Sweep all levels using patches gained from each level
for seed in range(50):
    run_seed(seed, hook, interesting_coeffs)

# %% Average diff over a bunch of seeds
values = np.zeros_like(cheese_diff_values(0, label, hook))
seeds = slice(int(10e5),int(10e5+100))
# Iterate over range specified by slice
for seed in range(seeds.start, seeds.stop):
    # Make values be rolling average of values from seeds
    values = (seed-seeds.start)/(seed-seeds.start+1)*values + cheese_diff_values(seed, label, hook)/(seed-seeds.start+1)

for seed in range(20):
    run_seed(seed, hook, [-1], values_tup=(values, f'avg from {seeds.start} to {seeds.stop}'))

# %% Generate a random values vector and then patch it in
values = t.rand_like(t.from_numpy(cheese_diff_values(0, label, hook))).numpy()
for seed in range(20):
    run_seed(seed, hook, [-1], values_tup=(values, 'garbage'))

# %% Try all labels for a fixed seed and diff_coeff
labels = list(hook.values_by_label.keys()) # TODO this dict was changing in size during the loop, but why?
# Interactive function to run all labels
@interact
def run_all_labels(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-3, max=3, step=0.1, value=-1), label=labels):
    fig, _, _ = plot_patched_vfield(seed, coeff, label, hook)
    plt.show()

# %% 
# Print the structure of the network
print(policy)

# %%
