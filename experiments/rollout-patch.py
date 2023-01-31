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
from ipywidgets import Text, interact
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
path_prefix = '../' if in_jupyter else ''

# %%
# Load model

policy = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15,
    t.device('cpu'))

# %% 
label = 'embedder.block2.res1.resadd_out'
interesting_coeffs = np.linspace(-3,3,10) # NOTE may break assumption that one of these is 0
hook = cmh.ModuleHook(policy)

# %%
# Interactive mode

from ipywidgets import interact, IntSlider, fixed, FloatSlider

@interact
def interactive_patching(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-3, max=3, step=0.1, value=1)):
    fig, _, _ = plot_patched_vfield(seed, coeff, label, hook)
    plt.show()


# %% RUN ABOVE here
# Try using one patch for many levels at different strengths
value_seed = 0
values_tup = get_values(value_seed, label, hook) 

for seed in range(0):  
    run_seed(seed, hook, interesting_coeffs, values_tup=values_tup)

# %%
# Save figures for a bunch of (seed, coeff) pairs
seeds = range(10)
coeffs = [-2, -1, -0.5, 0.5, 1, 2]
for seed, coeff in tqdm(list(itertools.product(seeds, coeffs))):
    fig, _ = plot_patched_vfield(seed, coeff)
    fig.savefig(f"../figures/patched_vfield_seed{seed}_coeff{coeff}.png", dpi=300)
    plt.clf()
    plt.close()
# %% 
# Try different activations

# %%
# Sweep all levels using patches gained from each level
for seed in range(50):
    run_seed(seed, hook, interesting_coeffs)

# %% 
# Average diff over a bunch of seeds
values = np.zeros_like(get_values(0, label, hook)[0])
seeds = slice(int(10e5),int(10e5+100))
# Iterate over range specified by slice
for seed in range(seeds.start, seeds.stop):
    # Make values be rolling average of values from seeds
    values = (seed-seeds.start)/(seed-seeds.start+1)*values + get_values(seed, label, hook)[0]/(seed-seeds.start+1)

for seed in range(20):
    run_seed(seed, hook, interesting_coeffs, values_tup=(values, f'avg from {seeds.start} to {seeds.stop}'))

# %% 
# Generate a random values vector and then patch it in
values = t.rand_like(t.from_numpy(get_values(0, label, hook)[0])).numpy()
for seed in range(20):
    run_seed(seed, hook, interesting_coeffs, values_tup=(values, 'garbage'))

# %% Try adding the cheese vector 
# Average diff over a bunch of seeds
values = np.zeros_like(get_values(0, label, hook)[0])
seeds = slice(int(10e5),int(10e5+100))
# Iterate over range specified by slice
for seed in range(seeds.start, seeds.stop):
    # Make values be rolling average of values from seeds
    values = (seed-seeds.start)/(seed-seeds.start+1)*values + get_values(seed, label, hook)[0]/(seed-seeds.start+1)
for seed in range(20):
    run_seed(seed, hook, -1 * np.array(interesting_coeffs), values_tup=(values, f'avg from {seeds.start} to {seeds.stop}'))


# %% 
# Try all labels for a fixed seed and diff_coeff
labels = list(hook.values_by_label.keys()) # TODO this dict was changing in size during the loop, but why?
for label in labels: 
    run_seed(0, hook, [1], label=label)

