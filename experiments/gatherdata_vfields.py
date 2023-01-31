# %%
# Imports
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh

import procgen_tools.models as models

# %% 
# Load two levels and get values
from patch_utils import *

# %%
# Load model

path_prefix = '../' if in_jupyter else ''
policy = models.load_policy(path_prefix + f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 15,
    t.device('cpu'))
hook = cmh.ModuleHook(policy)

# the one that works...
label = 'embedder.block2.res1.resadd_out'

# %%
# Save vector fields and figures for a bunch of (seed, coeff) pairs


import itertools, pickle
seeds = range(100)
coeffs = [-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3]
for seed, coeff in tqdm(list(itertools.product(seeds, coeffs))):
    fig, _, obj = plot_patched_vfield(seed, coeff, label, hook)
    name = f"patched_vfield_seed{seed}_coeff{coeff}"
    with open(f'../data/vfields/{name}.pkl.gz', 'wb') as fp:
        pickle.dump(obj, fp)

    fig.savefig(f"../figures/{name}.png")
    plt.clf()
    plt.close()
# %%
