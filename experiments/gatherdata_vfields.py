# %%
# Imports
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt

import circrl.module_hook as cmh

import procgen_tools.models as models

# %% 
# Load two levels and get values
from procgen_tools.patch_utils import *
# %%
# Save vector fields and figures for a bunch of (seed, coeff) pairs

if __name__ == '__main__':
    import itertools, pickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='trained_models/maze_I/model_rand_region_5.pth')
    parser.add_argument('--label', type=str, default='embedder.block2.res1.resadd_out')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_levels', type=int, default=100)
    parser.add_argument('--coeffs', type=str, default='-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3')
    parser.add_argument('--save-figures', default=True, action='store_false')
    args = parser.parse_args()

    rand_region = 5
    device = t.device(args.device)
    seeds = range(args.num_levels)
    coeffs = [float(c) for c in args.coeffs.split(',')]
    label = args.label
    path_prefix = ''

    policy = models.load_policy(path_prefix + args.model_file, 15, device)
    hook = cmh.ModuleHook(policy)

    for seed, coeff in tqdm(list(itertools.product(seeds, coeffs))):
        fig, _, obj = plot_patched_vfields(seed, coeff, label, hook)
        name = f"seed-{seed}_coeff-{coeff}_rr-{rand_region}_label-{label}"
        with open(f'{path_prefix}data/vfields/{name}.pkl', 'wb') as fp:
            del obj['patches'] # can't pickle lambda
            pickle.dump(obj, fp)

        if args.save_figures:
            fig.savefig(f"{path_prefix}figures/{name}.png")
        plt.clf()
        plt.close()

