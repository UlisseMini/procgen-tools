# %%
# Imports
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt

import circrl.module_hook as cmh

import procgen_tools.models as models
from procgen_tools import maze, patch_utils

# %%
# Load two levels and get values
from procgen_tools.patch_utils import *

# %%
# Save vector fields and figures for a bunch of (seed, coeff) pairs; run
# this from base directory

if __name__ == "__main__":
    import itertools, pickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="trained_models/maze_I/model_rand_region_5.pth",
    )
    parser.add_argument(
        "--label", type=str, default="embedder.block2.res1.resadd_out"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_levels", type=int, default=100)
    parser.add_argument(
        "--coeffs", type=str, default="-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3"
    )
    parser.add_argument("--save-figures", default=True, action="store_false")
    parser.add_argument("--vector_type", type=str, default="cheese")
    args = parser.parse_args()

    rand_region = 5
    device = t.device(args.device)
    seeds = range(args.num_levels)
    coeffs = [float(c) for c in args.coeffs.split(",")]
    label = args.label
    path_prefix = "experiments/statistics/"

    policy = models.load_policy(args.model_file, 15, device)
    hook = cmh.ModuleHook(policy)

    for seed, coeff in tqdm(list(itertools.product(seeds, coeffs))):
        name = f"seed-{seed}_coeff-{coeff}_rr-{rand_region}_label-{label}"
        filepath = f"{path_prefix}data/vfields/{args.vector_type}/{name}.pkl"
        if os.path.exists(filepath):
            continue
        values = None
        if args.vector_type == "top_right":
            venv_pair = maze.get_top_right_venv_pair(seed=seed)
            values = patch_utils.values_from_venv(
                layer_name=label, venv=venv_pair, hook=hook
            )

        fig, _, obj = plot_patched_vfields(
            seed, coeff, label, hook, values=values
        )
        with open(filepath, "wb") as fp:
            del obj["patches"]  # can't pickle lambda
            pickle.dump(obj, fp)

        if args.save_figures:
            fig.savefig(f"figures/{args.vector_type}_vfs/{name}.png")
        plt.clf()
        plt.close()
