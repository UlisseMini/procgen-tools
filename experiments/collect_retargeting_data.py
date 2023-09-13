# %%
try:
    import procgen_tools
except ImportError:
    get_ipython().run_line_magic(
        magic_name="pip",
        line="install -U git+https://github.com/ulissemini/procgen-tools",
    )

from procgen_tools.utils import setup

setup()  # create directory structure and download data

from procgen_tools.imports import *
from procgen_tools import maze, visualization, models, patch_utils
from typing import Tuple, Dict, List, Optional, Union
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import itertools, pickle
import argparse
import tqdm
import multiprocessing

import circrl.module_hook as cmh

SAVE_DIR = "experiments/statistics/data/retargeting"
AX_SIZE = 6

cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
effective_channels = [8, 55, 77, 82, 88, 89, 113]

# %% [markdown]
# # Showing mean probability of reaching each part of the maze


# Retargeting helpers
def retarget_to_square(
    venv,
    hook,
    channels: List[int],
    coord: Tuple[int, int],
    inner_coord: Tuple[int, int],
    magnitude: float = 5.5,
    default: Optional[float] = None,
) -> float:
    """Create a hook and retarget the given channels to the given
    square, returning the geometric average of the probabilities from
    the origin to that square.

    Args:
        venv: Vectorized environment
        hook: Hook to the network
        channels: List of channels to retarget
        coord: Coordinate of the channel position to retarget to
        inner_coord: Coordinate of the inner grid position to retarget to
        magnitude: Magnitude of the retargeting
        default: Default value to use for the retargeted channels,
        outside of the coord
    """
    patches = patch_utils.combined_pixel_patch(
        layer_name=default_layer,
        channels=channels,
        value=magnitude,
        coord=coord,
        default=default,
    )

    with hook.use_patches(patches):
        vf: Dict = visualization.vector_field(venv, hook.network)

    return maze.geometric_probability_path((0, 0), inner_coord, vf)


def cheese_at_square(
    venv,
    hook,
    inner_coord: Tuple[int, int],
    filter_coord: Optional[Tuple[int, int]],
) -> float:
    """Returns the probability of navigating to a square, given that
    the cheese is placed at the given square."""
    grid: np.ndarray = maze.state_from_venv(venv).inner_grid()
    padding: int = maze.get_padding(grid)
    new_coord: Tuple[int, int] = (
        inner_coord[0] + padding,
        inner_coord[1] + padding,
    )
    moved_venv = maze.move_cheese(venv, new_coord)
    vf: Dict = visualization.vector_field(moved_venv, hook.network)
    return maze.geometric_probability_path((0, 0), inner_coord, vf)


# Define the main processing routine for a single seed in a function
def process_seed(seed, args, SAVE_DIR, effective_channels, magnitude):
    filepath = os.path.join(SAVE_DIR, f"maze_retarget_{seed}.pkl")
    if os.path.exists(filepath):
        return

    with open(filepath, "wb") as fp:
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        new_venv = maze.remove_cheese(venv) if args.remove_cheese else venv

        data: pd.DataFrame = visualization.retarget_heatmap(
            new_venv,
            hook,
            retargeting_fn=retarget_to_square,
            channels=effective_channels,
            magnitude=magnitude,
            compute_normal=True,
        )

        if args.collect_cheese_data:
            cheese_data: pd.DataFrame = visualization.retarget_heatmap(
                new_venv,
                hook,
                retargeting_fn=cheese_at_square,
                compute_normal=False,
            )
            data = pd.concat([data, cheese_data])

        data["channels"] = effective_channels
        data["magnitude"] = magnitude
        data["removed_cheese"] = args.remove_cheese

        data["seed"] = seed

        data["retarget_prob"] = data["retarget_prob"].apply(np.ravel)
        data["ratio"] = data["retarget_prob"] / data["normal_prob"]
        data["diff"] = data["retarget_prob"] - data["normal_prob"]

        data.to_pickle(fp)


# %%
# Save retargeting data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="trained_models/maze_I/model_rand_region_5.pth",
    )
    parser.add_argument("--num_levels", type=int, default=100)
    parser.add_argument("--collect_cheese_data", type=bool, default=True)
    parser.add_argument("--remove_cheese", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    rand_region = 5
    device = t.device(args.device)
    seeds = range(args.num_levels)

    policy = models.load_policy(args.model_file, 15, device)
    hook = cmh.ModuleHook(policy)

    # Strength of the intervention
    magnitude: float = 2.3

    # Use a Pool to run the process_seed function across all seeds in parallel
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        pool.starmap(
            process_seed,
            [
                (seed, args, SAVE_DIR, effective_channels, magnitude)
                for seed in seeds
            ],
        )
