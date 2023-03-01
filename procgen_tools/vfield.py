from procgen_tools import models, maze
import matplotlib.pyplot as plt
from ipywidgets import *
from IPython.display import display, clear_output
from typing import Callable, List, Tuple

from procgen import ProcgenGym3Env
from warnings import warn
import torch
from torch import nn
import numpy as np
import os


def forward_func_policy(network : nn.Module, inp : torch.Tensor):
    """ Forward function for the policy network. """
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)

# %%
# Get vector field

# FIXME really stupid way to do this tbh, should use numpy somehow
def _tmul(tup: tuple, scalar: float):
    """ Multiply a tuple by a scalar. """
    return tuple(scalar * x for x in tup)

def _tadd(*tups : List[Tuple[int,int]]):
    """ Add a list of tuples elementwise. """
    return tuple(sum(axis) for axis in zip(*tups))

def _device(policy : nn.Module):
    return next(policy.parameters()).device


def vector_field(venv : ProcgenGym3Env, policy : nn.Module):
    """
    Get the vector field induced by the policy on the maze in venv env number 1.
    """
    venv_all, (legal_mouse_positions, grid) = maze.venv_with_all_mouse_positions(venv)
    return vector_field_tup(venv_all, legal_mouse_positions, grid, policy)


def get_arrows_and_probs(legal_mouse_positions : List[Tuple[int, int]], c_probs : torch.Tensor) -> List[dict]: 
    """ Get the arrows and probabilities for each mouse position. 
    
    Args:
        legal_mouse_positions: A list of (x, y) tuples, each assumed to be an outer grid coordinate. 
        c_probs: A tensor of shape (len(legal_mouse_positions), 15) of post-softmax probabilities, one for each mouse position.

    Returns:
        action_arrows: A list of lists of probability-weighted basis vectors -- an (x, y) tuple, one for each mouse position
        probs: A list of dicts of action -> probability, one for each mouse position
    """
    # FIXME: Vectorize this loop. It isn't as critical as the model though
    action_arrows, probs = [], []
    for i in range(len(legal_mouse_positions)):
        # Dict of action -> probability for this mouse position
        probs_dict = models.human_readable_actions(c_probs[i]) 
        # Convert to floats
        probs_dict = {k: v.item() for k, v in probs_dict.items()} 
        
        # Multiply each basis vector by the probability of that action, and append this list of action component arrows
        action_arrows.append([_tmul(models.MAZE_ACTION_DELTAS[act], p) for act, p in probs_dict.items()]) 
        # Append the {(action : str): (probability : float)} dict
        probs.append(tuple(probs_dict.values()))

    return action_arrows, probs

def vector_field_tup(venv_all : ProcgenGym3Env, legal_mouse_positions : List[Tuple[int, int]], grid : np.ndarray, policy : nn.Module):
    """
    Plot the vector field induced by the policy on the maze in venv env number 1.

    Args:
        venv_all: The venv to use to get the grid and legal mouse positions. Deleted after use.
        legal_mouse_positions: a list of (x, y) tuples, each assumed to be an outer grid coordinate.
        grid: The outer grid to use to compute the vector field.
        policy: The policy to use to compute the vector field.
    """
    # TODO: Hypothetically, this step could run in parallel to the others (cpu vs. gpu)
    batched_obs = torch.tensor(venv_all.reset(), dtype=torch.float32, device=_device(policy))
    del venv_all

    # use stacked obs list as a tensor
    with torch.no_grad():
        categorical, _ = policy(batched_obs)

    action_arrows, probs = get_arrows_and_probs(legal_mouse_positions, categorical.probs)

    # make vfield object for returning
    return {'arrows': action_arrows, 'legal_mouse_positions': legal_mouse_positions, 'grid': grid, 'probs': probs}



# %%
# Plot vector field for every mouse position
def plot_vector_field(venv : ProcgenGym3Env, policy : nn.Module, ax : plt.Axes = None, env_num : int = 0):
    """
    Plot the vector field induced by the policy on the maze in venv env number i.
    """
    warn('Deprecated in favor of calling vector_field and plot_vf directly.')
    venv = maze.copy_venv(venv, env_num)
    vf = vector_field(venv, policy)
    return plot_vf(vf, ax=ax)

def render_arrows(vf : dict, ax=None, human_render: bool = True, render_padding : bool = False, color : str = 'white', show_components : bool = False):
    """ Render the arrows in the vector field. 
    
    args:
        vf: The vector field dict
        ax: The matplotlib axis to render on
        human_render: Whether to render the grid in a human-readable way (high-res pixel view) or a machine-readable way (grid view)
        render_padding: Whether to render the padding around the grid
        color: The color of the arrows
        show_components: Whether to show one arrow for each cardinal action. If False, show one arrow for each action.
    """
    ax = ax or plt.gca()

    arrows, legal_mouse_positions, grid = vf['arrows'], vf['legal_mouse_positions'], vf['grid']

    inner_size = grid.shape[0] # The size of the inner grid
    arrow_rescale = maze.WORLD_DIM / (inner_size * 1.8) # Rescale arrow width and other properties to be relative to the size of the maze 
    width = .005 * arrow_rescale
    if show_components:
        # A list of length-four lists of (x, y) tuples, one for each mouse position
        for idx, tile_arrows in enumerate(arrows):
            ax.quiver(
                [legal_mouse_positions[idx][1]] * len(tile_arrows), [legal_mouse_positions[idx][0]] * len(tile_arrows),
                [arr[1] for arr in tile_arrows], [arr[0] for arr in tile_arrows], color=color, scale=1, scale_units='xy', width=width
            )

    else:
        arrows = [_tadd(*arr_list) for arr_list in arrows] # Add the arrows together to get a total vector for each mouse position
        ax.quiver(
            [pos[1] for pos in legal_mouse_positions], [pos[0] for pos in legal_mouse_positions],
            [arr[1] for arr in arrows], [arr[0] for arr in arrows], color=color, scale=1, scale_units='xy', width=width
        )

    if human_render:
        human_view = maze.render_outer_grid(grid) if render_padding else maze.render_inner_grid(grid)
        ax.imshow(human_view)
    else: 
        ax.imshow(grid, origin='lower')

    ax.set_xticks([])
    ax.set_yticks([])

def map_vf_to_human(vf : dict, account_for_padding : bool = False):
    """Map the vector field vf to the human view coordinate system.
    
    Args:
        vf: A vector field dict with the maze coordinate system.
        account_for_padding: Whether to account for the padding in the human view coordinate system.

    Returns:
        vf: A vector field dict with the human view coordinate system.
    """
    legal_mouse_positions, arrows, grid = vf['legal_mouse_positions'], vf['arrows'], vf['grid']

    # We need to transform the arrows to the human view coordinate system
    human_view = maze.render_outer_grid(grid)

    padding = maze.WORLD_DIM - grid.shape[0] 
    assert padding % 2 == 0
    padding //= 2
    rescale = human_view.shape[0] / maze.WORLD_DIM

    legal_mouse_positions = [((grid.shape[1] - 1) - row, col) for row, col in legal_mouse_positions] # flip y axis
    if account_for_padding: 
        legal_mouse_positions = [(row + padding, col + padding) for row, col in legal_mouse_positions]
    legal_mouse_positions = [((row+.5) * rescale, (col+.5) * rescale) for row, col in legal_mouse_positions]
    arrows = [[_tmul(arr, rescale) for arr in arr_list] for arr_list in arrows]

    return {'arrows': arrows, 'legal_mouse_positions': legal_mouse_positions, 'grid': grid}

def plot_vf(vf: dict, ax=None, human_render : bool = True, render_padding: bool = False, show_components : bool = False):
    "Plot the vector field given by vf. If human_render is true, plot the human view instead of the raw grid np.ndarray."
    render_arrows(map_vf_to_human(vf, account_for_padding=render_padding) if human_render else vf, ax=ax, human_render=human_render, render_padding=render_padding, color='white' if human_render else 'red', show_components=show_components)

def get_vf_diff(vf1 : dict, vf2 : dict):
    """ Get the difference "vf1 - vf2" between two vector fields. """
    def assert_compatibility(vfa, vfb):
        assert vfa['legal_mouse_positions'] == vfb['legal_mouse_positions'], "Legal mouse positions must be the same to render the vf difference."
        assert vfa['grid'].shape == vfb['grid'].shape, "Grids must be the same shape to render the vf difference."
        assert len(vfa['arrows']) == len(vfb['arrows']), "Arrows must be the same length to render the vf difference."
    
    # Remove cheese from the legal mouse positions and arrows, if levels are otherwise the same 
    for i in range(2):
        try: 
            assert_compatibility(vf1, vf2)
        except: 
            if (vf1['grid'] == maze.CHEESE).any():
                cheese_vf_idx = 0 
            elif (vf2['grid'] == maze.CHEESE).any():
                cheese_vf_idx = 1
            else:
                raise ValueError("Levels are not the same, but neither has cheese.")

            vfs = [vf1, vf2]
            cheese_location = maze.get_cheese_pos(vfs[cheese_vf_idx]['grid'])

            # Remove cheese from the legal mouse positions and arrows
            other_vf_idx = 1 - cheese_vf_idx
            vfs[other_vf_idx]['arrows'] = [arr for pos, arr in zip(vfs[other_vf_idx]['legal_mouse_positions'], vfs[other_vf_idx]['arrows']) if pos != cheese_location]
            vfs[other_vf_idx]['legal_mouse_positions'] = [pos for pos in vfs[other_vf_idx]['legal_mouse_positions'] if pos != cheese_location]

    arrow_diffs = [_tmul((a1[0] - a2[0], a1[1] - a2[1]), 1) for a1, a2 in zip(vf1['arrows'], vf2['arrows'])] # Halve the difference so it's easier to see
    
    # Check if any of the diffs have components greater than 2 (which would be a bug)
    assert all(abs(a[0]) <= 2 and abs(a[1]) <= 2 for a in arrow_diffs), "Arrow diffs must be less than 2 in each component."

    return {'arrows': arrow_diffs, 'legal_mouse_positions': vf2['legal_mouse_positions'], 'grid': vf2['grid']}

def plot_vf_diff(vf1 : dict, vf2 : dict, ax : plt.Axes = None, human_render : bool = True, render_padding : bool = False, show_components : bool = False): 
    """ Render the difference "vf1 - vf2" between two vector fields, plotting only the difference. """
    # Remove cheese from the legal mouse positions and arrows, if levels are otherwise the same 
    vf_diff = get_vf_diff(vf1, vf2)

    render_arrows(map_vf_to_human(vf_diff, account_for_padding=render_padding) if human_render else vf_diff, ax=ax, human_render=human_render, render_padding=render_padding, color='lime' if human_render else 'red', show_components=show_components)

    return vf_diff

def plot_vfs(vf1 : dict, vf2 : dict, human_render : bool = True, render_padding : bool = False, ax_size : int = 5, show_diff : bool = True, show_original : bool = True):
    """ Plot two vector fields and, if show_diff is True, their difference vf2 - vf1. Plots three axes in total. Returns the figure, axes, and the difference vector field. If show_original is False, don't plot the original vector field. """
    num_cols = 1 + show_diff + show_original
    fontsize = 16
    fig, axs = plt.subplots(1, num_cols, figsize=(ax_size*num_cols, ax_size))

    idx = 0
    if show_original:
        axs[idx].set_xlabel("Original", fontsize=fontsize)
        plot_vf(vf1, ax=axs[0], human_render=human_render, render_padding=render_padding)
        idx += 1
    
    axs[idx].set_xlabel("Patched", fontsize=fontsize)
    plot_vf(vf2, ax=axs[idx], human_render=human_render, render_padding=render_padding)
    idx += 1

    if show_diff:
        axs[idx].set_xlabel("Patched vfield minus original", fontsize=fontsize)
        # Pass in vf2 first so that the difference is vf2 - vf1, or the difference between the patched and original vector fields
        vf_diff = plot_vf_diff(vf2, vf1, ax=axs[idx], human_render=human_render, render_padding=render_padding)
    return fig, axs, (vf_diff if show_diff else None)
# %%
