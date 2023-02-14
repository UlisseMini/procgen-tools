# %%
# Imports

from procgen_tools import models, maze
import matplotlib.pyplot as plt
from ipywidgets import *
from IPython.display import display, clear_output
from typing import Callable

from procgen import ProcgenGym3Env
from warnings import warn
import torch
import os


# %%
# Get model probs for every mouse position in the maze

def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)

def set_mouse_pos(venv, pos, env_num=0):
    "FIXME: This should be in a library, and this should be two lines with more enlightened APIs."
    state_bytes_list = venv.env.callmethod('get_state')

    state = maze.EnvState(state_bytes_list[env_num])
    grid = state.inner_grid(with_mouse=False)
    assert grid[pos] == maze.EMPTY
    grid[pos] = maze.MOUSE
    state.set_grid(grid, pad=True)

    state_bytes_list[env_num] = state.state_bytes
    venv.env.callmethod('set_state', state_bytes_list)

# %%
# Get vector field

# really stupid way to do this tbh, should use numpy somehow
def _tmul(tup: tuple, s: float):
    return tuple(s * x for x in tup)
def _tadd(*tups):
    return tuple(sum(axis) for axis in zip(*tups))
def _device(policy):
    return next(policy.parameters()).device


# TODO: with the right APIs, this should be a few lines
def vector_field(venv, policy):
    """
    Plot the vector field induced by the policy on the maze in venv env number 1.
    """
    return vector_field_tup(maze.venv_with_all_mouse_positions(venv), policy)


def get_arrows_and_probs(legal_mouse_positions, c_probs):
    # FIXME: Vectorize this loop. It isn't as critical as the model though
    arrows = []
    probs = []
    for i in range(len(legal_mouse_positions)):
        probs_dict = models.human_readable_actions(c_probs[i])
        probs_dict = {k: v.item() for k, v in probs_dict.items()}
        deltas = [_tmul(models.MAZE_ACTION_DELTAS[act], p) for act, p in probs_dict.items()]
        arrows.append(_tadd(*deltas))
        probs.append(tuple(probs_dict.values()))
    return arrows, probs


# TODO: with the right APIs, this should be a few lines
def vector_field_tup(venv_all_tup, policy):
    """
    Plot the vector field induced by the policy on the maze in venv env number 1.
    """
    venv_all, (legal_mouse_positions, grid) = venv_all_tup

    # TODO: Hypothetically, this step could run in parallel to the others (cpu vs. gpu)
    batched_obs = torch.tensor(venv_all.reset(), dtype=torch.float32, device=_device(policy))
    del venv_all

    # use stacked obs list as a tensor
    with torch.no_grad():
        c, _ = policy(batched_obs)

    arrows, probs = get_arrows_and_probs(legal_mouse_positions, c.probs)

    # make vfield object for returning
    return {'arrows': arrows, 'legal_mouse_positions': legal_mouse_positions, 'grid': grid, 'probs': probs}



# %%
# Plot vector field for every mouse position

def plot_vector_field(venv, policy, ax=None, env_num=0):
    """
    Plot the vector field induced by the policy on the maze in venv env number i.
    """
    warn('Deprecated in favor of calling vector_field and plot_vf directly.')
    venv = maze.copy_venv(venv, env_num)
    vf = vector_field(venv, policy)
    return plot_vf(vf, ax=ax)

def render_arrows(vf : dict, ax=None, human_render: bool = True, render_padding : bool = False, color : str = 'white'):
    """ Render the arrows in the vector field. """
    ax = ax or plt.gca()

    arrows, legal_mouse_positions, grid = vf['arrows'], vf['legal_mouse_positions'], vf['grid']

    ax.quiver(
        [pos[1] for pos in legal_mouse_positions], [pos[0] for pos in legal_mouse_positions],
        [arr[1] for arr in arrows], [arr[0] for arr in arrows], color=color, scale=1, scale_units='xy'
    )

    if human_render:
        human_view = maze.render_outer_grid(grid) if render_padding else maze.render_inner_grid(grid)
        ax.imshow(human_view)
    else: 
        ax.imshow(grid, origin='lower')

    ax.set_xticks([])
    ax.set_yticks([])

def map_vf_to_human(vf : dict, render_padding : bool = False):
    "Map the vector field vf to the human view coordinate system."
    legal_mouse_positions, arrows, grid = vf['legal_mouse_positions'], vf['arrows'], vf['grid']

    # We need to transform the arrows to the human view coordinate system
    human_view = maze.render_outer_grid(grid)

    padding = maze.WORLD_DIM - grid.shape[0] 
    assert padding % 2 == 0
    padding //= 2
    rescale = human_view.shape[0] / maze.WORLD_DIM

    legal_mouse_positions = [((grid.shape[1] - 1) - row, col) for row, col in legal_mouse_positions] # flip y axis
    if render_padding: 
        legal_mouse_positions = [(row + padding, col + padding) for row, col in legal_mouse_positions]
    legal_mouse_positions = [((row+.5) * rescale, (col+.5) * rescale) for row, col in legal_mouse_positions]
    arrows = [_tmul(arr, rescale) for arr in arrows]

    return {'arrows': arrows, 'legal_mouse_positions': legal_mouse_positions, 'grid': grid}

def plot_vf(vf: dict, ax=None, human_render : bool = True, render_padding: bool = False):
    "Plot the vector field given by vf. If human_render is true, plot the human view instead of the raw grid np.ndarray."
    render_arrows(map_vf_to_human(vf, render_padding=render_padding) if human_render else vf, ax=ax, human_render=human_render, render_padding=render_padding, color='white' if human_render else 'red')

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

def plot_vf_diff(vf1 : dict, vf2 : dict, ax=None, human_render : bool = True, render_padding : bool = False): 
    """ Render the difference "vf1 - vf2" between two vector fields, plotting only the difference. """
    # Remove cheese from the legal mouse positions and arrows, if levels are otherwise the same 
    vf_diff = get_vf_diff(vf1, vf2)

    render_arrows(map_vf_to_human(vf_diff, render_padding=render_padding) if human_render else vf_diff, ax=ax, human_render=human_render, render_padding=render_padding, color='lime' if human_render else 'red')

    return vf_diff

def plot_vfs(vf1 : dict, vf2 : dict, human_render : bool = True, render_padding : bool = False, ax_size : int = 5, show_diff : bool = True):
    """ Plot two vector fields and, if show_diff is True, their difference vf2 - vf1. Plots three axes in total. Returns the figure, axes, and the difference vector field."""
    num_cols = 3 if show_diff else 2
    fig, axs = plt.subplots(1, num_cols, figsize=(ax_size*num_cols, ax_size))

    axs[0].set_xlabel("Original")
    plot_vf(vf1, ax=axs[0], human_render=human_render, render_padding=render_padding)
    
    axs[1].set_xlabel("Patched")
    plot_vf(vf2, ax=axs[1], human_render=human_render, render_padding=render_padding)
    
    if show_diff:
        axs[2].set_xlabel("Patched vfield minus original")
        # Pass in vf2 first so that the difference is vf2 - vf1, or the difference between the patched and original vector fields
        vf_diff = plot_vf_diff(vf2, vf1, ax=axs[2], human_render=human_render, render_padding=render_padding)
    return fig, axs, vf_diff if show_diff else None


# %%
# Load policy and maze, then plot vector field for a bunch of mazes

if __name__ == '__main__':
    import pickle
    from tqdm import tqdm

    rand_region = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = models.load_policy(f'../trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, torch.device('cpu'))
    policy.to(device)

    venv = ProcgenGym3Env(num=100, start_level=1, num_levels=0, env_name='maze', distribution_mode='hard', num_threads=1, render_mode='rgb_array')
    venv = maze.wrap_venv(venv)

    for i in tqdm(range(venv.num_envs)):
        plt.clf()
        vf_new = plot_vector_field(venv, policy, env_num=i)
        # plt.show()
        plt.savefig(f'../figures/maze_{i}_vfield.png', dpi=300)
        plt.close()

        # TESTING
        # vf_old = _vector_field_old(maze.copy_venv(venv, i), policy)
        # assert pickle.dumps(vf_new) == pickle.dumps(vf_old), f'pickle compare failed for i={i}'

# %%

