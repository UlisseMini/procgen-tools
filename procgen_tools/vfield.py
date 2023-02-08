# %%
# Imports

from procgen_tools import models, maze
import matplotlib.pyplot as plt
from ipywidgets import *
from IPython.display import display, clear_output

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

def vector_field(venv, policy):
    """
    Plot the vector field induced by the policy on the maze in venv env number 1.
    """
    assert venv.num_envs == 1, f'Did you forget to use maze.copy_venv to get a single env?'

    env_state = maze.EnvState(venv.env.callmethod('get_state')[0])
    grid = env_state.inner_grid(with_mouse=False)
    legal_mouse_positions = [(x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1]) if grid[x, y] == maze.EMPTY]

    # convert coords from inner to outer grid coordinates
    assert (env_state.world_dim - grid.shape[0]) % 2 == 0
    padding = (env_state.world_dim - grid.shape[0]) // 2

    # create a venv for each legal mouse position
    state_bytes_list = []
    for (mx, my) in legal_mouse_positions:
        # we keep a backup of the state bytes for efficiency, as calling set_mouse_pos
        # implicitly calls _parse_state_bytes, which is slow. this is a hack.
        # NOTE: Object orientation hurts us here. It would be better to have functions.
        sb_back = env_state.state_bytes
        env_state.set_mouse_pos(mx+padding, my+padding)
        state_bytes_list.append(env_state.state_bytes)
        env_state.state_bytes = sb_back

    venv_all = maze.create_venv(
        num=len(legal_mouse_positions),
        num_threads=1 if len(legal_mouse_positions) < 100 else os.cpu_count(), # total bullshit
        num_levels=1, start_level=1
    )
    venv_all.env.callmethod('set_state', state_bytes_list)

    # TODO: Hypothetically, this step could run in parallel to the others (cpu vs. gpu)
    batched_obs = torch.tensor(venv_all.reset(), dtype=torch.float32)
    del venv_all

    # use stacked obs list as a tensor
    with torch.no_grad():
        c, _ = policy(batched_obs.to(_device(policy)))

    # FIXME: Vectorize this loop. It isn't as critical as the model though
    arrows = []
    probs = []
    for i in range(len(legal_mouse_positions)):
        probs_dict = models.human_readable_actions(c.probs[i])
        probs_dict = {k: v.item() for k, v in probs_dict.items()}
        deltas = [_tmul(models.MAZE_ACTION_DELTAS[act], p) for act, p in probs_dict.items()]
        arrows.append(_tadd(*deltas))
        probs.append(tuple(probs_dict.values()))

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

def plot_vf_diff(vf1 : dict, vf2 : dict, ax=None, human_render : bool = True, render_padding : bool = False): 
    """ Render the difference between two vector fields. """
    assert vf1['legal_mouse_positions'] == vf2['legal_mouse_positions'], "Legal mouse positions must be the same to render the vf difference."
    assert (vf1['grid'] == vf2['grid']).all(), "Grids must be the same to render the vf."

    arrow_diffs = [(a1[0] - a2[0], a1[1] - a2[1]) for a1, a2 in zip(vf1['arrows'], vf2['arrows'])] 
    
    # Check if any of the diffs have components greater than 2 (which would be a bug)
    assert all(abs(a[0]) <= 2 and abs(a[1]) <= 2 for a in arrow_diffs), "Arrow diffs must be less than 2 in each component."

    vf_diff = {'arrows': arrow_diffs, 'legal_mouse_positions': vf1['legal_mouse_positions'], 'grid': vf1['grid']}

    render_arrows(map_vf_to_human(vf_diff, render_padding=render_padding) if human_render else vf_diff, ax=ax, human_render=human_render, render_padding=render_padding, color='red' if human_render else 'blue')

def custom_vfield(policy : torch.nn.Module, seed : int = 0):
    """ Given a policy and a maze seed, create a maze editor and a vector field plot. Update the vector field whenever the maze is edited. Returns a VBox containing the maze editor and the vector field plot. """
    output = Output()
    fig, ax = plt.subplots(1,1, figsize=(3,3))
    plt.close()
    single_venv = maze.create_venv(num=1, start_level=seed, num_levels=1)

    # We want to update ax whenever the maze is edited
    def update_plot():
        # Clear the existing plot
        with output:
            vfield = vector_field(single_venv, policy)
            plot_vf(vfield, ax=ax)

            # Update the existing figure in place 
            clear_output(wait=True)
            display(fig)

    update_plot()

    # Then make a callback which updates the render in-place when the maze is edited
    editors = maze.venv_editors(single_venv, check_on_dist=False, env_nums=range(1), callback=lambda _: update_plot())

    # Display the maze editor and the plot in an HBox
    widget_vbox = VBox(editors + [output])
    return widget_vbox


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

