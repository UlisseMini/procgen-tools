# %%
# Imports

from procgen_tools import models, maze
import matplotlib.pyplot as plt
from procgen import ProcgenGym3Env
import torch


# %%
# Get model probs for every mouse position in the maze

def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)

def set_mouse_pos_env_state(state, pos):
    return state

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
    Plot the vector field induced by the policy on the maze in venv env number i.
    """
    assert venv.num_envs == 1, f'Did you forget to use maze.copy_venv to get a single env?'

    grid = maze.EnvState(venv.env.callmethod('get_state')[0]).inner_grid(with_mouse=False)
    legal_mouse_positions = [(x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1]) if grid[x, y] == maze.EMPTY]
    obs_list = []
    for pos in legal_mouse_positions:
        set_mouse_pos(venv, pos)
        obs = venv.reset()
        obs_list.append(torch.tensor(obs[0], dtype=torch.float32))

    # use stacked obs list as a tensor
    with torch.no_grad():
        c, _ = policy(torch.stack(obs_list).to(_device(policy)))

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
    venv = maze.copy_venv(venv, env_num)

    vf = vector_field(venv, policy)
    arrows, legal_mouse_positions, grid = vf['arrows'], vf['legal_mouse_positions'], vf['grid']

    ax = ax if ax is not None else plt.gca()

    # ax.quiver(legal_mouse_positions, arrows, color='red')
    ax.quiver([x[1] for x in legal_mouse_positions], [x[0] for x in legal_mouse_positions], [x[1] for x in arrows], [x[0] for x in arrows], color='red')
    ax.imshow(grid, origin='lower')
    # ax.imshow(venv.env.get_info()[0]['rgb'])

    return vf


# %%
# Load policy and maze, then plot vector field for a bunch of mazes

if __name__ == '__main__':
    from tqdm import tqdm

    rand_region = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = models.load_policy(f'../trained_models/maze_I/model_rand_region_{rand_region}.pth', 15, torch.device('cpu'))
    policy.to(device)

    venv = ProcgenGym3Env(num=10, start_level=0, num_levels=0, env_name='maze', distribution_mode='hard', num_threads=1, render_mode='rgb_array')
    venv = maze.wrap_venv(venv)

    for i in tqdm(range(venv.num_envs)):
        plt.clf()
        plot_vector_field(venv, policy, env_num=i)
        plt.savefig(f'../figures/maze_{i}_vfield.png', dpi=300)
        plt.close()

# %%
