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

def set_mouse_pos(venv, pos, env_num=0):
    "FIXME: This should be in a library, and this should be two lines."
    state_bytes_list = venv.env.callmethod('get_state')
    state = maze.EnvState(state_bytes_list[env_num])
    grid = state.inner_grid(with_mouse=False)
    assert grid[pos] == maze.EMPTY
    grid[pos] = maze.MOUSE
    state.set_grid(grid, pad=True)
    state_bytes_list[env_num] = state.state_bytes
    venv.env.callmethod('set_state', state_bytes_list)


# %%
# Plot vector field for every mouse position

def plot_vector_field(venv, policy, env_num=0):
    """
    Plot the vector field induced by the policy on the maze in venv env number i.
    """
    i = env_num

    # really stupid way to do this tbh, should use numpy somehow
    def tmul(tup: tuple, s: float):
        return tuple(s * x for x in tup)
    def tadd(*tups):
        return tuple(sum(axis) for axis in zip(*tups))

    arrows = []

    plt.clf()

    grid = maze.EnvState(venv.env.callmethod('get_state')[env_num]).inner_grid(with_mouse=False)
    legal_mouse_positions = [(x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1]) if grid[x, y] == maze.EMPTY]
    for pos in legal_mouse_positions:
        set_mouse_pos(venv, pos, env_num)
        obs = venv.reset()

        c, _ = policy(torch.Tensor(obs[env_num]).unsqueeze(0))
        probs_dict = models.human_readable_actions(c)
        probs_dict = {k: v[0].item() for k, v in probs_dict.items()}
        deltas = [tmul(models.MAZE_ACTION_DELTAS[act], p) for act, p in probs_dict.items()]
        arrows.append(tadd(*deltas))


    # plt.quiver(legal_mouse_positions, arrows, color='red')
    plt.quiver([x[1] for x in legal_mouse_positions], [x[0] for x in legal_mouse_positions], [x[1] for x in arrows], [x[0] for x in arrows], color='red')
    plt.imshow(grid, origin='lower')
    # plt.imshow(venv.env.get_info()[0]['rgb'])

# %%


policy = models.load_policy(f'../trained_models/maze_I/model_rand_region_5.pth', 15, torch.device('cpu'))
venv = ProcgenGym3Env(num=2, start_level=0, num_levels=0, env_name='maze', distribution_mode='hard', num_threads=1, render_mode='rgb_array')
venv = maze.wrap_venv(venv)
obs = venv.reset()

for i in range(venv.num_envs):
    plot_vector_field(venv, policy, i=i)
    plt.show()


# %%
# Load policy and maze, then plot vector field for a bunch of mazes
# FIXME(uli): There's a memory leak in plot_vector_field, num_envs=100 barely works on my 16GB machine.

if __name__ == '__main__':
    from tqdm import tqdm

    policy = models.load_policy(f'../trained_models/maze_I/model_rand_region_5.pth', 15, torch.device('cpu'))
    venv = ProcgenGym3Env(num=10, start_level=0, num_levels=0, env_name='maze', distribution_mode='hard', num_threads=1, render_mode='rgb_array')
    venv = maze.wrap_venv(venv)
    for i in tqdm(range(venv.num_envs)):
        plot_vector_field(venv, policy, env_num=i)
        plt.savefig(f'../figures/maze_{i}_vfield.png')
        plt.close()

# %%
