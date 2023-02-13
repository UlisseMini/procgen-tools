# %%

import matplotlib.pyplot as plt
from procgen_tools import maze, models, vfield
import circrl.module_hook as cmh
import torch as t
from ipywidgets import interact
from ipywidgets.widgets import IntSlider, Dropdown
from glob import glob

# %%

model_files = glob('../trained_models/**.pth')
default_model = next(f for f in model_files if 'rand_region_5' in f)

@interact
def plot_by_seed(
    seed = IntSlider(min=0, max=100, value=0),
    model_file = Dropdown(options=model_files, value=default_model),
):
    policy = models.load_policy(model_file, 15, 'cpu')
    # hook = cmh.ModuleHook(policy)

    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    venv_all, (legal_mouse_positions, grid) = maze.venv_with_all_mouse_positions(venv)
    obs_all = t.tensor(venv_all.reset(), dtype=t.float32)
    # obs_all.requires_grad = True
    # c, v = hook.network(obs_all)
    c, v = policy(obs_all)

    vf = vfield.vector_field(venv, policy)

    g = grid.copy()
    g[g==maze.BLOCKED] = 0 # 50 screws up plot
    g[g==maze.CHEESE] = 2.718**7 # 10 is more in-line with rest of things
    for i, pos in enumerate(legal_mouse_positions):
        g[pos] = v[i].exp().item()
    fig, ax = plt.subplots(1,1)
    fig.colorbar(ax.imshow(g, origin='lower', cmap='RdBu_r'), ax=ax)

    # Cross entropy between policy and argmax of value function.
    # Plot largest disagreements.

    cheese_pos = maze.get_cheese_pos(grid)
    agreements, total = 0, 0
    colors = []
    for i, pos in enumerate(legal_mouse_positions):
        neighbors = [n for n in maze.get_empty_neighbors(grid, *pos) if n != cheese_pos]
        if len(neighbors) == 0:
            colors.append('white')
            continue
        neighbor_indexes = [legal_mouse_positions.index(n) for n in neighbors]
        neighbor_values = [v[i] for i in neighbor_indexes]
        n = neighbors[neighbor_values.index(max(neighbor_values))]
        argmax_value_dir = (n[0]-pos[0], n[1]-pos[1])
        argmax_policy_dir = tuple(models.MAZE_ACTION_DELTAS_BY_INDEX[c.probs[i].argmax().item()])
        agreements += argmax_value_dir == argmax_policy_dir
        total += 1
        if argmax_value_dir != argmax_policy_dir:
            colors.append('red')
        else:
            colors.append('green')
    print(f'argmax behavior: {agreements}/{total} = {agreements/total:.2f}')


    # Quiver
    ax.quiver(
        [pos[1] for pos in legal_mouse_positions], [pos[0] for pos in legal_mouse_positions],
        [arr[1] for arr in vf["arrows"]], [arr[0] for arr in vf["arrows"]], color=colors, scale=1, scale_units='xy'
    )

    # ax[1].imshow(grid, origin='lower')
    plt.show()


# %%
# TODO: Figure out how to incorporate this into the above
# grads = t.autograd.grad(v[0], obs_all, retain_graph=True)
# len(grads), grads[0].shape

# %%
