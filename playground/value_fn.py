# %%

import matplotlib.pyplot as plt
from procgen_tools import maze, models, vfield
import circrl.module_hook as cmh
import torch as t
from ipywidgets import interact
from ipywidgets.widgets import IntSlider, Dropdown
from glob import glob
import procgen_tools.stats as stats

# %%



def plot_value_vfield(venv, policy):
    venv_all, (legal_mouse_positions, grid) = maze.venv_with_all_mouse_positions(venv)
    # obs_all = t.tensor(venv_all.reset(), dtype=t.float32)
    obs_all = t.tensor(stats.timeit('venv_all.reset()')(lambda: venv_all.reset())(), dtype=t.float32)

    vf = vfield.vector_field(venv, policy)
    # _, v = policy(obs_all)
    stats.counts['obs_generated'] += len(obs_all)
    _, v = stats.timeit('policy(obs_all)')(lambda: policy(obs_all))()

    g = grid.copy()
    g[g==maze.BLOCKED] = 0 # 50 screws up plot
    g[g==maze.CHEESE] = 0 # temp for normalizing
    for i, pos in enumerate(legal_mouse_positions):
        # g[pos] = v[i].item()
        g[pos] = stats.timeit('v[i].item()')(lambda: v[i].item())()
    
    # g = (g-g.mean())/g.std()
    g = (g-g.min())/(g.max()-g.min())
    g[g==maze.CHEESE] = 1
    fig, ax = plt.subplots(1,1)
    fig.colorbar(ax.imshow(g, origin='lower', cmap='RdBu_r'), ax=ax)

    ax.quiver(
        [pos[1] for pos in legal_mouse_positions], [pos[0] for pos in legal_mouse_positions],
        [arr[1] for arr in vf["arrows"]], [arr[0] for arr in vf["arrows"]], color='white', scale=1, scale_units='xy'
    )


def plot_model_by_seed(seed: int, model_file: str):
    policy = models.load_policy(model_file, 15, 'cpu')
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    
    return plot_value_vfield(venv, policy)


if __name__ == '__main__':
    model_files = glob('../trained_models/**.pth')
    default_model = next(f for f in model_files if 'rand_region_5' in f)

    @interact
    def interactive(
        seed = IntSlider(min=0, max=30, value=0),
        model_file1 = Dropdown(options=model_files, value=default_model),
        model_file2 = Dropdown(options=model_files, value=default_model),
    ):
        plot_model_by_seed(seed, model_file1)
        plot_model_by_seed(seed, model_file2)


# %%
# TODO: Figure out how to incorporate this into the above
# grads = t.autograd.grad(v[0], obs_all, retain_graph=True)
# len(grads), grads[0].shape

# %%
