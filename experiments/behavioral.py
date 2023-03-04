# %%
%reload_ext autoreload
%autoreload 2

# %%

try:
    import procgen_tools
except ImportError or ModuleNotFoundError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup(download_data=False) # create directory structure and download data 

from procgen_tools.imports import *
from procgen_tools import visualization, patch_utils, maze, vfield

# %% Generate vfields for randomly generated seeds
AX_SIZE = 4
rows = 1
cols = 2

# Control show_components with a checkbox
checkbox = widgets.Checkbox(value=False, description='Show action components')

def generate_plots(max_size : int = 3):
    """ Generate plots for random seeds with inner grid size at most max_size. """
    # Ensure no double-shown plots
    assert max_size >= 3, f"The smallest mazes have size 3; {max_size} is too small!"

    plt.close('all')
    fig, axs = plt.subplots(rows, cols, figsize=(AX_SIZE*cols, AX_SIZE*rows))
    for idx, ax in enumerate(axs.flatten()): 
        seed = np.random.randint(0, 100000)
        while maze.get_inner_grid_from_seed(seed=seed).shape[0] > max_size:
            print(f"Seed {seed} has inner grid size {maze.get_inner_grid_from_seed(seed=seed).shape[0]}; skipping...")
            seed = np.random.randint(0, 100000)
            print(seed)
        print(maze.get_inner_grid_from_seed(seed=seed).shape[0])
        print(seed)
        venv = maze.create_venv(num=1, start_level=seed, num_levels=0)
        vf = vfield.vector_field(venv, policy=hook.network)
        vfield.plot_vf(vf, ax=ax, show_components=checkbox.value)
        ax.set_title(f'Seed: {seed:,}')
        ax.axis('off')
    plt.show(fig)

# Make a button to generate new plots
button = widgets.Button(description='Generate new plots')
button.on_click(lambda _: generate_plots())
display(HBox([button, checkbox]))

generate_plots()
# %%
