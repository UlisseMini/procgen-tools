# %%
%reload_ext autoreload
%autoreload 2

# %%

try:
    import procgen_tools
except ImportError or ModuleNotFoundError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup() # create directory structure and download data 

from procgen_tools.imports import *
from procgen_tools import visualization, patch_utils, maze, vfield

# %% Generate vfields for randomly generated seeds
AX_SIZE = 4
rows = 1
cols = 2

# Control show_components with a checkbox
checkbox = widgets.Checkbox(value=False, description='Show action components')

def generate_plots():
    """ Generate plots for random seeds. """
    # Ensure no double-shown plots
    plt.close('all')

    fig, axs = plt.subplots(rows, cols, figsize=(AX_SIZE*cols, AX_SIZE*rows))
    for idx, ax in enumerate(axs.flatten()): 
        seed = np.random.randint(0, 100000)
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
