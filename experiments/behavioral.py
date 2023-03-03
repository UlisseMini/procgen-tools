# %%
%reload_ext autoreload
%autoreload 2

# %%

try:
    import procgen_tools
except ImportError or ModuleNotFoundError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.imports import *
from procgen_tools import visualization, patch_utils, maze, vfield

# %% Generate vfields for 8 random seeds, in a 4x2 grid
AX_SIZE = 4
rows = 4
cols = 2

plt.close('all')
fig, axs = plt.subplots(rows, cols, figsize=(AX_SIZE*cols, AX_SIZE*rows))

# Generate the mazes and plot the vfields
for idx, ax in enumerate(axs.flatten()): 
    venv = maze.create_venv(num=1, start_level=0, num_levels=0)
    vf = vfield.vector_field(venv, policy=hook.network)
    vfield.plot_vf(vf, ax=ax, show_components=False)
    ax.axis('off')

plt.show()
# %%
