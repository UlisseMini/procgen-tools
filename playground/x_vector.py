# %% Let's find more vectors besides just the cheese vector
%reload_ext autoreload
%autoreload 2

# %%
try:
    import procgen_tools
except ImportError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup() # create directory structure and download data 

# %%
from procgen_tools.imports import *
from procgen_tools import visualization, patch_utils, maze, vfield
AX_SIZE = 4 

load_venv = False
pwd = 'playground'
venv_folder = 'venv'
venv_fname = f'{pwd}/{venv_folder}/top_right_venv.pkl'

# %% Try generating a top-right vector; prediction of .3 that my first idea works
if load_venv:
    venv = maze.load_venv(venv_fname)
else:
    venv = maze.create_venv(num=2, start_level=0, num_levels=1)
    maze_editors = maze.venv_editor(venv, show_full=True, check_on_dist=False)
    display(maze_editors)
# %%
if not load_venv: maze.save_venv(venv=venv, filename=venv_fname) # TODO check these functions work
# %% Get a value patch from this pair of envs
# Compare vector fields for this patch
@interact
def examine_patch(target_seed=IntSlider(min=0,max=100,step=1,value=0), coeff=FloatSlider(min=-5,max=5,step=.1,value=1)): 
    top_right_patch = patch_utils.patch_from_venv_pair(venv, layer_name=default_layer, hook=hook, coeff=coeff)
    target_venv = maze.create_venv(num=1, start_level=target_seed, num_levels=1)
    fig, axs, info = patch_utils.compare_patched_vfields(target_venv, top_right_patch, hook, render_padding=False, ax_size=AX_SIZE)
    plt.show(fig)
# %%
