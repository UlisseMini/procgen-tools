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

pwd = 'playground'
mazes_folder = 'mazes'
basenames = {'top right': 'top_right_path', 'cheese top right': 'cheese_top_right', 'translate': 'translate'}
venv_fname = f'{pwd}/{mazes_folder}/{basenames["translate"]}.pkl'

# %% Try generating a top-right vector; prediction of .3 that my first idea works (EDIT: It did!)
@interact
def generate_x_vector(basename=Dropdown(options=basenames.keys(), value='top right')):
    # Make a field for making new basenames
    new_basename = widgets.Text(
        value='',
        placeholder='Type something',
        description='New basename:',
        disabled=False
    )
    
    # Make a load button 
    load_button = widgets.Button(description='Load')

    def load_cb():
        venv = maze.load_venv(f'{pwd}/{mazes_folder}/{basenames[new_basename.value]}.pkl')
        maze_editors = maze.venv_editor(venv, show_full=True, check_on_dist=False)
        display(maze_editors)
    load_button.on_click(lambda x: load_cb())
    display(HBox([new_basename, load_button]))
    
    venv = maze.create_venv(num=2, start_level=1, num_levels=1)
    maze_editors = maze.venv_editor(venv, show_full=True, check_on_dist=False)
    display(maze_editors)

    def save_cb():
        maze.save_venv(venv=venv, filename=f'{pwd}/{mazes_folder}/{basenames[basename]}.pkl')
    
    # Make a save button
    save_button = widgets.Button(description='Save')
    save_button.on_click(lambda x: save_cb())
    display(save_button)
load_venv = False
if load_venv:
    venv = maze.load_venv(venv_fname)
else:
    venv = maze.create_venv(num=2, start_level=1, num_levels=1)
    maze_editors = maze.venv_editor(venv, show_full=True, check_on_dist=False)
    display(maze_editors)
# %%
# Prompt user before saving TODO also get top_right_path.pkl
# TODO fix rename of function in gatherdata/metrics?
if not load_venv: maze.save_venv(venv=venv, filename=venv_fname) 
# %% Get a value patch from this pair of envs
# Compare vector fields for this patch
@interact
def examine_patch(target_seed=IntSlider(min=0,max=100,step=1,value=0), coeff=FloatSlider(min=-5,max=5,step=.1,value=1)): 
    patch = patch_utils.patch_from_venv_pair(venv, layer_name=default_layer, hook=hook, coeff=coeff)
    target_venv = maze.create_venv(num=1, start_level=target_seed, num_levels=1)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patch, hook, render_padding=False, ax_size=AX_SIZE, show_components=True)
    plt.show(fig)

# %% Test patch composition with cheese vector patch (credence: 75% this works) EDIT: It did
@interact
def compose_patch(target_seed=IntSlider(min=0,max=100,step=1,value=25), top_right_coeff=FloatSlider(min=-5,max=5,step=.1,value=1), use_cheese_vector=Checkbox(value=True), use_tr_vector=Checkbox(value=True)):
    patch_list = []
    if use_tr_vector: 
        tr_patch = patch_utils.patch_from_venv_pair(venv, layer_name=default_layer, hook=hook, coeff=top_right_coeff)
        patch_list.append(tr_patch)

    if use_cheese_vector:
        cheese_diff_values = patch_utils.cheese_diff_values(seed=target_seed, layer_name=default_layer, hook=hook)
        cheese_patch = patch_utils.get_values_diff_patch(values=cheese_diff_values, coeff=-1, layer_name=default_layer)
        patch_list.append(cheese_patch)
    
    patch = patch_utils.compose_patches(*patch_list) if patch_list else {}

    target_venv = maze.create_venv(num=1, start_level=target_seed, num_levels=1)
    fig, axs, info = patch_utils.compare_patched_vfields(target_venv, patch, hook, render_padding=False, ax_size=AX_SIZE)

    # Title which patches we're using 
    title = 'Patches: '
    if use_tr_vector: title += 'Top-right vector, '
    if use_cheese_vector: title += 'Cheese vector, '
    title = title[:-2] # Remove trailing comma
    fig.suptitle(title)

    plt.show(fig)
# %% Show vfield for full maze 
single_venv = maze.copy_venv(venv, idx=0) # Modified maze is at index 0
vf_box = visualization.custom_vfield(policy=hook.network, venv=single_venv, show_full=True, ax_size=AX_SIZE) # NOTE broken for non-traditional padding venvs if you don't pass in an ax_size 
display(vf_box)
# %% Run behavioral stats on two environments
stats_venv = maze.create_venv(num=2, start_level=0, num_levels=1)
vf_boxes = visualization.custom_vfields(venv=stats_venv, policy=hook.network, show_full=True, ax_size=AX_SIZE)
display(vf_boxes)
# %%
