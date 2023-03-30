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

# %% Try generating a top-right vector; prediction of .3 that my first
# idea works (EDIT: It did!)
venv = maze.get_top_right_venv_pair(seed=1)
# venv = maze.create_venv(num=2, start_level=1, num_levels=1)
# venv = maze.load_venv(f'{pwd}/{mazes_folder}/translate.pkl')
maze_editors = maze.venv_editor(venv, show_full=True, check_on_dist=False)
output = Output()
# with output: TODO get this working
#    display(maze_editors)
display(maze_editors)

# %% Interactively save and load mazes
# Make a load button 
load_button = widgets.Button(description='Load')
# Get a dropdown of basenames from the mazes folder
basenames_from_folder = os.listdir(mazes_folder)
basenames_from_folder = [os.path.splitext(basename)[0] for basename in basenames_from_folder]
load_dropdown = widgets.Dropdown(
    options=basenames_from_folder,
    value='top right',
    description='Basename:',
    disabled=False,
)

def load_cb():
    venv = maze.load_venv(f'{pwd}/{mazes_folder}/{basenames[load_dropdown.value]}.pkl')
    maze_editors = maze.venv_editor(venv, show_full=True, check_on_dist=False) # TODO update with callable?
    output.clear_output()
    with output:
        display(maze_editors)

load_button.on_click(lambda x: load_cb())
display(HBox([load_dropdown, load_button]))

# Make a save button         
def save_cb():
    maze.save_venv(venv=venv, filename=f'{pwd}/{mazes_folder}/{basenames[new_basename.value]}.pkl')
new_basename = widgets.Text(
    value='',
    placeholder='Type something',
    description='New basename:',
    disabled=False
)
save_button = widgets.Button(description='Save')
save_button.on_click(lambda x: save_cb())
display(save_button)

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

# %% Try modifying channels which encode mouse pos
mouse_channels = [68, 69, 76, 83, 93, 96, 97, 111, 121, 126] # NOTE only half of mouse channels # 20% that this works reasonably well
@interact
def modify_mouse_channels(seed=IntSlider(min=0,max=100,step=1,value=0), use_default = Checkbox(value=True), value=FloatSlider(min=-5,max=5,step=.1,value=1)):
    mouse_pos = maze.get_mouse_pos_from_seed(seed)
    grid_pos = visualization.get_channel_from_grid_pos(mouse_pos, layer=default_layer)
    patch = patch_utils.combined_pixel_patch(layer_name=default_layer, value=value, coord=grid_pos, channels=mouse_channels, default=use_default)
    
    target_venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    fig, axs, info = patch_utils.compare_patched_vfields(target_venv, patch, hook, render_padding=False, ax_size=AX_SIZE)
    plt.show(fig)

# %%
