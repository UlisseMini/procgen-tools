# %% [markdown]
# One particularly interesting channel in `block2.res1.resadd_out` is _channel 55_. In this notebook, we will: 
# 1. Visualize channel 55 and demonstrate **that it positively activates on cheese and weakly negatively activates elsewhere**, 
# 2. Demonstrate how the agent can sometimes be retargeted using a simple synthetic activation patch, and
# 3. Show that this channel can weakly increase cheese-seeking (multiply by >1), decrease cheese-seeking (zero- or mean-ablate), strongly promote cheese-avoidance (multiply by < -1), and promote no-ops (multiply by << -1). 
# 4. Demonstrate that in `block2.res1.resadd_out` `cheese_channels=[7,8,42,44,55,77,82,88,99,113]` appear to encode cheese position in a similar manner. With the exception of channels `7, 44, 99`, the "cheese channels" mimic channel 55 in that they also individually ret arget the agent's behavior

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
from procgen_tools import visualization, patch_utils, maze


# %%
cheese_channels = [77, 113, 44, 88, 55, 42, 7, 8, 82, 99] 
effective_channels = [77, 113, 88, 55, 8, 82, 89]

SAVE_DIR = 'experiments'
AX_SIZE = 6

# %% [markdown]
# # Visualizing channel 55

# %% [markdown]
# Try clicking on the left-ward level editor below. Move the cheese around the maze by clicking on the yellow tile, and then clicking on the tile you want to contain the cheese. Watch the positive blue activations equivariantly translate along with the cheese!

# %%
# Show a maze editor side-by-side with the interactive plotter
SEED = 0
venv = maze.create_venv(num=1, start_level=SEED, num_levels=1) # This has to be a single maze, otherwise the vfield wont work

default_settings = {'channel_slider': 55, 'label_widget': 'block2.res1.resadd_out'}
custom_maze_plotter = visualization.ActivationsPlotter(labels, lambda activations, fig: visualization.plot_activations(activations[0], fig=fig), patch_utils.values_from_venv, hook, defaults=default_settings, venv=venv)

widget_box = visualization.custom_vfield(policy, venv=venv, callback=custom_maze_plotter.update_plotter, ax_size = 2) 
display(widget_box)
    
custom_maze_plotter.display() 

# %% [markdown]
# # Intervening on 55
# It turns out that channel 55 lets us retarget the agent somewhat reliably and strongly, moving around only a single activation in a single convolutional layer.

# %%
# %% Try synthetically modifying each channel individually
@interact
def interactive_channel_patch(seed=IntSlider(min=0, max=20, step=1, value=0), value=FloatSlider(min=-30, max=30, step=0.1, value=5.6), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel=Dropdown(options=cheese_channels, value=55)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, value=value, coord=(row, col)) 
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    fig.suptitle(f'Synthetically patching {channel} (value={value})')

    # Draw a red pixel at the location of the patch
    visualization.plot_dots(axs[1:], (row, col), color='red')
    plt.show() 

    # Add a button to save the figure to experiments/visualizations
    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/c{channel}_pixel_patch', fig=fig, descriptors=defaultdict[str, float](seed=seed, value=value, row=row, col=col))
    display(button)
    
    # Render the synthetic patch
    synth_activations = patches[default_layer](hook.values_by_label[default_layer])
    act_fig = go.Figure()
    visualization.plot_activations(synth_activations[0,55], fig=act_fig)
    visualization.format_plotter(fig=act_fig, activations=synth_activations, bounds=(-.8, .8)) 
    display(act_fig)

# %%
@interact
def double_channel_55(seed=IntSlider(min=0, max=100, step=1, value=0), multiplier=FloatSlider(min=0, max=2, step=0.1, value=2)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)

    patches = patch_utils.get_multiply_patch(layer_name=default_layer, channel=55, pos_multiplier=multiplier, neg_multiplier=multiplier)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/visualizations/double_c55/', fig=fig, descriptors={'seed': seed, 'multiplier': multiplier})

# %%



