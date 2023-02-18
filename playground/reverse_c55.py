# %% Don't have to restart kernel and reimport each time you modify a dependency
%reload_ext autoreload
%autoreload 2

# %%
try:
    import procgen_tools
except ImportError or ModuleNotFoundError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup() # create directory structure and download data 

# %% Super-long import code!
from procgen_tools.imports import *
from procgen_tools import patch_utils, visualization, vfield, maze

# %% Doubling c55
@interact
def double_channel_55(seed=IntSlider(min=0, max=100, step=1, value=0), multiplier=FloatSlider(min=-15, max=15, step=0.1, value=5.5)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_multiply_patch(layer_name=default_layer, channel=55, multiplier=multiplier)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    plt.show()

    def save_fig(b):
        fig.savefig(f'visualizations/c55_multiplier_{multiplier}_seed_{seed}.png')
    button = Button(description='Save figure')
    button.on_click(save_fig)
    display(button)

# %% [markdown]
# Let's see whether the c55 synthetic patch reproduces behavior in the unpatched model. 

# %%
# For each seed, compute the cheese location and then find an appropriate channel patch
@interact 
def find_cheese(seed=IntSlider(min=0, max=100, step=1, value=0)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    visualization.visualize_venv(venv, mode='numpy', flip_numpy=False)

    cheese_row, cheese_col = maze.get_cheese_pos_from_seed(seed)
    print(cheese_row, cheese_col)
    chan_row, chan_col = visualization.get_channel_from_grid_pos((cheese_row, cheese_col), layer=default_layer)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=55, coord=(chan_row, chan_col), value=5.6) # TODO set this back to zero

    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    # plot_pixel_dot(axs[1], row=chan_row, col=chan_col)
    visualization.plot_pixel_dot(axs[1], row=chan_col, col=chan_row)
    visualization.plot_pixel_dot(axs[1], row=cheese_row, col=cheese_col, color='yellow') # Cheese location
    fig.suptitle(f'Channel 55, position {chan_row}, {chan_col}')
    
    plt.show()

    def save_fig(b): # TODO make this a library call
        fig.savefig(f'playground/visualizations/c55_synthetic_seed_{seed}.png')
    button = Button(description='Save figure')
    button.on_click(save_fig)
    display(button)

# %% Compare with patching a different channel with the same synthetic patch
@interact
def c55_patch_transfer_channels(seed=IntSlider(min=0, max=100, step=1, value=0), channel=IntSlider(min=0, max=63, step=1, value=54)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    cheese_pos = maze.get_cheese_pos_from_seed(seed)
    channel_pos = visualization.get_channel_from_grid_pos(cheese_pos, layer=default_layer)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, coord=channel_pos)

    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    fig.suptitle(f'Channel {channel}, position {channel_pos}')
    plt.show()

    def save_fig(b):
        fig.savefig(f'playground/visualizations/c{channel}_synthetic_seed_{seed}.png')
    button = Button(description='Save figure')
    button.on_click(save_fig)
    display(button)

# %% Random patching channels
channel_slider = IntSlider(min=-1, max=63, step=1, value=55)
@interact
def random_channel_patch(seed=IntSlider(min=0, max=100, step=1, value=0), layer_name=Dropdown(options=labels, value=default_layer), channel=channel_slider):
    """ Replace the given channel's activations with values from a randomly sampled observation. This invokes patch_utils.get_random_patch from patch_utils. If channel=-1, then all channels are replaced. """
    channel_slider.max = patch_utils.num_channels(hook, layer_name) -1
    channel = channel_slider.value = min(channel_slider.value, channel_slider.max)

    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_random_patch(layer_name=layer_name, hook=hook, channel=channel)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    plt.show()

    def save_fig(b):
        fig.savefig(f'playground/visualizations/c55_random_seed_{seed}.png')
    button = Button(description='Save figure')
    button.on_click(save_fig)
    display(button)

# %%
