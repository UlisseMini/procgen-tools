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

from procgen_tools.patch_utils import * # TODO stop using import * 
from procgen_tools.visualization import *
from procgen_tools import vfield
from procgen_tools.maze import *

# %% Doubling c55
@interact
def double_channel_55(seed=IntSlider(min=0, max=100, step=1, value=0), multiplier=FloatSlider(min=-15, max=15, step=0.1, value=5.5)):
    venv = get_cheese_venv_pair(seed=seed)
    patches = get_multiply_patch(layer_name=default_layer, channel=55, multiplier=multiplier)
    fig, axs, info = compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    plt.show()
    # print(info['patched_vfield']['probs'])

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
def find_cheese(seed=IntSlider(min=0, max=100, step=1, value=1)):
    venv = get_cheese_venv_pair(seed=seed)
    cheese_pos = get_cheese_pos_from_seed(seed)
    channel_pos = get_channel_from_grid_pos(cheese_pos, layer=default_layer)
    patches = get_channel_pixel_patch(layer_name=default_layer, channel=55, coord=channel_pos)

    fig, axs, info = compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    fig.suptitle(f'Channel 55, position {channel_pos}')
    plt.show()

    def save_fig(b): # TODO make this a library call
        fig.savefig(f'playground/visualizations/c55_synthetic_seed_{seed}.png')
    button = Button(description='Save figure')
    button.on_click(save_fig)
    display(button)

# %% Compare with patching a different channel with the same synthetic patch
@interact
def c55_patch_transfer_channels(seed=IntSlider(min=0, max=100, step=1, value=0), channel=IntSlider(min=0, max=63, step=1, value=54)):
    venv = get_cheese_venv_pair(seed=seed)
    cheese_pos = get_cheese_pos_from_seed(seed)
    channel_pos = get_channel_from_grid_pos(cheese_pos, layer=default_layer)
    patches = get_channel_pixel_patch(layer_name=default_layer, channel=channel, coord=channel_pos)
    patches = get_random_patch(layer_name=default_layer, hook=hook, channel=channel) # TODO why doesn't this work?

    fig, axs, info = compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    fig.suptitle(f'Channel {channel}, position {channel_pos}')
    plt.show()

    def save_fig(b):
        fig.savefig(f'playground/visualizations/c{channel}_synthetic_seed_{seed}.png')
    button = Button(description='Save figure')
    button.on_click(save_fig)
    display(button)


# %%

# %% Random patching channels
channel_slider = IntSlider(min=0, max=63, step=1, value=55)
@interact
def random_channel_patch(seed=IntSlider(min=0, max=100, step=1, value=0), layer_name=Dropdown(options=labels, value=default_layer), channel=channel_slider):
    """ Replace channel 55's activations with values from a randomly sampled observation. This invokes get_random_patch from patch_utils. """
    channel_slider.max = num_channels(hook, layer_name) -1
    channel = channel_slider.value = min(channel_slider.value, channel_slider.max)

    venv = get_cheese_venv_pair(seed=seed)
    patches = get_random_patch(layer_name=layer_name, hook=hook, channel=channel)
    fig, axs, info = compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    plt.show()

    def save_fig(b):
        fig.savefig(f'playground/visualizations/c55_random_seed_{seed}.png')
    button = Button(description='Save figure')
    button.on_click(save_fig)
    display(button)

# %%
