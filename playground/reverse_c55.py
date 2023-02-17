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
from procgen_tools.procgen_imports import * # TODO doesn't let us autoreload

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
def get_channel_from_grid_pos(cheese_pos : Tuple[ int, int ], layer : str = default_layer):
    """ Given a grid position, find the channel location that corresponds to that position. """
    # Ensure cheese_pos is valid
    assert cheese_pos[0] >= 0 and cheese_pos[0] < maze.WORLD_DIM and cheese_pos[1] >= 0 and cheese_pos[1] < maze.WORLD_DIM, f'Invalid cheese position: {cheese_pos}'

    # Convert to pixel location
    cheese_pos = ((cheese_pos[0] + .5) * maze.PX_PER_TILE, (cheese_pos[1] + .5) * maze.PX_PER_TILE)

    px_per_channel_idx = get_stride(layer) # How many pixels per channel index
    channel_pos = (cheese_pos[0] // px_per_channel_idx, cheese_pos[1] // px_per_channel_idx)
    return (int(channel_pos[0]), int(channel_pos[1]))

@interact 
def find_cheese(seed=IntSlider(min=0, max=100, step=1, value=1)):
    venv = get_cheese_venv_pair(seed=seed)
    cheese_pos = get_cheese_pos_from_seed(seed)
    channel_pos = get_channel_from_grid_pos(cheese_pos, layer=default_layer)
    patches = get_channel_pixel_patch(layer_name=default_layer, channel=55, coord=channel_pos)
    # print(f'Cheese position: {cheese_pos}, channel position: {channel_pos} (channel 55)')

    fig, axs, info = compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    fig.suptitle(f'Channel position: {channel_pos} (channel 55)')
    plt.show()


    def save_fig(b):
        fig.savefig(f'visualizations/c55_synthetic_seed_{seed}.png')
    button = Button(description='Save figure')
    button.on_click(save_fig)
    display(button)
# %%
