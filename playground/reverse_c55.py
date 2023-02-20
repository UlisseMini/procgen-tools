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

def create_button(prefix : str, fig : plt.Figure, descriptors : defaultdict[str, float]): # TODO put in visualization.py?
    """ Create a button that saves fig to a file. 
    
    Args:
        prefix (str): The prefix of the filename.
        fig (plt.Figure): The figure to save.
        descriptors (defaultdict[str, float]): A dictionary of descriptors to add to the filename.
    """
    def save_fig(b):
        """ Save the figure to a file. """
        filename = f'{prefix}_'
        for key, value in descriptors.items():
            # Replace any dots with underscores
            value = str(value).replace('.', '_')
            filename += f'{key}_{value}_'
        filename = filename[:-1] + '.png' # remove trailing underscore
        fig.savefig(filename)
        # Display the filename
        display(Markdown(f'Figure saved to `{filename}`'))

    button = Button(description='Save figure')
    button.on_click(save_fig)
    return button

save_dir = 'playground/visualizations'
AX_SIZE = 6

# %% Doubling c55
@interact
def double_channel_55(seed=IntSlider(min=0, max=100, step=1, value=0), multiplier=FloatSlider(min=-15, max=15, step=0.1, value=5.5)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_multiply_patch(layer_name=default_layer, channel=55, multiplier=multiplier)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    button = create_button(prefix=f'{save_dir}/c55_multiplier', fig=fig, descriptors=defaultdict[str, float](seed=seed, multiplier=multiplier))
    display(button)

# %% [markdown]
# Let's see whether the c55 synthetic patch reproduces behavior in the unpatched model. 

# %%
# For each seed, compute the cheese location and then synthesize a patch that assigns positive activation to the corresponding location in the c55 channel
@interact 
def find_cheese(seed=IntSlider(min=0, max=100, step=1, value=0), value=FloatSlider(min=-15, max=15, step=0.1, value=.9)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)

    cheese_row, cheese_col = maze.get_cheese_pos_from_seed(seed)
    chan_row, chan_col = visualization.get_channel_from_grid_pos((cheese_row, cheese_col), layer=default_layer)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=55, coord=(chan_row, chan_col), value=value) 

    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    visualization.plot_pixel_dot(axs[1], row=chan_row, col=chan_col)

    fig.suptitle(f'Channel 55, position {chan_row}, {chan_col}')
    
    plt.show()

    button = create_button(prefix=f'{save_dir}/c55_pixel_synthetic', fig=fig, descriptors=defaultdict[str, float](seed=seed, value=value))
    display(button)

# %% Compare with patching a different channel with the same synthetic patch
@interact
def c55_patch_transfer_across_channels(seed=IntSlider(min=0, max=100, step=1, value=0), channel=IntSlider(min=0, max=63, step=1, value=54)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    cheese_pos = maze.get_cheese_pos_from_seed(seed)
    channel_pos = visualization.get_channel_from_grid_pos(cheese_pos, layer=default_layer)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, coord=channel_pos)

    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    fig.suptitle(f'Channel {channel}, position {channel_pos}')
    plt.show()

    button = create_button(prefix=f'{save_dir}/c55_synthetic_cheese_transfer', fig=fig, descriptors=defaultdict[str, float](seed=seed, channel=channel))
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
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    button = create_button(prefix=f'{save_dir}/random_channel_patch', fig=fig, descriptors=defaultdict[str, float](seed=seed, layer_name=layer_name, channel=channel))
    display(button)

# %% Causal scrub 55
# We want to replace the channel 55 activations with the activations from a randomly generated maze with cheese at the same location
GENERATE_NUM = 1 
@interact
def causal_scrub_55(target=IntSlider(min=0, max=100, step=1, value=0)):
    venv = patch_utils.get_cheese_venv_pair(seed=target)
    # Get the cheese location from the target seed
    cheese_row, cheese_col = maze.get_cheese_pos_from_seed(target)
    # Generate another seed with cheese at the same location, moving cheese to the appropriate locations 
    seeds, grids = maze.generate_mazes_with_cheese_at_location((cheese_row, cheese_col), num_mazes = GENERATE_NUM, skip_seed=source_seed)
    patches = patch_utils.get_causal_scrub_patch(layer_name=default_layer, channel=55, seed=target)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    button = create_button(prefix=f'{save_dir}/c55_causal_scrub', fig=fig, descriptors=defaultdict[str, float](source=source, target=target))
