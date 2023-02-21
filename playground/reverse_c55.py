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
import procgen_tools.visualization as visualization
import procgen_tools.patch_utils as patch_utils
import procgen_tools.maze as maze

# %%
def plot_dots(axes : List[plt.Axes], coord : Tuple[int, int], color : str = 'red', flip_y : bool = True, is_grid : bool = False):
    """ Plot dots on the given axes, given a channel coordinate. If flip_y, flips the y-axis. If is_grid, assumes the coord is an outer grid coordinate. """
    row, col = visualization.get_channel_from_grid_pos(pos=coord) if is_grid else coord
    if flip_y: 
        row = 15 - row # NOTE assumes that channel width is 16
    for ax in axes:
        visualization.plot_pixel_dot(ax, row=row, col=col, color=color)
    
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

# %% Patch a single channel we've found to track cheese
cheese_channels = [77, 113, 44, 88, 55, 42, 7, 8, 82, 99] 
effective_channels = [77, 113, 88, 55, 8, 82, 89]

def get_combined_pixel_patch(layer_name : str, value : float, coord : Tuple[int, int], channels : List[int]):
    """ Get a patch that modifies multiple channels at once. """
    patches = [patch_utils.get_channel_pixel_patch(layer_name=layer_name, channel=channel, value=value, coord=coord) for channel in channels]
    combined_patch = patch_utils.compose_patches(*patches)
    return combined_patch


# %% 
@interact
def apply_all_cheese_patches(seed=IntSlider(min=0, max=20, step=1, value=0), value=FloatSlider(min=-10, max=10, step=0.1, value=2.8), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel_list=Dropdown(options=[effective_channels, cheese_channels], value=effective_channels)):
    combined_patch = get_combined_pixel_patch(layer_name=default_layer, value=value, coord=(row, col), channels=channel_list)

    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, combined_patch, hook, render_padding=True, ax_size=AX_SIZE)

    # Draw a red pixel at the location of the patch
    plot_dots(axs[1:], (row, col)) 
    plt.show()

    button = create_button(prefix=f'{save_dir}/{"all" if channel_list == cheese_channels else "effective"}_cheese_patches', fig=fig, descriptors=defaultdict[str, float](seed=seed, value=value, row=row, col=col))
    display(button)

# %% Try synthetically modifying each channel individually
@interact
def interactive_channel_patch(seed=IntSlider(min=0, max=20, step=1, value=0), value=FloatSlider(min=-30, max=30, step=0.1, value=5.6), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel=Dropdown(options=cheese_channels, value=55)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, value=value, coord=(row, col)) 
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)

    # Draw a red pixel at the location of the patch
    plot_dots(axs[1:], (row, col)) # TODO red dot not rendering
    plt.show() 

    # Add a button to save the figure to experiments/visualizations
    button = create_button(prefix=f'{save_dir}/c{channel}_pixel_patch', fig=fig, descriptors=defaultdict[str, float](seed=seed, value=value, row=row, col=col))
    display(button)

# %% Multiplying c55, treating both positive and negative activations separately
@interact
def multiply_channel_55(seed=IntSlider(min=0, max=100, step=1, value=0), pos_multiplier=FloatSlider(min=-15, max=15, step=0.1, value=5.5), neg_multiplier=FloatSlider(min=-15, max=15, step=0.1, value=5.5)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_multiply_patch(layer_name=default_layer, channel=55, pos_multiplier=pos_multiplier, neg_multiplier=neg_multiplier)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    button = create_button(prefix=f'{save_dir}/c55_multiplier', fig=fig, descriptors=defaultdict[str, float](seed=seed, pos=pos_multiplier, neg=neg_multiplier))
    display(button)

# %% [markdown]
# Let's see whether the c55 synthetic patch reproduces behavior in the unpatched model. 

# %%
# For each seed, compute the cheese location and then synthesize a patch that assigns positive activation to the corresponding location in the c55 channel
@interact 
def find_cheese(seed=IntSlider(min=0, max=100, step=1, value=20), value=FloatSlider(min=-15, max=15, step=0.1, value=.9), perturb=Dropdown(options=[False, True], value=False)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)

    cheese_row, cheese_col = maze.get_cheese_pos_from_seed(seed, flip_y=True) 

    chan_row, chan_col = visualization.get_channel_from_grid_pos((cheese_row, cheese_col), layer=default_layer)  
    
    if perturb: # Check that the location matters (debugging)
        chan_row += 2
        chan_col += 2

    # patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=55, coord=(chan_row, chan_col), value=value) 
    combined_patch = get_combined_pixel_patch(layer_name=default_layer, value=value, coord=(chan_row, chan_col), channels=cheese_channels)

    fig, axs, info = patch_utils.compare_patched_vfields(venv, combined_patch, hook, render_padding=True, ax_size=AX_SIZE)
    plot_dots(axs[1:], (chan_row, chan_col))

    fig.suptitle(f'Cheese patch channels, position {chan_row}, {chan_col}')
    # fig.suptitle(f'Channel 55, position {chan_row}, {chan_col}') # TODO clean up / generalize
    
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
def random_combined_px_patch(layer_name : str, channels : List[int], cheese_loc : Tuple[int, int] = None):
    """ Get a combined patch which randomly replaces channel activations with other activations from different levels. """
    patches = [patch_utils.get_random_patch(layer_name=layer_name, hook=hook, channel=channel, cheese_loc=cheese_loc) for channel in channels]
    combined_patch = patch_utils.compose_patches(*patches)
    return combined_patch

@interact
def causal_scrub_55(seed=IntSlider(min=0, max=100, step=1, value=17)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)

    # TODO statistically measure cheese pos activations, and average negative activations?
    
    cheese_row, cheese_col = maze.get_cheese_pos_from_seed(seed, flip_y=False)  # TODO flip_y should be false here, and also false for plot_dots -- will simplify logic
    patches = random_combined_px_patch(layer_name=default_layer, channels=cheese_channels, cheese_loc=(cheese_row, cheese_col))
    # patches = random_combined_px_patch(layer_name=default_layer, channels=[55])
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)

    plot_dots(axs[1:], (cheese_row, cheese_col), is_grid=True, flip_y=False)
    plt.show()

    button = create_button(prefix=f'{save_dir}/c55_causal_scrub', fig=fig, descriptors=defaultdict[str, float](seed=seed))
    display(button)

# %%
