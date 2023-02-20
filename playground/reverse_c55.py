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

# %% 
@interact
def apply_all_cheese_patches(seed=IntSlider(min=0, max=20, step=1, value=0), value=FloatSlider(min=-30, max=30, step=0.1, value=2.8), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel_list=Dropdown(options=[effective_channels, cheese_channels], value=effective_channels)):
    patches = [patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, value=value, coord=(row, col)) for channel in effective_channels]
    combined_patch = patch_utils.compose_patches(*patches)

    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, combined_patch, hook, render_padding=True, ax_size=AX_SIZE)

    # Draw a red pixel at the location of the patch
    for idx in (1,2):
        visualization.plot_pixel_dot(axs[idx], 15 - row, col) 
    plt.show()

    button = create_button(prefix=f'{save_dir}/all_cheese_patches', fig=fig, descriptors=defaultdict[str, float](seed=seed, value=value))
    display(button)

# %% 
@interact
def interactive_channel_patch(seed=IntSlider(min=0, max=20, step=1, value=0), value=FloatSlider(min=-30, max=30, step=0.1, value=5.6), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel=Dropdown(options=cheese_channels, value=55)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, value=value, coord=(row, col)) 
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)

    # Draw a red pixel at the location of the patch
    for idx in (1,2):
        visualization.plot_pixel_dot(axs[idx], 15 - row, col) 
    plt.show() 

    # Add a button to save the figure to experiments/visualizations
    button = create_button(prefix=f'{save_dir}/c{channel}_pixel_patch', fig=fig, descriptors=defaultdict[str, float](seed=seed, value=value, row=row, col=col))
    display(button)



# %% Multiplying c55, treating both positive and negative activations separately
@interact
def double_channel_55(seed=IntSlider(min=0, max=100, step=1, value=0), multiplier=FloatSlider(min=-15, max=15, step=0.1, value=5.5)):
    venv = get_cheese_venv_pair(seed=seed)
    patches = get_multiply_patch(layer_name=default_layer, channel=55, multiplier=multiplier)
    fig, axs, info = compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=6)
    plt.show()
    # print(info['patched_vfield']['probs'])

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
cheese_channels = [77, 113, 44, 88, 55, 42, 7] # 44 does nothing? 42 is a bit weird, it doesnt do anything at -30 (might get relu'd) but screws things up at eg +20
# 77 113 88 and 55 seem to work
effective_channels = [77, 113, 88, 55]

# %% 
@interact
def apply_all_cheese_patches(seed=IntSlider(min=0, max=20, step=1, value=0), value=FloatSlider(min=-30, max=30, step=0.1, value=2.5), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5)):
    patches = [patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, value=value, coord=(row, col)) for channel in effective_channels]
    combined_patch = patch_utils.compose_patches(*patches)

    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, combined_patch, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    # Draw a red pixel at the location of the patch
    for idx in (1,2):
        visualization.plot_pixel_dot(axs[idx], 15 - row, col) 

    button = create_button(prefix=f'{save_dir}/all_cheese_patches', fig=fig, descriptors=defaultdict[str, float](seed=seed, value=value))
    display(button)

# %% 
@interact
def interactive_channel_patch(seed=IntSlider(min=0, max=20, step=1, value=0), value=FloatSlider(min=-30, max=30, step=0.1, value=5.6), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel=Dropdown(options=effective_channels, value=55)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, value=value, coord=(row, col)) 
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)

    # Draw a red pixel at the location of the patch
    for idx in (1,2):
        visualization.plot_pixel_dot(axs[idx], 15 - row, col) 
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
@interact
def causal_scrub_55(seed=IntSlider(min=0, max=100, step=1, value=0)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    
    cheese_row, cheese_col = maze.get_cheese_pos_from_seed(seed)
    patches = patch_utils.get_random_patch(layer_name=default_layer, hook=hook, channel=55, cheese_loc=(cheese_row-3, cheese_col+5))
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    button = create_button(prefix=f'{save_dir}/c55_causal_scrub', fig=fig, descriptors=defaultdict[str, float](seed=seed))
    display(button)

# %%
