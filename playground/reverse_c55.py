# %% Don't have to restart kernel and reimport each time you modify a dependency
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

SAVE_DIR = 'playground/visualizations'
AX_SIZE = 6

cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
effective_channels = [8, 55, 77, 82, 88, 89, 113]

# %% 
@interact
def apply_all_cheese_patches(seed=IntSlider(min=0, max=100, step=1, value=0), value=FloatSlider(min=-10, max=10, step=0.1, value=1.1), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel_list=Dropdown(options=[effective_channels, cheese_channels], value=effective_channels)):
    combined_patch = patch_utils.combined_pixel_patch(layer_name=default_layer, value=value, coord=(row, col), channels=channel_list, default=None)

    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, combined_patch, hook, render_padding=True, ax_size=AX_SIZE)

    # Draw a red pixel at the location of the patch
    visualization.plot_dots(axs[1:], (row, col)) 
    plt.show()

    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/{"all" if channel_list == cheese_channels else "effective"}_cheese_patches_', fig=fig, descriptors=defaultdict(seed=seed, value=value, row=row, col=col))
    display(button)

# %% Try synthetically modifying each channel individually
@interact
def interactive_channel_patch(seed=IntSlider(min=0, max=100, step=1, value=0), value=FloatSlider(min=-30, max=30, step=0.1, value=5.6), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel=Dropdown(options=cheese_channels, value=42)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, value=value, coord=(row, col), default=None) 
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=False, ax_size=AX_SIZE)
    fig.suptitle(f'Synthetically patching {channel} (value={value})', fontsize=20)

    # Draw a red pixel at the location of the patch
    visualization.plot_dots(axs[1:], (row, col), color='red')
    plt.show() 

    # Add a button to save the figure to experiments/visualizations
    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/c{channel}_pixel_patch', fig=fig, descriptors=defaultdict(seed=seed, value=value, row=row, col=col))
    display(button)

# %% Multiplying c55, treating both positive and negative activations separately
@interact
def multiply_channel_55(seed=IntSlider(min=0, max=100, step=1, value=0), pos_multiplier=FloatSlider(min=-15, max=15, step=0.1, value=5.5), neg_multiplier=FloatSlider(min=-15, max=15, step=0.1, value=5.5)):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_multiply_patch(layer_name=default_layer, channel=55, pos_multiplier=pos_multiplier, neg_multiplier=neg_multiplier)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/c55_multiplier', fig=fig, descriptors=defaultdict(seed=seed, pos=pos_multiplier, neg=neg_multiplier))
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
    combined_patch = patch_utils.combined_pixel_patch(layer_name=default_layer, value=value, coord=(chan_row, chan_col), channels=cheese_channels, default=-.2)

    fig, axs, info = patch_utils.compare_patched_vfields(venv, combined_patch, hook, render_padding=True, ax_size=AX_SIZE)
    visualization.plot_dots(axs[1:], (chan_row, chan_col))

    fig.suptitle(f'Cheese patch channels, position {chan_row}, {chan_col}')
    # fig.suptitle(f'Channel 55, position {chan_row}, {chan_col}') # TODO clean up / generalize
    
    plt.show()

    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/c55_pixel_synthetic', fig=fig, descriptors=defaultdict(seed=seed, value=value))
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

    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/c55_synthetic_cheese_transfer', fig=fig, descriptors=defaultdict(seed=seed, channel=channel))
    display(button)

# %% Random patching channels
seed_slider = IntSlider(min=0, max=100, step=1, value=0)
layer_slider = Dropdown(options=labels, value=default_layer)
channel_slider = IntSlider(min=-1, max=127, step=1, value=55)

from procgen_tools import models
def random_channel_patch(seed : int, layer_name : str, channel : int):
    """ Replace the given channel's activations with values from a randomly sampled observation. This invokes patch_utils.get_random_patch from patch_utils. If channel=-1, then all channels are replaced. """
    channel_slider.max = models.num_channels(hook, layer_name) -1
    channel = channel_slider.value = min(channel_slider.value, channel_slider.max)

    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_random_patch(layer_name=layer_name, hook=hook, channel=channel) 
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()

    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/random_channel_patch', fig=fig, descriptors=defaultdict(seed=seed, layer_name=layer_name, channel=channel))
    display(button)

interact(random_channel_patch, seed=seed_slider, layer_name=layer_slider, channel=channel_slider)

# %% Causal scrub 55
# We want to replace the channel 55 activations with the activations from a randomly generated maze with cheese at the same location
def random_combined_px_patch(layer_name : str, channels : List[int], cheese_loc : Tuple[int, int] = None):
    """ Get a combined patch which randomly replaces channel activations with other activations from different levels. """
    patches = [patch_utils.get_random_patch(layer_name=layer_name, hook=hook, channel=channel, cheese_loc=cheese_loc) for channel in channels]
    combined_patch = patch_utils.compose_patches(*patches)
    return combined_patch # NOTE we're resampling from a fixed maze for all target forward passes

def resample_activations(seed : int, channels : List[int], different_location : bool = False):
    """ Resample activations for default_layer with the given channels. 
    
    Args:
        seed (int): The seed for the maze
        channels (List[int]): The channels to resample
        different_location (bool, optional): If True, then the resampling location is randomly sampled. Otherwise, it is the cheese location. Defaults to False.
    """
    render_padding = False
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    resampling_loc = (14, 14) if different_location else maze.get_cheese_pos_from_seed(seed, flip_y=False)  # NOTE assumes cheese loc isn't near (14, 14)
    patches = random_combined_px_patch(layer_name=default_layer, channels=channels, cheese_loc=resampling_loc)

    # patches = random_combined_px_patch(layer_name=default_layer, channels=[55])
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=render_padding, ax_size=AX_SIZE)
    channel_description = f'channels {channels}' if len(channels) > 1 else f'channel {channels[0]}'
    fig.suptitle(f'Resampling {channel_description} on seed {seed}', fontsize=20)

    visualization.plot_dots(axs[1:], resampling_loc, is_grid=True, flip_y=False, hidden_padding = 0 if render_padding else padding)
    plt.show()

    # Display the average vector field difference magnitude
    avg_prob_diff = vfield.vf_diff_magnitude(info['diff_vfield']) / 2
    print(f'Action probability distributions changed by {avg_prob_diff * 100 :2.1f}% on average')

    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/causal_scrub', fig=fig, descriptors=defaultdict(seed=seed, different_location=different_location))
    display(button)

# %% Resample activations interactively
def get_alternate_channels(avoid_channels : List[int]) -> List[int]:
    """ Get a list of random channels which aren't in avoid_channels. """
    candidate_channels = [channel for channel in range(128) if channel not in avoid_channels]
    return sorted(np.random.choice(candidate_channels, size=len(avoid_channels), replace=False).tolist())

interactive(resample_activations, seed=IntSlider(min=0, max=100, step=1, value=60), channels=Dropdown(options=[cheese_channels, [42, 55, 77, 88], get_alternate_channels(cheese_channels), [55]], value=cheese_channels), different_location=Checkbox(value=False))
# %% See how resampling works for a range of seeds
for seed in range(0, 30):
    resample_activations(seed=seed, channels=cheese_channels)
    plt.close('all')
# %% Choose three random seeds and compare resampling properly with resampling a random location
seeds = np.random.choice(range(100), size=3, replace=False)
for seed in seeds:
    seed = int(seed)
    resample_activations(seed=seed, channels=cheese_channels)
    resample_activations(seed=seed, channels=cheese_channels, different_location=True)
    plt.close('all')

# %% Quantitively measure effects of random resampling
def avg_resampling(channels_lsts : List[List[int]], num_seeds : int = 10):
    """ For a list of channel lists, generate num_seeds random seeds and compute the effect of each channel random resampling on these seeds using vfield.vf_diff_magnitude_from_seed. This takes a while! """
    seeds = np.random.choice(range(10000), size=num_seeds, replace=False).tolist()
    avg_diffs = defaultdict(int)

    for seed in seeds:
        for change_loc in [False, True]:
            for channel_lst in channels_lsts:
                patches = random_combined_px_patch(layer_name=default_layer, channels=channel_lst, cheese_loc=(14, 14) if change_loc else None) # NOTE assumes cheese loc isn't near (14, 14)
                avg_diffs[tuple(channel_lst), change_loc] += vfield.vf_diff_magnitude_from_seed(seed, patches) / num_seeds
    return avg_diffs

# %% Compare stats on action probability shifts
channels_lsts = [cheese_channels, get_alternate_channels(cheese_channels), effective_channels, get_alternate_channels(effective_channels), [55], get_alternate_channels(avoid_channels=[55])]
avg_diffs = avg_resampling(channels_lsts, num_seeds=30)

# %% Show the avg_diffs data in plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

fig = make_subplots(rows=1, cols=2, subplot_titles=['Same Location', 'Different Location']) # TODO is this switched?
for change_loc in [False, True]:
    fig.add_trace(go.Bar(x=[str(channels) for channels in channels_lsts], y=[avg_diffs[tuple(channels), change_loc] for channels in channels_lsts]), row=1, col=2 if change_loc else 1)
fig.update_layout(title_text='Average Probability Shifts for Random Resampling')
# Set same y axes to be max of both
fig.update_yaxes(range=[0, max([max([avg_diffs[tuple(channels), change_loc] for channels in channels_lsts]) for change_loc in [False, True]])])

# Set legend
fig.update_layout(showlegend=False)

fig.show()
# %%
