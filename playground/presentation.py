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


# %% Setup code for the rest of the notebook
cheese_channels = [77, 113, 44, 88, 55, 42, 7, 8, 82, 99] 
effective_channels = [77, 113, 88, 55, 8, 82, 89]

SAVE_DIR = 'playground/visualizations'
AX_SIZE = 6
gif_dir = f'{SAVE_DIR}/pixel_gifs'

def save_channel_patch_image(seed : int, value : float, row : int, col : int, channel : int):
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=channel, value=value, coord=(row, col)) 
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=False, ax_size=AX_SIZE)
    fig.suptitle(f'Synthetic patch on channel {channel}', fontsize=20)
    # Ensure the title is large and close to the figures
    fig.subplots_adjust(top=1)

    # Draw a red pixel at the location of the patch
    visualization.plot_dots(axs[1:], (row, col), color='cyan', hidden_padding = padding)
    save_dir = f'{gif_dir}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fname = f'{save_dir}/seed{seed}_col{col}.png'
    fig.savefig(fname)
    plt.show(fig)
    # plt.close(fig)
    
    return fname 

# %% 
import imageio

def save_sweep(channel : int, seed : int, coords : List[Tuple[int, int]]):
    """ Save a sweep of the maze, where the pixel patch moves from one coordinate to the next. """
    images = []
    grid = maze.get_inner_grid_from_seed(seed)
    for row, col in coords:
        if not maze.inside_inner_grid(grid, row, col): continue # Skip if the coordinate is outside the maze
        fname = save_channel_patch_image(seed, 5.6, row, col, channel=channel)
        images.append(imageio.imread(fname))
        # Delete the file
        os.remove(fname)

    target = f'{gif_dir}/c{channel}_seed{seed}.gif'
    # Overwrite if necessary
    if os.path.exists(target):
        os.remove(target)
    imageio.mimsave(target, images, duration=0.5)


# %% 
channels = [55, 88, 42]

def get_z_coords(seed : int):
    """ Get a list of (row, col) coordinates which trace the inner grid of the maze, first running along the top edge, then along the top-right to bottom-left diagonal, then to the bottom-right. """
    grid = maze.get_inner_grid_from_seed(seed)
    padding = maze.get_padding(grid)
    bound = grid.shape[0] - padding - 1 # The last row and column of the maze, in inner grid coordinates

    start = (padding, padding)
    coords = [start]
    coords += list((start[0], start[1]) for col in range(padding + 1, bound + 1))
    end = (bound, bound)
# Make a set of (x,y) coordinates which linearly interpolate from (4, 4) to (4, 12) to (12, 4) to (12, 12)
bound = 11
z_coords = list((4, col) for col in range(4, bound+1))
z_coords += list((4 + row, bound - row) for row in range(1, (bound - 4) + 1))
z_coords += list((bound, col) for col in range(5, 13))

# %%
for seed in (0, 20, 60):
    for channel in channels:
        if True: # NOTE Flag for when running the notebook
            save_sweep(channel=channel, seed=seed, coords=z_coords)

# %% Show multiple channels get updated simultaneously
# Make a maze editor and then render the maze using the human view
venv = patch_utils.get_cheese_venv_pair(seed=0)
venv.reset()
editors = maze.venv_editors(venv, check_on_dist=False, env_nums=range(1))
display(HBox(editors))

visualization.visualize_venv(venv, ax_size=3, show_padding=False, show_plot=False) # TODO implement show_padding 

# Make subplots where the left view is a double-row axis, and the right side is two rows of half-height axes

# %% Plot different channels at once
channels_to_plot = [55, 88]
assert len(channels_to_plot) % 2 == 0, 'Must have an even number of channels to plot'
# Plot figure with subplots of different sizes

fig, axs = plt.subplots(nrows=2, ncols=len(channels_to_plot)//2, sharex=True, sharey=True, figsize=(2*AX_SIZE, AX_SIZE*len(channels_to_plot)//2))

ax = axs[0, 0]
ax.set_title(f'Maze {seed}') 
visualization.visualize_venv(venv, ax=ax, show_plot=False, show_padding=False, ax_size=AX_SIZE)

for i, channel in enumerate(channels_to_plot):
    ax = axs[(i + 1) % 2, (i + 1) // 2]
    values 
    ax.set_title(f'Channel {channel}')
    ax.set_xlabel('')