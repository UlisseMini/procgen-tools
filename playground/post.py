# %%
%reload_ext autoreload
%autoreload 2

# %%
try:
    import procgen_tools
except ImportError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup(dl_data=False) # create directory structure and download data 

# %%
from procgen_tools.imports import *
from procgen_tools import visualization, patch_utils, maze, vfield

# %% Generate vfields for randomly generated seeds
AX_SIZE = 4

# Control show_components with a checkbox
checkbox = widgets.Checkbox(value=False, description='Show action components')

rows = 1
cols = 2

fig_out = widgets.Output()
text_out = widgets.Output()
display(text_out)
display(fig_out)

def generate_plots(max_size : int = 18, min_size : int = 3, cols : int = 2, rows : int = 1, show_vfield : bool = True):
    """ Generate rows*cols plots for random seeds with inner grid size at most max_size and at least min_size. """
    assert 3 <= min_size <= max_size <= maze.WORLD_DIM, 'Invalid min/max size'

    # Indicate that the plots are being generated    
    with text_out:
        print(f'Generating {rows*cols} plots...')

    fig, axs = plt.subplots(rows, cols, figsize=(AX_SIZE*cols, AX_SIZE*rows))
    for idx, ax in enumerate(axs.flatten()): 
        seed = np.random.randint(0, 100000)
        while maze.get_inner_grid_from_seed(seed=seed).shape[0] > max_size or maze.get_inner_grid_from_seed(seed=seed).shape[0] < min_size:
            seed = np.random.randint(0, 100000)
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        if show_vfield:
            vf = vfield.vector_field(venv, policy=hook.network)
            vfield.plot_vf(vf, ax=ax, show_components=checkbox.value, render_padding = False)
        else:
            visualization.visualize_venv(venv, mode='human', idx=0, ax=ax, show_plot=False, render_padding=False, render_mouse=False)
        ax.set_title(f'Seed: {seed:,}')
        ax.axis('off')  

    # Indicate that the plots are done being generated
    text_out.clear_output()
    fig_out.clear_output(wait=True)
    with fig_out:     
        plt.show()


# Make slider for max inner grid size
slider = widgets.IntSlider(min=3, max=25, step=1, value=18, description='Max grid size')
display(slider)

# Make a button to generate new plots
button = widgets.Button(description='Generate new plots')
button.on_click(lambda _: generate_plots(max_size = slider.value))
display(HBox([button, checkbox]))

generate_plots(max_size = slider.value)
# %% Make a totally empty venv and then visualize it
for fill_type in (maze.EMPTY, maze.CHEESE):
    venv = maze.get_filled_venv(fill_type=fill_type)
    img = visualization.visualize_venv(venv, mode='human', idx=0, show_plot=True, render_padding=True, render_mouse=True)

# %% Visualize seeds 3893 and 45816, side-by-side

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for idx, ax in enumerate(axs.flatten()):
    seed = [3893, 45816][idx]
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    visualization.visualize_venv(venv, mode='human', idx=0, ax=ax, show_plot=False, render_padding=True, render_mouse=True)
    ax.set_title(f'Seed: {seed:,}')
    ax.axis('off')

# %% Visualize seed 3893 with and without padding

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for idx, ax in enumerate(axs.flatten()):
    seed = 3893
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    visualization.visualize_venv(venv, mode='human', idx=0, ax=ax, show_plot=False, render_padding=1-idx, render_mouse=True)
    ax.set_title('Padding shown' if bool(1-idx) else 'Padding hidden')
    ax.axis('off') 

# %% Generate main GIF
import xarray as xr

# Helper predict function that matches the interface for selecting 
# an action that is expected by run_rollout from circrl.
# Uses the hooked network so patching can be applied if needed,
# and activations can be accessed.
def predict(obs, deterministic):
    obs = t.FloatTensor(obs)
    dist, value = hook.network(obs)
    if deterministic:
        act = dist.mode.numpy()
    else:
        act = dist.sample().numpy()
    return act, None

# Define a grid-gathering function to pass to cro.run_rollout
def gather_grid_data(env : ProcgenGym3Env, **kwargs):
    """ Given an environment, return the outer grid. Ignore kwargs. """
    env_state = maze.state_from_venv(env, idx=0)
    full_grid = env_state.full_grid(with_mouse=True)
    return full_grid 

custom_data_funcs = {'grid': gather_grid_data}

# Define a list of patches to apply to the network
target_locations = [(5, 5), (5, 11), (4, 9), (4, 11), (8, 11), (8, 9), (4, 8), (5, 5)]
timesteps = 50


# Initialize the venv of choice
seed = 0
venv = maze.create_venv(num=1, start_level=seed, num_levels=1)

# Run the rollout and show the video
for idx, loc in enumerate(target_locations):
    patch = patch_utils.get_channel_pixel_patch(layer_name=default_layer, channel=55, value=5.6, coord=loc)

    # Run a patched rollout
    with hook.use_patches(patch):
        new_seq, _, _ = cro.run_rollout(predict, venv, max_steps=timesteps, deterministic=False, custom_data_funcs=custom_data_funcs)
    new_grids = new_seq.custom['grid']

    # Append more observational data to the first DataArray
    grid_seq = xr.concat([grid_seq, new_grids], dim='step') if idx > 0 else new_grids # "step" is the existing dimension

#%% Make a gif from the grid sequence
# Render all of the observations -- somehow we need to get grids or smth so that we avoid the padding?
np_grids = grid_seq.values
inner_grid = maze.get_inner_grid_from_seed(seed)
padding = maze.get_padding(grid=inner_grid)

import PIL

imgs = []
fig, ax = plt.subplots(1, 1, figsize=(AX_SIZE, AX_SIZE))


for idx, grid in enumerate(np_grids):
    # Render the grid
    ax.clear()
    venv = maze.venv_from_grid(grid)
    visualization.visualize_venv(venv, mode='human', ax=ax, idx=0, show_plot=False, render_padding=False, render_mouse=True)
    
    # Plot the dot for the current timestep
    visualization.plot_dots(axes=[ax], coord=target_locations[idx // timesteps], hidden_padding=padding, color='cyan')
    
    # Get the axis as an image
    fig.canvas.draw()
    img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    imgs.append(img)

# Make a gif from the images
import imageio
SAVE_DIR = 'playground/visualizations'
gif_dir = f'{SAVE_DIR}/pixel_gifs'
target = f'{gif_dir}/retargeting_rollout.gif'
imageio.mimsave(target, imgs, duration=0.08)
# %%
