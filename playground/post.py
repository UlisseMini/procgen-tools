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
AX_SIZE = 4
# %% Generate vfields for randomly generated seeds
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

# %% Visualize two seeds, side-by-side

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for idx, ax in enumerate(axs.flatten()):
    seed = [59195, 1442][idx]
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    # Plot the vf
    vf = vfield.vector_field(venv, policy=hook.network)
    vfield.plot_vf(vf, ax=ax, show_components=False, render_padding = False)
    
    ax.set_title(f'Seed: {seed:,}')
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

def get_grid_seq(seed : int, target_locations : List[Tuple[int, int]], timesteps : int = 50) -> np.ndarray:
    """ Generate a sequence of grids for a given seed, taking timesteps steps for each target_location. """
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
    return grid_seq.values # Return the numpy array

# %%
seed = 0
render_preview = True
target_locations = [(5, 5), (5, 11), (4, 9), (4, 11), (8, 11), (8, 9), (4, 8), (5, 5)]
preview_GIF_locations = [(8, 11), (8,9), (8, 11)]

locations = preview_GIF_locations if render_preview else target_locations
timesteps = 50
np_grids = get_grid_seq(seed=seed, target_locations=locations, timesteps=timesteps)

#%% Make a gif from the grid sequence
inner_grid = maze.get_inner_grid_from_seed(seed)
padding = maze.get_padding(grid=inner_grid)

imgs = []
fig, ax = plt.subplots(1, 1, figsize=(AX_SIZE, AX_SIZE))

for idx, grid in enumerate(np_grids):
    # Render the grid
    ax.clear()
    venv = maze.venv_from_grid(grid)
    visualization.visualize_venv(venv, mode='human', ax=ax, idx=0, show_plot=False, render_padding=False, render_mouse=True)
    
    # Plot the dot for the current timestep
    visualization.plot_dots(axes=[ax], coord=locations[idx // timesteps], hidden_padding=padding, color='red')
    
    # Get the axis as an image
    img = visualization.img_from_fig(fig, palette=imgs[0] if idx > 0 else None)
    imgs.append(img)

# %% Save the images as a GIF
SAVE_DIR = 'playground/visualizations'
gif_dir = f'{SAVE_DIR}/pixel_gifs'
target = f'{gif_dir}/retargeting_rollout_{seed}_{"preview" if render_preview else "full"}'

start_step = 29 # Start the gif at this frame

imgs[start_step].save(target + '.gif', format="GIF", save_all=True, append_images=imgs[start_step+1:], duration=40 if render_preview else 50, loop=0)

print(f'Gif saved to {target}')

# %% Applying all cheese patches
AX_SIZE = 6

cheese_channels = [77, 113, 44, 88, 55, 42, 7, 8, 82, 99] 
effective_channels = [77, 113, 88, 55, 8, 82, 89]

@interact
def apply_all_cheese_patches(seed=IntSlider(min=0, max=100, step=1, value=0), value=FloatSlider(min=-10, max=10, step=0.1, value=2.3), row=IntSlider(min=0, max=15, step=1, value=5), col=IntSlider(min=0, max=15, step=1, value=5), channel_list=Dropdown(options=[effective_channels, cheese_channels], value=effective_channels), mask_channels=Checkbox(value=False)):
    render_padding = False
    padding = maze.get_padding(grid=maze.get_inner_grid_from_seed(seed))

    combined_patch = patch_utils.combined_pixel_patch(layer_name=default_layer, value=value, coord=(row, col), channels=channel_list, default=-.2 if mask_channels else None)

    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, combined_patch, hook, render_padding=render_padding, ax_size=AX_SIZE)

    # Draw a red pixel at the location of the patch
    visualization.plot_dots(axs[1:], (row, col), hidden_padding=0 if render_padding else padding) # Note this can go out of bounds if render_padding is False 
    plt.show()

    button = visualization.create_save_button(prefix=f'{SAVE_DIR}/{"all" if channel_list == cheese_channels else "effective"}_cheese_patches_', fig=fig, descriptors=defaultdict(seed=seed, value=value, row=row, col=col, mask_channels=mask_channels))
    display(button)
interesting_settings = [{'seed': 83, 'value': 2, 'row': 7, 'col': 7, 'channel_list': cheese_channels, 'mask_channels': True}]

# %%
