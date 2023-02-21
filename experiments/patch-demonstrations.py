# %%
%reload_ext autoreload
%autoreload 2

# %%
# Install procgen tools if needed
try:
    import procgen_tools
except ImportError or ModuleNotFoundError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup() # create directory structure and download data 

# %%
from procgen_tools.imports import *
from procgen_tools.procgen_imports import *

save_dir = os.getcwd() + '/visualizations'

# %% Automatically find the highest-change patch for each seed
def argmax_coords(seed : int, value : float = 5.6, top_k : int = 5):
    # Get the top-k patches for each seed
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    venv1 = copy_venv(venv, 0)
    original_vf = vfield.vector_field(venv1, hook.network)

    top_coords = []
    for row in range(16):
        for col in range(16):
            patches = get_channel_pixel_patch(layer_name=default_layer, channel=55, value=value, coord=(row, col))
            with hook.use_patches(patches):
                patched_vf = vfield.vector_field(venv1, hook.network)
            diff = vfield.get_vf_diff(original_vf, patched_vf)
            top_coords.append((diff, row, col))
    # Take the norm of each diff vf
    top_coords = sorted(top_coords, key=lambda x: sum(np.linalg.norm(arrow) for arrow in x[0]['arrows']), reverse=True) 
    return [(row, col) for (_, row, col) in top_coords[:top_k]]

# %% Visualize the top-k patches for each seed
def visualize_top_k_patches(seed : int, value : float = 5.6, top_k : int = 5):
    coords = argmax_coords(seed, value, top_k)
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    venv1 = copy_venv(venv, 0)
    # Use compare_vector_fields for each patch
    for i, (row, col) in enumerate(coords):
        patches = get_channel_pixel_patch(layer_name=default_layer, channel=55, value=value, coord=(row, col))
        fig, axs, info = compare_patched_vfields(venv1, patches, hook)
        fig.suptitle(f'Seed {seed}, patch {i+1}/{top_k} at ({row}, {col})')
        for idx in (1,2): 
            plot_pixel_dot(axs[idx], row, col)
    plt.show()

for seed in (0, 4, 5): # TODO this is sooo slow
    visualize_top_k_patches(seed, top_k=50)

# TODO make wider patch
# TODO set to top-right and collect statistics 

# %% Sanity-check that the patching performance is not changed at the original square
for seed in range(5):
    cheese_pair = patch_utils.get_cheese_venv_pair(seed=seed, has_cheese_tup=(False, True))
    values = patch_utils.cheese_diff_values(seed, default_layer, hook)
    patches = patch_utils.get_values_diff_patch(values, coeff=-1, layer_name=default_layer)

    original_vfield = vfield.vector_field(copy_venv(cheese_pair, 0), hook.network)
    with hook.use_patches(patches):
        patched_vfield = vfield.vector_field(copy_venv(cheese_pair, 1), hook.network)
    diff_vf = vfield.get_vf_diff(original_vfield, patched_vfield)

    mouse_pos = maze.get_mouse_pos(maze.get_inner_grid_from_seed(seed=seed))

    mouse_idx = original_vfield['legal_mouse_positions'].index(mouse_pos)
    orig_arrow = original_vfield['arrows'][mouse_idx]
    patch_arrow = patched_vfield['arrows'][mouse_idx]
    diff = np.linalg.norm(np.array(orig_arrow) - np.array(patch_arrow))
    assert diff < 1e-3, "The patching performance is changed at the original square"

# %% Vfields on each maze
""" The vector field is a plot of the action probabilities for each state in the maze. Let's see what the vector field looks like for a given seed. We'll compare the vector field for the original and patched networks. 
"""
@interact
def interactive_patching(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-3, max=3, step=0.1, value=-1)):
    fig, _, _ = plot_patched_vfields(seed, coeff, default_layer, hook)
    plt.show()

# %% Patching from a fixed seed
""" Let's see what happens when we patch the network from a fixed seed. We'll compare the vector field for the original and patched networks.
"""
value_seed = 0
values_tup = cheese_diff_values(value_seed, default_layer, hook), value_seed
for seed in range(10):  
    run_seed(seed, hook, [-1], values_tup=values_tup)

# %% We can construct a patch which averages over a range of seeds, and see if that generalizes better (it doesn't)
seeds = slice(int(10e5),int(10e5+19))
last_labels = ['embedder.block3.res2.conv2_out', 'embedder.block3.res2.resadd_out', 'embedder.relu3_out', 'embedder.flatten_out', 'embedder.fc_out', 'embedder.relufc_out']
@interact
def interactive_patching(target_seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-2, max=2, step=0.1, value=-1), layer_name=Dropdown(options=last_labels, value=last_labels[0])):
    values = np.zeros_like(cheese_diff_values(0, layer_name, hook))
    for seed in range(seeds.start, seeds.stop):
        # Make values be rolling average of values from seeds
        values = (seed-seeds.start)/(seed-seeds.start+1)*values + cheese_diff_values(seed, layer_name, hook)/(seed-seeds.start+1)

    fig, _, _ = plot_patched_vfields(target_seed, coeff, layer_name, hook, values=values)
    plt.show()

# %% Patching with a random vector 
""" Are we just seeing noise? Let's try patching with a random vector and see if that works. First, let's find appropriate-magnitude random vectors."""
rand_magnitude = .25
for mode in ['random', 'cheese']:
    vectors = []
    for value_seed in range(100):
        if mode == 'random':
            vectors.append(np.random.randn(*cheese_diff_values(0, default_layer, hook).shape, ) * rand_magnitude)
        else:
            vectors.append(cheese_diff_values(value_seed, default_layer, hook))
        
    norms = [np.linalg.norm(v) for v in vectors]
    print(f'For {mode}-vectors, the norm is {np.mean(norms):.2f} with std {np.std(norms):.2f}. Max absolute-value difference of {np.max(np.abs(vectors)):.2f}.')

# %% Run the patches
values = np.random.randn(*cheese_diff_values(0, default_layer, hook).shape) * rand_magnitude
# Cast this to float32
values = values.astype(np.float32)
print(np.max(values).max())
for seed in range(5):
    run_seed(seed, hook, [-1], values_tup=(values, 'garbage'))

# It doesn't work, and destroys performance. In contrast, the cheese vector has a targeted and constrained effect on the network (when not transferring to other mazes), and does little when attempting transfer. This seems intriguing.

# %% Zero out each layer
@interact
def run_label(seed=IntSlider(min=0, max=20, step=1, value=0), zero_target=Dropdown(options=labels, value='embedder.block2.res1.conv2_out')):
    venv = create_venv(num=1, start_level=seed, num_levels=1)
    patches = get_zero_patch(layer_name=zero_target)
    fig, axs, info = compare_patched_vfields(venv, patches, hook, ax_size=5)
    # title the fig with layer
    fig.suptitle(zero_target)
    plt.show()

# %% Generate random mouse observations and then mean-ablate
obs = maze.get_random_obs(50, spawn_cheese=False)

@interact 
def mean_ablate(seed=IntSlider(min=0, max=20, step=1, value=0), layer_name=Dropdown(options=labels, value='embedder.block3.res2.resadd_out')):
    venv = create_venv(num=1, start_level=seed, num_levels=1)
    hook.run_with_input(obs)
    random_values = hook.get_value_by_label(layer_name)
    patches = get_mean_patch(random_values, layer_name=layer_name) 
    fig, axs, info = compare_patched_vfields(venv, patches, hook, ax_size=5)
    # title the fig with layer_name
    fig.suptitle(f'Mean patching layer {layer_name}')
    # Ensure the title is close to the plots
    fig.subplots_adjust(top=1.05)
    plt.show() 


# %% Patching different layers
""" We chose the layer block2.res1.resadd_out because it seemed to have a strong effect on the vector field. Let's see what happens when we patch other layers. """

@interact
def run_all_labels(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-3, max=3, step=0.1, value=-1), layer_name=Dropdown(options=labels)):
    fig, _, _ = plot_patched_vfields(seed, coeff, layer_name, hook)
    plt.show()    
    print(f'Patching {layer_name} layer')

# %% Try all patches at once 
@interact 
def run_all_patches(seed=IntSlider(min=0, max=20, step=1, value=0), coeff=FloatSlider(min=-1, max=1, step=0.025, value=-.05)):
    venv = patch_utils.get_cheese_venv_pair(seed) 
    patches = {}
    for layer_name in labels:
        if layer_name == 'fc_value_out': continue
        values = values_from_venv(venv, hook, layer_name)
        patches.update(get_values_diff_patch(values=values, coeff=coeff, layer_name=layer_name))
        
    fig, _, _ = compare_patched_vfields(venv, patches, hook)
    plt.show()


# %% Check how patch transferability changes with cheese location 
GENERATE_NUM = 50 # Number of seeds to generate

def test_transfer(patches : dict, source_seed : int = 0, col_translation : int = 0, row_translation : int = 0, target_index : int = 0):
    """ Visualize what happens if patches are transferred to a maze with the cheese translated by the given amount. TODO refactor into a helper class, with hook and other variables saved as class variables.
    TODO take list of source_seeds
    
    Args:
        patches (dict): The patches to transfer.
        source_seed (int): The seed of the maze to transfer from.
        col_translation (int): The number of columns to translate the cheese by.
        row_translation (int): The number of rows to translate the cheese by.
        target_index (int): The index of the target maze to use, among the seeds generated or searched for. 
        skip_seed (int): The seed to skip when searching for a target maze.
    """
    cheese_location = maze.get_cheese_pos_from_seed(source_seed)

    assert cheese_location[0] < maze.WORLD_DIM - row_translation, f"Cheese is too close to the bottom for it to be translated by {row_translation}."
    assert cheese_location[1] < maze.WORLD_DIM - col_translation, f"Cheese is too close to the right for it to be translated by {col_translation}."

    seeds, grids = maze.generate_mazes_with_cheese_at_location((cheese_location[0] , cheese_location[1]+col_translation), num_mazes = GENERATE_NUM, skip_seed=source_seed)
    venv = maze.venv_from_grid(grid=grids[target_index])
    fig, _, _ = compare_patched_vfields(venv, patches, hook, render_padding=False)

    display(fig)
    print(f'The true cheese location is {cheese_location}. The new location is row {cheese_location[0] + row_translation}, column {cheese_location[1]+col_translation}.\nRendered seed: {seeds[target_index]}.')

# %% Synthetic transfer to same cheese locations
""" Most levels don't have cheese in the same spot. Let's try a synthetic transfer, where we find levels with an open spot at the appropriate location, and then move the cheese there. """
@interact
def test_synthetic_transfer(source_seed=IntSlider(min=0, max=20, step=1, value=0), col_translation=IntSlider(min=-5, max=5, step=1, value=0), row_translation=IntSlider(min=-5, max=5, step=1, value=0), target_index=IntSlider(min=0, max=GENERATE_NUM-1, step=1, value=0)):
    values = cheese_diff_values(source_seed, default_layer, hook)
    patches = get_values_diff_patch(values, coeff=-1, layer_name=default_layer)
    test_transfer(patches, source_seed, col_translation, row_translation, target_index, skip_seed=source_seed)

# %% Try generating two disjoint patches and combining both
def combine_patches(patches_lst : List[dict]):
    assert len(patches_lst) > 0, "Must provide at least one patch."
    # Assert they all have the same keys TODO relax this req?
    for patches in patches_lst:
        assert patches.keys() == patches_lst[0].keys(), "All patches must have the same keys."
    
    # Combine the patches
    patches = {}
    for key in patches_lst[0].keys():
        patches[key] = lambda outp: sum([patch[key](outp) for patch in patches_lst]) - outp*(len(patches_lst)-1) # Don't double-add the original vector
    return patches 

patch_lst = []
source_seeds = (0, 2) # TODO just get small 5x5 covering?
# First 50 works horribly
for seed in source_seeds:
    values = cheese_diff_values(seed, default_layer, hook)
    location = maze.get_cheese_pos_from_seed(seed)
    print(f'Cheese location for seed {seed}: {location}.')
    patch_lst.append(get_values_diff_patch(values, coeff=-1, layer_name=default_layer))
combined_patch = combine_patches(patch_lst)

@interact
def test_multiple_transfer(source_seed=Dropdown(options=source_seeds), target_index=IntSlider(min=0, max=GENERATE_NUM-1, step=1, value=0)): # NOTE investigate
    test_transfer(combined_patch, source_seed=source_seed, target_index=target_index)

# %% See if the cheese patch blinds the agent
values = cheese_diff_values(0, default_layer, hook)
patches = get_values_diff_patch(values, coeff=-1, layer_name=default_layer)

@interact 
def compare_with_original(seed=IntSlider(min=0, max=10000, step=1, value=0)):
    # Close out unshown queued plots
    plt.close('all')
    venv_pair = patch_utils.get_cheese_venv_pair(seed, has_cheese_tup = (False, True))
    fig, axs, info = compare_patched_vfields(venv_pair, patches, hook, render_padding=False, reuse_first=False) 
    plt.show()

# %%
