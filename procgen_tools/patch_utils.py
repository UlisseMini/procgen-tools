from procgen_tools.imports import * 
import procgen_tools.maze as maze
import procgen_tools.vfield as vfield

def get_cheese_venv_pair(seed: int, has_cheese_tup : Tuple[bool, bool] = (True, False)):
    "Return a venv of 2 environments from a seed, with cheese in the first environment if has_cheese_tup[0] and in the second environment if has_cheese_tup[1]."
    venv = maze.create_venv(num=2, start_level=seed, num_levels=1)

    for idx in range(2):
        if has_cheese_tup[idx]: continue # Skip if we want cheese in this environment
        maze.remove_cheese(venv, idx=idx)

    return venv

def get_custom_venv_pair(seed: int, num_envs=2):
    """ Allow the user to edit num_envs levels from a seed. Return a venv containing both environments. """
    venv = maze.create_venv(num=num_envs, start_level=seed, num_levels=1)
    display(HBox(maze.venv_editor(venv, check_on_dist=False)))
    return venv

def load_venv_pair(path: str):
    """ Load a venv pair from a file. """
    venv = maze.create_venv(num=2, start_level=1, num_levels=1)
    with open(path_prefix + path, 'rb') as f:
        state_bytes = pkl.load(f) 
    venv.env.callmethod('set_state', state_bytes)
    def _step(*_, **__):
        raise NotImplementedError('This venv is only used as a template for copy_env')
    venv.step = _step
    return venv

# %%
# Load model

def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)


# %% 
def logits_to_action_plot(logits, title=''):
    """
    Plot the action logits as a heatmap, ignoring bogus repeat actions. Use px.imshow. Assumes logits is a DataArray of shape (n_steps, n_actions).
    """
    logits_np = logits.to_numpy()
    prob = t.softmax(t.from_numpy(logits_np), dim=-1)
    action_indices = models.MAZE_ACTION_INDICES
    prob_dict = models.human_readable_actions(t.distributions.categorical.Categorical(probs=prob))
    prob_dist = t.stack(list(prob_dict.values()))
    px.imshow(prob_dist, y=[k.title() for k in prob_dict.keys()],title=title).show()

def num_channels(hook : cmh.ModuleHook, layer_name : str):
    """ Get the number of channels in the given layer. """
    # Ensure hook has been run on dummy input
    assert hook.get_value_by_label(layer_name) is not None, "Hook has not been run on any input"
    return hook.get_value_by_label(layer_name).shape[1]

NUM_CHANNEL_DICT = dict([(layer_name, num_channels(hook, layer_name)) for layer_name in labels if layer_name != '_out']) # NOTE assumes existence of "labels" and "hook" variables

# PATCHES
def channel_patch_or_broadcast(layer_name : str,  patch_fn : Callable[[np.ndarray], np.ndarray], channel : int = -1):
    """ Apply the patching function to the given channel at the given layer. If channel is -1, apply the patching function to all channels. """
    patch_single_channel = channel >= 0

    def patch_fn_new(outp : np.ndarray):
        new_out = patch_fn(outp[:, channel, ...])
        if isinstance(new_out, np.ndarray):
            new_out = t.from_numpy(new_out)
        if patch_single_channel: 
            outp[:, channel, ...] = new_out
        else: 
            outp[:] = new_out
        return outp
    return {layer_name: patch_fn_new} 

def get_values_diff_patch(values: np.ndarray, coeff: float, layer_name: str):
    """ Get a patch function that patches the activations at layer_name with coeff*(values[0, ...] - values[1, ...]). """
    cheese = values[0,...]
    no_cheese = values[1,...]
    assert np.any(cheese != no_cheese), "Cheese and no cheese values are the same"

    cheese_diff = cheese - no_cheese # Add this to activations during forward passes
    return {layer_name: lambda outp: outp + coeff*cheese_diff} # can't pickle
    # return {layer_name: cmh.PatchDef(value=coeff*cheese_diff, mask=np.array(True))} # can pickle

def get_zero_patch(layer_name: str, channel : int = -1):
    """ Get a patch function that patches the activations at layer_name with 0. """
    return channel_patch_or_broadcast(layer_name=layer_name, channel=channel, patch_fn=lambda outp: t.zeros_like(outp))

def get_mean_patch(layer_name: str, values: np.ndarray = None, channel : int = -1, num_samples : int = 50):
    """ Get a patch that replaces the activations at layer_name with the mean of values, taken across the batch (first) dimension. If channel is specified (>= 0), take the mean across the channel dimension. If values is not specified, sample num_samples random observations and use the activations at layer_name. """ 
    patch_single_channel = channel >= 0

    if values is None:
        # Get activations at this layer and channel for a randomly sampled observation
        rand_obs = maze.get_random_obs(num_obs=num_samples, on_training=False)
        values = hook.get_value_by_label(layer_name)
    mean_vals = reduce(t.from_numpy(values[:, channel, ...] if patch_single_channel else values), 'b ... -> ...', 'mean')

    return channel_patch_or_broadcast(layer_name, channel=channel, patch_fn=lambda outp: mean_vals) 

def get_random_patch(layer_name : str, hook : cmh.ModuleHook, channel : int = -1):
    """ Get a patch that replaces the activations at layer_name with a random sample from the activations at that layer. If channel is specified (>= 0), only patch that channel, leaving the rest of the layer's activations unchanged. """
    patch_single_channel = channel >= 0
    
    # Get activations at this layer and channel for a randomly sampled observation
    rand_obs = maze.get_random_obs(num_obs=1, on_training=False) # TODO switch to "opts"
    hook.run_with_input(rand_obs, func=forward_func_policy)
    values = hook.get_value_by_label(layer_name) # shape (batch, channels, ...)
    if patch_single_channel:
        values = values[:, channel, ...] # shape (batch, ...)
    random_vals = t.from_numpy(values[0, ...]) # shape (...)

    """ 
    rand_obs = (1, 3, 64, 64)
    values = (1, 128, 16, 16)
    if patch_single_channel: 
        values = (1, 16, 16)
        random_vals: (16, 16) 
    else:
        values: (1, 128, 16, 16)
        random_vals: (128, 16, 16)
    outp (96, 128, 16, 16)
    
    Want to apply patch to each output channel, so we need to broadcast the random values to the entire output
    

    """

    def patch_fn(outp): 
        return random_vals 
        
    # If patch_single_channel, this will be applied to the channel dimension; otherwise, it will be applied to the entire output
    return channel_patch_or_broadcast(layer_name, channel=channel, patch_fn=patch_fn)


def get_channel_pixel_patch(layer_name: str, channel : int, value : int = 1, coord : Tuple[int, int] = (0, 0)):
    """ Values has shape (batch, channels, ....). Returns a patch which sets the activations at layer_name to 1 in the top left corner of the given channel. """
    assert channel >= 0
    WIDTH = NUM_CHANNEL_DICT[layer_name]
    assert 0 <= coord[0] < WIDTH and 0 <= coord[1] < WIDTH, "Coordinate is out of bounds"    

    default = -.2
    def new_corner_patch(outp): 
        """ outp has shape (batch, ...) -- without a channel dimension. """
        new_features = t.ones_like(outp[0, ...]) * default
        new_features[coord] = value
        outp[:, ...] = new_features
        return outp

    return channel_patch_or_broadcast(layer_name, channel=channel, patch_fn=new_corner_patch) # TODO make box activation

def get_multiply_patch(layer_name : str, channel : int = -1, multiplier : float = 2):
    """ Get a patch that multiplies the activations at layer_name by multiplier. If channel is specified (>= 0), only multiply the given channel. """
    return channel_patch_or_broadcast(layer_name, channel=channel, patch_fn=lambda outp: outp * multiplier)

# %% 
# Infrastructure for running different kinds of seeds
def values_from_venv(layer_name: str, hook: cmh.ModuleHook, venv: ProcgenGym3Env):
    """ Get the values of the activations at the layer for the given venv. """
    obs = venv.reset().astype(np.float32) 
    hook.run_with_input(obs, func=forward_func_policy)
    return hook.get_value_by_label(layer_name)

def cheese_diff_values(seed:int, layer_name:str, hook: cmh.ModuleHook): # TODO rename to cheese_ablation_values ?
    """ Get the cheese/no-cheese activations at the layer for the given seed. """
    venv = get_cheese_venv_pair(seed) 
    return values_from_venv(layer_name, hook, venv)

def compare_patched_vfields(venv : ProcgenGym3Env, patches : dict, hook: cmh.ModuleHook, render_padding: bool = False, ax_size : int = 4, reuse_first : bool = True, show_diff : bool = True):
    """ Takes as input a venv with one or two maze environments. If one and reuse_first is true, we compare vfields for original/patched on that fixed venv. If two, we show the vfield for the original on the first venv environment, and the patched on the second, and the difference between the two. """

    assert 1 <= venv.num_envs <= 2, "Needs one or environments to compare the vector fields"
    venv1, venv2 = maze.copy_venv(venv, 0), maze.copy_venv(venv, 0 if venv.num_envs == 1 or reuse_first else 1)

    original_vfield = vfield.vector_field(venv1, hook.network)
    with hook.use_patches(patches):
        patched_vfield = vfield.vector_field(venv2, hook.network)

    fig, axs, vf_diff = vfield.plot_vfs(original_vfield, patched_vfield, render_padding=render_padding, ax_size=ax_size, show_diff=show_diff)

    obj = {
        'patches': patches,
        'original_vfield': original_vfield,
        'patched_vfield': patched_vfield,
        'diff_vfield': vf_diff,
    }

    return fig, axs, obj


def plot_patched_vfields(seed: int, coeff: float, layer_name: str, hook: cmh.ModuleHook, values: Optional[np.ndarray] = None, venv: Optional[ProcgenGym3Env] = None, show_title: bool = False, title:str = '', render_padding: bool = False, ax_size : int = 5):
    """ Plot the original and patched vector fields for the given seed, coeff, and layer_name. If values is provided, use those values for the patching. Otherwise, generate them via a cheese/no-cheese activation diff. """
    values = cheese_diff_values(seed, layer_name, hook) if values is None else values
    patches = get_values_diff_patch(values, coeff, layer_name) 
    venv = maze.copy_venv(get_cheese_venv_pair(seed) if venv is None else venv, 0) # Get env with cheese present / first env in the pair

    fig, axs, obj = compare_patched_vfields(venv, patches, hook, render_padding=render_padding, ax_size=ax_size)
    obj.update({
        'seed': seed,
        'coeff': coeff,
        'patch_layer_name': layer_name,
        })

    if show_title:
        fig.suptitle(title if title != '' else f"Level {seed} coeff {coeff} layer {layer_name}")

    return fig, axs, obj