from procgen_tools.imports import *
import procgen_tools.maze as maze
import procgen_tools.visualization as viz

# IMPORTANT: Files still import these from patch_utils, so for backwards compatibility, we need to import them here.
from procgen_tools.maze import get_cheese_venv_pair, get_custom_venv_pair

# %%

NUM_CHANNEL_DICT = dict(
    [
        (layer_name, models.num_channels(hook, layer_name))
        for layer_name in labels
        if layer_name != "_out"
    ]
)  # NOTE assumes existence of "labels" and "hook" variables

# %%
# Load model


def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)


# %%
def logits_to_action_plot(logits, title=""):
    """
    Plot the action logits as a heatmap, ignoring bogus repeat actions. Use px.imshow. Assumes logits is a DataArray of shape (n_steps, n_actions).
    """
    logits_np = logits.to_numpy()
    prob = t.softmax(t.from_numpy(logits_np), dim=-1)
    action_indices = models.MAZE_ACTION_INDICES
    prob_dict = models.human_readable_actions(
        t.distributions.categorical.Categorical(probs=prob)
    )
    prob_dist = t.stack(list(prob_dict.values()))
    px.imshow(
        prob_dist, y=[k.title() for k in prob_dict.keys()], title=title
    ).show()


# PATCHES
def channel_patch_or_broadcast(
    layer_name: str,
    patch_fn: Callable[[np.ndarray], np.ndarray],
    channel: int = -1,
):
    """Apply the patching function to the given channel at the given layer. If channel is -1, apply the patching function to all channels.
    """
    patch_single_channel = channel >= 0

    def patch_fn_new(outp: np.ndarray):
        new_out = patch_fn(outp[:, channel, ...])
        if isinstance(new_out, np.ndarray):
            new_out = t.from_numpy(new_out)
        if patch_single_channel:
            outp[:, channel, ...] = new_out
        else:
            outp[:] = new_out
        return outp

    return {layer_name: patch_fn_new}


def compose_patches(*patches: List[dict]):
    """Compose a list of patches into a single patch. The order of the patches is the order in which they are applied. Note that the new patch only applies for the layers which are shared by all patches.
    """
    # Find all shared keys
    shared_keys = set.intersection(*[set(patch.keys()) for patch in patches])

    # Compose patches
    patch = {}
    for key in shared_keys:
        patch[key] = lambda outp: outp
        for new_patch in patches:
            patch[key] = fn.compose(new_patch[key], patch[key])
    return patch


def get_values_diff_patch(values: np.ndarray, coeff: float, layer_name: str):
    """Get a patch function that adds to the activations at layer_name with coeff*(values[0, ...] - values[1, ...]).
    """
    vals_diff = (
        values[0, ...] - values[1, ...]
    )  # Add this to activations during forward passes
    return {layer_name: lambda outp: outp + coeff * vals_diff}


def get_zero_patch(layer_name: str, channel: int = -1):
    """Get a patch function that patches the activations at layer_name with 0.
    """
    return channel_patch_or_broadcast(
        layer_name=layer_name,
        channel=channel,
        patch_fn=lambda outp: t.zeros_like(outp),
    )


def get_mean_patch(
    layer_name: str,
    values: np.ndarray = None,
    channel: int = -1,
    num_samples: int = 50,
):
    """Get a patch that replaces the activations at layer_name with the mean of values, taken across the batch (first) dimension. If channel is specified (>= 0), take the mean across the channel dimension. If values is not specified, sample num_samples random observations and use the activations at layer_name.
    """
    patch_single_channel = channel >= 0

    if values is None:
        # Get activations at this layer and channel for a randomly sampled observation
        rand_obs = maze.get_random_obs(num_obs=num_samples, on_training=False)
        values = hook.get_value_by_label(layer_name)
    mean_vals = reduce(
        t.from_numpy(
            values[:, channel, ...] if patch_single_channel else values
        ),
        "b ... -> ...",
        "mean",
    )

    return channel_patch_or_broadcast(
        layer_name, channel=channel, patch_fn=lambda outp: mean_vals
    )


def get_random_patch(
    layer_name: str,
    hook: cmh.ModuleHook,
    channel: int = -1,
    cheese_loc: Tuple[int, int] = None,
    num_obs: int = 1,
):
    """Get a patch that replaces the activations at layer_name with a random sample from the activations at that layer. If channel is specified (>= 0), only patch that channel, leaving the rest of the layer's activations unchanged. If cheese_loc is specified, sample random observations with cheese at that location. Cycle through num_obs random observations, randomly generating the index of the observation to use at every invocation of the random patch.
    """
    assert num_obs > 0, "Must sample at least one observation"
    assert cheese_loc is None or (
        0 <= cheese_loc[0] < maze.WORLD_DIM
        and 0 <= cheese_loc[1] < maze.WORLD_DIM
    ), "Cheese location is out of bounds."

    patch_single_channel = channel >= 0

    # Get activations at this layer and channel for a randomly sampled observation
    rand_obs = maze.get_random_obs_opts(
        num_obs=num_obs, on_training=False, cheese_pos_outer=cheese_loc
    )
    hook.run_with_input(rand_obs, func=forward_func_policy)
    values = hook.get_value_by_label(
        layer_name
    )  # shape (batch, channels, ...)
    if patch_single_channel:
        values = values[:, channel, ...]  # shape (batch, ...)
    random_vals = t.from_numpy(values)  # shape (batch, ...)

    def patch_fn(outp):
        random_idx = np.random.randint(
            0, num_obs
        )  # TODO i think this only invokes once
        new_vals = random_vals[random_idx, ...]
        return new_vals

    # If patch_single_channel, this will be applied to the channel dimension; otherwise, it will be applied to the entire output
    return channel_patch_or_broadcast(
        layer_name, channel=channel, patch_fn=patch_fn
    )


def get_channel_pixel_patch(
    layer_name: str,
    channel: int,
    value: int = 1,
    coord: Tuple[int, int] = (0, 0),
    default: float = None,
):
    """Values has shape (batch, channels, ....). Returns a patch which sets the activations at layer_name to 1 in the top left corner of the given channel.

    args:
        layer_name: name of the layer to patch
        channel: channel to patch
        value: value to set the pixel at coord
        coord: coordinate of the pixel to set
        default: value to set all other pixels to. If None, set to the value of the pixel at coord in the original activations.
    """
    assert channel >= 0
    WIDTH = NUM_CHANNEL_DICT[layer_name]
    assert (
        0 <= coord[0] < WIDTH and 0 <= coord[1] < WIDTH
    ), "Coordinate is out of bounds"

    def new_corner_patch(outp):
        """outp has shape (batch, ...) -- without a channel dimension."""
        new_features = (
            t.ones_like(outp[0, ...]) * default
            if default is not None
            else outp[0, ...].clone()
        )
        new_features[coord] = value
        outp[:, ...] = new_features
        return outp

    return channel_patch_or_broadcast(
        layer_name, channel=channel, patch_fn=new_corner_patch
    )  # TODO make box activation


def combined_pixel_patch(
    layer_name: str,
    value: float,
    coord: Tuple[int, int],
    channels: List[int],
    default: float = None,
):
    """Get a patch that modifies multiple channels at once.

    args:
        layer_name: name of the layer to patch
        value: value to set the pixel at coord
        coord: coordinate of the pixel to set
        channels: list of channels to patch
        default: value to set all other pixels to. If None, preserve the values at all other pixels.
    """
    patches = [
        get_channel_pixel_patch(
            layer_name=layer_name,
            channel=channel,
            value=value,
            coord=coord,
            default=default,
        )
        for channel in channels
    ]
    combined_patch = compose_patches(*patches)
    return combined_patch


def get_multiply_patch(
    layer_name: str,
    channel: int = -1,
    pos_multiplier: float = None,
    neg_multiplier: float = None,
):
    """Get a patch that multiplies the activations at layer_name by multiplier. If channel is specified (>= 0), only multiply the given channel. If pos_multiplier is specified, multiply only positive activations by that value. If neg_multiplier is specified, multiply only negative activations by that value.
    """

    def multiply_outp(outp: t.Tensor):
        new_vals = outp
        if pos_multiplier is not None:
            new_vals[outp > 0] = outp[outp > 0] * pos_multiplier
        if neg_multiplier is not None:
            new_vals[outp < 0] = outp[outp < 0] * neg_multiplier
        return new_vals

    return channel_patch_or_broadcast(
        layer_name, channel=channel, patch_fn=multiply_outp
    )


# %%
# Infrastructure for running different kinds of seeds
def values_from_venv(
    layer_name: str, hook: cmh.ModuleHook, venv: ProcgenGym3Env
):
    """Get the values of the activations at the layer for the given venv."""
    obs = venv.reset().astype(np.float32)
    hook.run_with_input(obs, func=forward_func_policy)
    return hook.get_value_by_label(layer_name)


def patch_from_venv_pair(
    venv: ProcgenGym3Env,
    layer_name: str,
    hook: cmh.ModuleHook,
    coeff: float = 1.0,
):
    """Get a patch which creates an 'X-vector' from the given venv pair."""
    assert venv.num_envs == 2, "Must have two environments in the venv."

    values = values_from_venv(layer_name, hook, venv)
    return get_values_diff_patch(
        values=values, layer_name=layer_name, coeff=coeff
    )


def cheese_diff_values(seed: int, layer_name: str, hook: cmh.ModuleHook):
    """Get the cheese/no-cheese activations at the layer for the given seed."""
    venv = get_cheese_venv_pair(seed)
    return values_from_venv(layer_name, hook, venv)


def compare_patched_vfields(
    venv: ProcgenGym3Env,
    patches: dict,
    hook: cmh.ModuleHook,
    render_padding: bool = False,
    ax_size: int = 4,
    reuse_first: bool = True,
    show_diff: bool = True,
    show_original: bool = True,
    show_components: bool = False,
):
    """Takes as input a venv with one or two maze environments. If one and reuse_first is true, we compare vfields for original/patched on that fixed venv. If two, we show the vfield for the original on the first venv environment, and the patched on the second, and the difference between the two.

    Args:
        venv: The venv to use for the vector field.
        patches: A dictionary of patches to apply to the network.
        hook: The hook to use to get the activations.
        render_padding: Whether to render the padding around the maze.
        ax_size: The size of each axis in the plot.
        reuse_first: Whether to reuse the first environment in the venv for the patched vfield.

        show_diff: Whether to show the difference between the two vector fields.
        show_original: Whether to show the original vector field.
        show_components: Whether to show the action-based components of the vector field.
    """

    assert (
        1 <= venv.num_envs <= 2
    ), "Needs one or environments to compare the vector fields"
    venv1, venv2 = maze.copy_venv(venv, 0), maze.copy_venv(
        venv, 0 if venv.num_envs == 1 or reuse_first else 1
    )

    original_vfield = viz.vector_field(venv1, hook.network)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv2, hook.network)

    fig, axs, vf_diff = viz.plot_vfs(
        original_vfield,
        patched_vfield,
        render_padding=render_padding,
        ax_size=ax_size,
        show_diff=show_diff,
        show_original=show_original,
        show_components=show_components,
    )

    obj = {
        "patches": patches,
        "original_vfield": original_vfield,
        "patched_vfield": patched_vfield,
        "diff_vfield": vf_diff,
    }

    return fig, axs, obj


def plot_patched_vfields(
    seed: int,
    coeff: float,
    layer_name: str,
    hook: cmh.ModuleHook,
    values: Optional[np.ndarray] = None,
    venv: Optional[ProcgenGym3Env] = None,
    show_title: bool = False,
    title: str = "",
    render_padding: bool = False,
    ax_size: int = 5,
    show_components: bool = False,
):
    """Plot the original and patched vector fields for the given seed, coeff, and layer_name. If values is provided, use those values for the patching. Otherwise, generate them via a cheese/no-cheese activation diff.
    """
    values = (
        cheese_diff_values(seed, layer_name, hook)
        if values is None
        else values
    )
    patches = get_values_diff_patch(values, coeff, layer_name)
    venv = maze.copy_venv(
        get_cheese_venv_pair(seed) if venv is None else venv, 0
    )  # Get env with cheese present / first env in the pair

    fig, axs, obj = compare_patched_vfields(
        venv,
        patches,
        hook,
        render_padding=render_padding,
        ax_size=ax_size,
        show_components=show_components,
    )
    obj.update(
        {
            "seed": seed,
            "coeff": coeff,
            "patch_layer_name": layer_name,
        }
    )

    if show_title:
        fig.suptitle(
            title
            if title != ""
            else f"Level {seed} coeff {coeff} layer {layer_name}"
        )

    return fig, axs, obj
