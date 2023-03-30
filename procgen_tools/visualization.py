from procgen_tools.imports import *
from procgen_tools import maze
from typing import Dict
import PIL
from warnings import warn
from torch import nn
import torch

# Getting an image from figures
def img_from_fig(
    fig: plt.Figure, palette: PIL.Image = None, tight_layout: bool = True
):
    """Get an image from a matplotlib figure. If palette is not None, then the image is quantized to the palette.
    """
    # Prepare the fig
    if tight_layout:
        fig.tight_layout()
    fig.canvas.draw()

    # Get the image from the figure
    img = PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    img = img.quantize(
        colors=254, method=PIL.Image.Quantize.MAXCOVERAGE, palette=palette
    )
    return img


# LABEL HANDLING
def format_label(label: str):
    """Format a label for display in the"""
    return label.replace("embedder.", "")


def expand_label(label: str):
    if not (label.startswith("fc_policy") or label.startswith("fc_value")):
        return "embedder." + label
    else:
        return label


def format_labels(labels: List[str]):
    """Format labels for display in the"""
    return list(map(format_label, labels))


AGENT_OBS_WIDTH = 64


def get_impala_num(label: str):
    """Get the block number of a layer."""
    if not label.startswith("embedder.block"):
        raise ValueError(f"Not in the Impala blocks.")

    # The labels are formatted as embedder.block{blocknum}.{residual_block_num}
    return int(label.split(".")[1][-1])


def get_residual_num(label: str):
    """Get the residual block number of a layer."""
    if not label.startswith("embedder.block"):
        raise ValueError(f"Not in the Impala blocks.")

    # The labels are formatted as embedder.block{blocknum}.{residual_block_num}
    return int(label.split(".")[2][-1])


# Plotting
def is_internal_activation(label: str):
    """Return True if the label is an internal activation, i.e. not an input or output.
    """
    # Check if 'in' is in the label
    if "in" in label:
        return False
    # Check if 'out' is in the label
    if "out" in label and "embedder" not in label:
        return False
    return True


def plot_layer_stats(
    hook: cmh.ModuleHook, mode: str = "activations", fig: go.Figure = None
):
    """Create and show a plotly bar chart of the number of activations per layer of policy. The default mode is "activations", which plots the number of activations per layer. If mode is "parameters", then the number of parameters per layer is plotted.
    """
    if mode not in ("activations", "parameters"):
        raise ValueError(
            f"mode must be either 'activations' or 'parameters', got {mode}."
        )

    # Get the number of activations/parameters per layer
    quantities = {}
    zipped_list = (
        filter(
            lambda tup: is_internal_activation(tup[0]),
            hook.values_by_label.items(),
        )
        if mode == "activations"
        else hook.network.named_parameters()
    )
    for name, quantity in zipped_list:
        quantities[format_label(name)] = quantity.numel()
    total_quantity = sum(quantities.values())

    # Aggregate bias quantities
    if mode == "parameters":
        bias_quants = {
            label: quantity
            for label, quantity in quantities.items()
            if "bias" in label
        }
        for label, quantity in bias_quants.items():
            if (
                "bias" in label
            ):  # If the label is a bias, add the quantity to the corresponding weight
                quantities[label.replace("bias", "weight")] += quantity
                del quantities[label]

    key_list = [
        key.replace(".weight", "") if mode == "parameters" else key
        for key in quantities.keys()
    ][::-1]
    values_list = list(quantities.values())[::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                y=key_list,
                x=values_list,
                orientation="h",
                textposition="outside",
                texttemplate="%{text:,}",
            )
        ]
    )

    formatted_total = format(total_quantity, ",")
    # Set layout
    fig.update_layout(
        title={
            "text": (
                f"Model {mode.title()} Per Layer (Total: {formatted_total})"
            ),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title=mode.title()
        if mode == "parameters"
        else "Activation count",
        yaxis_title="Layer" if mode == "parameters" else "Activations",
    )

    # Ensure there's enough height to see all the x labels
    fig.update_layout(height=max(500, 20 * len(quantities)))

    # Set mouse-over text to show the number of quantity
    fig.update_traces(
        text=values_list,
        hovertemplate="%{text:,}" + f" {mode}<extra></extra>",
        hoverlabel=dict(bgcolor="white"),
    )

    # Set x axis to be logarithmic
    fig.update_xaxes(type="log")

    fig.show()


# Navigating the feature maps
def get_stride(label: str):
    """Get the stride of the layer referred to by label. How many pixels required to translate a single entry in the feature maps of label.
    """
    if not label.startswith("embedder.block"):
        raise ValueError(f"Not in the Impala blocks.")

    block_num = get_impala_num(label)
    if (
        "conv_out" in label
    ):  # Before that Impala layer's maxpool has been applied
        block_num -= 1
    return 2**block_num


dummy_venv = maze.get_cheese_venv_pair(seed=0)
human_view = dummy_venv.env.get_info()[0]["rgb"]
PIXEL_SIZE = human_view.shape[0]  # width of the human view input image


def get_pixel_loc(val: int, channel_size: int = 16):
    """Given a single channel position value, find the pixel location that corresponds to that channel.
    """
    assert (
        val < channel_size
    ), f"channel_pos {val} must be less than channel_size {channel_size}"
    assert val >= 0, f"channel_pos {val} must be non-negative"

    scale = PIXEL_SIZE / channel_size
    return int(scale * (val + 0.5))


def get_pixel_coords(
    channel_pos: Tuple[int, int], channel_size: int = 16, flip_y: bool = True
):
    """Given a channel position, find the pixel location that corresponds to that channel. If flip_y is True, the y-axis will be flipped from the underlying numpy coords to the conventional human rendering format.
    """
    row, col = channel_pos
    assert 0 <= row < channel_size and 0 <= col < channel_size, (
        f"channel_pos {channel_pos} must be within the channel_size"
        f" {channel_size}."
    )

    if flip_y:
        row = (channel_size - 1) - row

    return get_pixel_loc(row, channel_size), get_pixel_loc(col, channel_size)


def plot_pixel_dot(
    ax: plt.Axes,
    row: int,
    col: int,
    color: str = "r",
    size: int = 50,
    hidden_padding: int = 0,
):
    """Plot a dot on the pixel grid at the given row and column of the block2.res1.resadd_out channel. hidden_padding is the number of tiles which are not shown in the human view, presumably due to render_padding being False in some external call.
    """
    px_row, px_col = get_pixel_coords((row, col))
    padding_offset = (PIXEL_SIZE / maze.WORLD_DIM) * hidden_padding
    dot_rescale_from_padding = maze.WORLD_DIM / (
        maze.WORLD_DIM - hidden_padding
    )

    # Ensure the dot is within the bounds of the pixel grid
    new_row, new_col = (coord - padding_offset for coord in (px_row, px_col))
    if 0 <= new_row <= PIXEL_SIZE and 0 <= new_col <= PIXEL_SIZE:
        ax.scatter(
            y=new_row, x=new_col, c=color, s=size * dot_rescale_from_padding
        )


def plot_dots(
    axes: List[plt.Axes],
    coord: Tuple[int, int],
    color: str = "red",
    flip_y: bool = True,
    is_grid: bool = False,
    hidden_padding: int = 0,
):
    """Plot dots on the given axes, given a channel coordinate in the full grid. If flip_y, flips the y-axis. If is_grid, assumes the coord is an outer grid coordinate.

    Args:
        axes: The axes to plot on.
        coord: The coordinate to plot.
        color: The color of the dot.
        flip_y: Whether to flip the y-axis.
        is_grid: Whether the coord is a grid coordinate.
        hidden_padding: The padding in this maze which should be ignored when plotting.
    """
    row, col = get_channel_from_grid_pos(pos=coord) if is_grid else coord
    if flip_y:
        row = 15 - row  # NOTE assumes that channel width is 16
    for ax in axes:
        plot_pixel_dot(
            ax, row=row, col=col, color=color, hidden_padding=hidden_padding
        )


def get_channel_from_grid_pos(
    pos: Tuple[int, int], layer: str = default_layer
):
    """Given a grid position, find the channel location that corresponds to that position.
    """
    # Ensure cheese_pos is valid
    row, col = pos
    assert (
        0 <= row < maze.WORLD_DIM and 0 <= col < maze.WORLD_DIM
    ), f"Invalid position: {pos}"

    # Convert to pixel location
    px_row, px_col = (
        (row + 0.5) * maze.AGENT_PX_PER_TILE,
        (col + 0.5) * maze.AGENT_PX_PER_TILE,
    )

    px_per_channel_idx = get_stride(layer)  # How many pixels per channel index
    chan_row, chan_col = (
        px_row // px_per_channel_idx,
        px_col // px_per_channel_idx,
    )
    return (int(chan_row), int(chan_col))


def pixels_at_grid(
    row: int,
    col: int,
    img: np.ndarray,
    removed_padding: int = 0,
    flip_y: bool = True,
):
    """Get the pixels in the image corresponding to the given grid position.

    Args:
        row: The row of the grid position.
        col: The column of the grid position.
        img: The image to get the pixels from, assumed to be rendered from the human view.
        removed_padding: The number of tiles which are not shown in the human view, presumably due to render_padding being False in some external call.
    """
    assert (
        0 <= row < maze.WORLD_DIM and 0 <= col < maze.WORLD_DIM
    ), f"Invalid position: {row, col}"
    assert (
        img.shape[2] == 3
    ), (  # Ensure image is RGB
        f"Image must have 3 channels, but has {img.shape[2]}"
    )
    assert (
        maze.WORLD_DIM // 2 > removed_padding >= 0
    ), f"removed_padding must be non-negative, but is {removed_padding}"

    if flip_y:
        row = (maze.WORLD_DIM - 1) - row
    row, col = row - removed_padding, col - removed_padding

    row_lb, row_ub = (
        row * maze.HUMAN_PX_PER_TILE,
        (row + 1) * maze.HUMAN_PX_PER_TILE,
    )
    col_lb, col_ub = (
        col * maze.HUMAN_PX_PER_TILE,
        (col + 1) * maze.HUMAN_PX_PER_TILE,
    )

    # add 12 to the bounds to account for the 6 pixel border, and cast as ints
    # FIXME 512x512 is only for full level; smaller levels are different (?!) Thus subtracting 12 can lead to row_ub being too large for img, leading to a ValueError from shape mismatch in parent.
    row_lb, row_ub = (
        int(coord + maze.HUMAN_PX_PADDING * 2) for coord in (row_lb, row_ub)
    )
    col_lb, col_ub = (
        math.floor(coord + 1) for coord in (col_lb, col_ub)
    )  # FIXME e.g. seed 19 still has a few pixels off

    return img[row_lb:row_ub, col_lb:col_ub, :]


def visualize_venv(
    venv: ProcgenGym3Env,
    idx: int = 0,
    mode: str = "human",
    ax: plt.Axes = None,
    ax_size: int = 3,
    show_plot: bool = False,
    flip_numpy: bool = True,
    render_padding: bool = True,
    render_mouse: bool = True,
):
    """Visualize the environment. Returns an img if show_plot is false.

    Parameters:
    venv: The environment to visualize
    idx: The index of the environment to visualize, in the vectorized environment.
    mode: The mode to visualize in. Can be "human", "agent", or "numpy"
    ax: The axis to plot on. If None, a new axis will be created.
    ax_size: The size of the axis to create, if ax is None.
    show_plot: Whether to show the plot.
    flip_numpy: Whether to vertically flip the numpy view.
    render_padding: Whether to render the padding in the human or numpy view.
    render_mouse: Whether to render the mouse in the human view.
    """
    assert not (mode == "agent" and not render_padding), (
        "This parameter combination is unsupported; must render padding in"
        " agent mode."
    )
    if not render_mouse:
        assert mode == "human", "render_mouse is only supported in human mode."

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(ax_size, ax_size))

    env_state = maze.state_from_venv(venv, idx=idx)
    full_grid = env_state.full_grid()
    inner_grid = env_state.inner_grid()

    if mode == "human":
        if render_padding:
            img = venv.env.get_info()[idx]["rgb"]
        else:
            img = maze.render_inner_grid(inner_grid)
        if not render_mouse:
            # First get the mouse position and the corresponding pixels
            mouse_pos = maze.get_mouse_pos(grid=full_grid)
            pad = 0 if render_padding else maze.get_padding(grid=inner_grid)
            mouse_px = pixels_at_grid(*mouse_pos, img=img, removed_padding=pad)

            # Now get an empty position and the corresponding pixels
            empty_pos = maze.get_object_pos_in_grid(full_grid, maze.EMPTY)
            empty_px = pixels_at_grid(*empty_pos, img=img, removed_padding=pad)

            # Now replace the mouse pixels with the empty pixels
            mouse_px[:] = empty_px
    elif mode == "agent":
        img = venv.reset()[idx].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    elif mode == "numpy":
        rendered_grid = full_grid if render_padding else inner_grid
        img = rendered_grid[
            :: (-1 if flip_numpy else 1), :
        ]  # Flip the numpy view vertically
    else:
        raise ValueError(f"Invalid mode {mode}")

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(img)
    if show_plot:
        plt.show()
    return img


def custom_vfield(
    policy: t.nn.Module,
    venv: ProcgenGym3Env = None,
    seed: int = 0,
    ax_size: int = None,
    callback: Callable = None,
    show_vfield: bool = True,
    show_components: bool = False,
    show_full: bool = False,
):
    """Given a policy and a maze seed, create a maze editor and a vector field plot. Update the vector field whenever the maze is edited. Returns a VBox containing the maze editor and the vector field plot.

    Args:
        policy: The policy to use to compute the vector field.
        venv: The environment to use to compute the vector field. If None, a new environment will be created.
        seed: The seed to use to create the environment.
        ax_size: The size of the vector field plot.
        callback: A callback to call whenever the maze is edited.
        show_vfield: Whether to show the vector field plot.
        show_components: Whether to show the vectors for each action.
        show_full: Whether to edit the full level with padding included.
    """
    output = Output()
    if venv is None:
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)

    # Dynamically compute ax_size if not provided
    if ax_size is None:
        grid = (
            maze.state_from_venv(venv, idx=0).full_grid()
            if show_full
            else maze.state_from_venv(venv, idx=0).inner_grid()
        )
        grid_size = grid.shape[0]
        ax_size = 4 * grid_size / 16

    fig, ax = plt.subplots(1, 1, figsize=(ax_size, ax_size))
    plt.close("all")

    # We want to update ax whenever the maze is edited
    def update_plot():
        # Clear the existing plot
        with output:
            ax.clear()
            if show_vfield:
                vf = vector_field(venv, policy)
                plot_vf(vf, ax=ax, show_components=show_components)
            else:
                visualize_venv(
                    venv,
                    ax=ax,
                    ax_size=ax_size,
                    show_plot=False,
                    render_padding=show_full,
                )

            # Update the existing figure in place
            clear_output(wait=True)
            display(fig)

    update_plot()

    def cb(gridm):  # Callback for when the maze is edited
        if callback is not None:
            callback(gridm)
        update_plot()

    # Then make a callback which updates the render in-place when the maze is edited
    editors = maze.venv_editors(
        venv, check_on_dist=False, callback=cb, show_full=show_full
    )  # NOTE can't block off initial mouse position?
    # Set the editors so that they don't space out when the window is resized
    for editor in editors:
        editor.layout.height = "100%"

    widget_vbox = Box(
        children=editors + [output],
        layout=Layout(
            display="flex",
            flex_flow="row",
            align_items="stretch",
            width="100%",
        ),
    )

    return widget_vbox


def custom_vfields(policy: t.nn.Module, venv: ProcgenGym3Env, **kwargs):
    """Create a vector field plot for each maze in the environment, using policy to generate the vfields.
    """
    venvs = [maze.copy_venv(venv, idx=i) for i in range(venv.num_envs)]
    return VBox(
        [custom_vfield(venv=venv, policy=policy, **kwargs) for venv in venvs]
    )


### activation management
def get_activations(obs: np.ndarray, hook: cmh.ModuleHook, layer_name: str):
    hook.run_with_input(obs)  # run the model with the given obs
    return hook.get_value_by_label(
        layer_name
    )  # shape is (b, c, h, w) at conv layers, (b, activations) at linear layers


### Plotters
def plot_activations(activations: np.ndarray, fig: go.FigureWidget):
    """Plot the activations given a single (non-batched) activation tensor."""
    fig.update(data=[go.Heatmap(z=activations)])


def plot_nonzero_activations(activations: np.ndarray, fig: go.FigureWidget):
    """Plot the nonzero activations in a heatmap."""
    # Find nonzero activations and cast to floats
    nz = (activations != 0).astype(np.float32)
    fig.update(data=[go.Heatmap(z=nz)])


def plot_nonzero_diffs(activations: np.ndarray, fig: go.FigureWidget):
    """Plot the nonzero activation diffs in a heatmap."""
    diffs = activations[0] - activations[1]
    plot_nonzero_activations(diffs, fig)


NUM_ACTIONS = 15


def format_plotter(
    fig: go.Figure,
    activations: np.ndarray,
    title: str = None,
    is_policy_out: bool = False,
    bounds: Tuple[int, int] = None,
    px_dims: Tuple[int, int] = None,
):
    """Format the figure. Takes activations as input so that the x- and z-axes can be formatted according to the activations.

    If is_policy_out is True, the x-axis will be formatted as a policy output. If bounds is not None, the z-axis will be formatted to show the activations in the given bounds.
    """
    h, w = px_dims or (500, 500)
    fig.update_layout(height=h, width=w, title_text=title)
    fig.update_xaxes(side="top")

    if is_policy_out:
        # Transform each index into the corresponding action label, according to maze.py
        fig.update_xaxes(
            ticktext=[
                models.human_readable_action(i).title()
                for i in range(NUM_ACTIONS)
            ],
            tickvals=np.arange(activations.shape[3]),
        )
    else:  # Reset the x-axis ticks to match the y-axis ticks
        yaxis_ticktext, yaxis_tickvals = (
            fig.layout.yaxis.ticktext,
            fig.layout.yaxis.tickvals,
        )  # TODO double the step of yaxis ticks?
        fig.update_xaxes(ticktext=yaxis_ticktext, tickvals=yaxis_tickvals)

    fig.update_xaxes(side="top")  # Set the x ticks to the top
    fig.update_yaxes(autorange="reversed")  # Reverse the row-axis autorange

    # Set the min and max to be the min and max of all channels at this label
    if bounds is None:
        max_act = np.abs(activations).max()
        bounds = (-1 * max_act, max_act)
    fig.update_traces(zmin=bounds[0], zmid=0, zmax=bounds[1])

    # Change the colorscale to split red (negative) -- white (zero) -- blue (positive)
    fig.update_traces(colorscale="RdBu")


# To indicate that fig can be matplotlib or plotly, we use the type go.FigureWidget


def plot_patch(
    patch: dict,
    hook: cmh.ModuleHook,
    layer: str = default_layer,
    channel: int = 0,
    fig: go.FigureWidget = None,
    title: str = None,
    bounds: Tuple[int, int] = None,
    px_dims: Tuple[int, int] = None,
):
    """Plot the activations of a single patch, at the given layer and channel. Returns a figure.
    """
    assert layer in patch, f"Layer {layer} not in patch {patch}"

    if fig is None:
        fig = go.FigureWidget()
    activations = patch[layer](hook.values_by_label[default_layer])
    assert (
        activations.shape[1] > channel
    ), f"Channel {channel} not in activations {activations.shape}"

    activations = activations[0, channel]  # Shape is (h, w)
    plot_activations(activations, fig)
    format_plotter(
        fig, activations, title=title, bounds=bounds, px_dims=px_dims
    )
    return fig


# Widget helpers
def create_save_button(
    prefix: str, fig: plt.Figure, descriptors: Dict[str, float]
):
    """Create a button that saves fig to a file.

    Args:
        prefix (str): The prefix of the filename. Typically "experiments/visualizations/" or "playground/visualizations/".
        fig (plt.Figure): The figure to save.
        descriptors (defaultdict[str, float]): A dictionary of descriptors to add to the filename.
    """

    def save_fig(b):
        """Save the figure to a file."""
        filename = prefix
        for key, value in descriptors.items():
            # Replace any dots with underscores
            value = str(value).replace(".", "_")
            filename += f"{key}_{value}_"
        filename = filename[:-1] + ".png"  # remove trailing underscore
        fig.savefig(filename)
        # Display the filename
        display(Markdown(f"Figure saved to `{filename}`"))

    button = Button(description="Save figure")
    button.on_click(save_fig)
    return button  # TODO integrate with activationsPlotter


class ActivationsPlotter:
    def __init__(
        self,
        labels: List[str],
        plotter: Callable,
        activ_gen: Callable,
        hook,
        coords_enabled: bool = False,
        defaults: dict = None,
        save_dir="experiments/visualizations",
        **act_kwargs,
    ):
        """
        labels: The labels of the layers to plot
        plotter: A function that takes a label, channel, and activations and plots them
        activ_gen: A function that takes a label and obs and returns the activations which should be sent to plotter
        hook: The hook that contains the activations
        coords_enabled: Whether to enable the row and column sliders
        defaults: A dictionary of default values for the plotter, where the keys are attributes of this class which themselves have the "value" attribute. The class value will be set to the corresponding dictionary value.
        act_kwargs: Keyword arguments to pass to the activations generator
        """
        self.fig = go.FigureWidget()
        self.plotter = plotter
        self.activ_gen = activ_gen
        self.act_kwargs = act_kwargs
        self.hook = hook
        self.save_dir = save_dir

        # Remove the _out layer and "embedder." prefixes
        formatted_labels = format_labels(labels)
        self.label_widget = Dropdown(
            options=formatted_labels,
            value=formatted_labels[0],
            description="Layers",
        )
        self.channel_slider = IntSlider(
            min=0, max=127, step=1, value=0, description="Channel"
        )

        # Add channel increment and decrement buttons
        button_width = "10px"
        decrement_button, increment_button = [
            Button(description=descr_str, layout=Layout(width=button_width))
            for descr_str in ("-", "+")
        ]

        def add_to_slider(x: int):
            # Clip the value to the min and max
            self.channel_slider.value = np.clip(
                self.channel_slider.value + x,
                self.channel_slider.min,
                self.channel_slider.max,
            )
            self.update_plotter()

        decrement_button.on_click(lambda _: add_to_slider(-1))
        increment_button.on_click(lambda _: add_to_slider(1))
        self.widgets = [
            self.fig,
            self.label_widget,
            HBox([self.channel_slider, decrement_button, increment_button]),
        ]  # TODO make this a helper for converting arbitrary sliders

        # Add row and column sliders if enabled
        self.coords_enabled = coords_enabled
        if coords_enabled:
            self.col_slider, self.row_slider = (
                IntSlider(
                    min=0, max=62, step=1, value=32, description="Column"
                ),
                IntSlider(min=0, max=63, step=1, value=32, description="Row"),
            )
            self.widgets.extend([self.col_slider, self.row_slider])

        # Add a custom filename widget
        self.filename_widget = Text(
            value="", placeholder="Custom filename", disabled=False
        )
        self.filename_widget.layout.width = "150px"
        self.button = Button(description="Save image")
        self.button.on_click(self.save_image)
        self.widgets.append(HBox([self.filename_widget, self.button]))

        # Set the default values for the plotter, if provided
        if defaults is not None:
            for key, value in defaults.items():
                getattr(self, key).value = value

        # Ensure that the plot is updated when the widgets are changed
        for widget in self.widgets:
            if widget != self.fig:
                widget.observe(self.update_plotter, names="value")

        # Set the initial plot
        self.update_plotter()

    def display(self):
        """Display the elements; this function separates functionality from display.
        """
        display(self.fig)
        display(
            VBox(self.widgets[1:-1])
        )  # Show a VBox of the label dropdown and the sliders, centered beneath the plot
        display(self.widgets[-1])

    def save_image(self, b):  # Add a save button to save the image
        basename = (
            self.filename_widget.value
            if self.filename_widget.value != ""
            else (
                f"{self.label_widget.value}_{self.channel_slider.value}{f'_{self.col_slider.value}_{self.row_slider.value}' if self.coords_enabled else ''}"
            )
        )
        filepath = f"{self.save_dir}/{basename}.png"

        # Annotate to the outside of the plot
        old_title = self.fig.layout.title
        self.fig.layout.title = (
            f"{self.label_widget.value};\nchannel"
            f" {self.channel_slider.value}{f' at ({self.col_slider.value}, {self.row_slider.value})' if self.coords_enabled else ''}"
        )

        self.fig.update_yaxes(autorange="reversed")
        self.fig.write_image(filepath)
        print(f"Saved image to {filepath}")

        self.fig.layout.title = old_title  # Clear the title

        self.filename_widget.value = ""  # Clear the filename_widget box

    def update_plotter(self, b=None):
        """Update the plot with the current values of the widgets."""
        label = expand_label(self.label_widget.value)
        self.channel_slider.max = (
            models.num_channels(hook=self.hook, layer_name=label) - 1
        )
        channel = self.channel_slider.value = min(
            self.channel_slider.value, self.channel_slider.max
        )

        if self.coords_enabled:
            col, row = self.col_slider.value, self.row_slider.value
            activations = self.activ_gen(
                row, col, label, self.hook, **self.act_kwargs
            )
        else:
            activations = self.activ_gen(
                label, self.hook, **self.act_kwargs
            )  # shape is (b, c, h, w) at conv layers, (b, activations) at linear layers

        if len(activations.shape) == 2:  # Linear layer (batch, hidden_dim)
            # Ensure shape[1] is a perfect square
            sqrt_act = int(math.sqrt(activations.shape[1]))
            if sqrt_act * sqrt_act == activations.shape[1]:
                activations = np.reshape(
                    activations,
                    newshape=(activations.shape[0], 1, sqrt_act, sqrt_act),
                )  # Make a dummy channel dimension
                # Annotate that there is no spatial meaning to the activations
                self.fig.update_layout(
                    title_text=(
                        f"{self.label_widget.value}; reshaped to 2D; no"
                        " spatial meaning"
                    )
                )
            else:
                activations = np.expand_dims(
                    activations, axis=(1, 2)
                )  # Add a dummy dimension to the activations

        self.plotter(
            activations=activations[:, channel], fig=self.fig
        )  # Plot the activations
        format_plotter(
            fig=self.fig,
            activations=activations,
            is_policy_out=self.label_widget.value == "fc_policy_out",
            title=self.label_widget.value,
        )


# ============================= The following used to be vfields.py =============================


def render_arrows(
    vf: dict,
    ax=None,
    human_render: bool = True,
    render_padding: bool = False,
    color: str = "white",
    show_components: bool = False,
):
    """Render the arrows in the vector field.

    args:
        vf: The vector field dict
        ax: The matplotlib axis to render on
        human_render: Whether to render the grid in a human-readable way (high-res pixel view) or a machine-readable way (grid view)
        render_padding: Whether to render the padding around the grid
        color: The color of the arrows
        show_components: Whether to show one arrow for each cardinal action. If False, show one arrow for each action.
    """
    ax = ax or plt.gca()

    arrows, legal_mouse_positions, grid = (
        vf["arrows"],
        vf["legal_mouse_positions"],
        vf["grid"],
    )

    inner_size = grid.shape[0]  # The size of the inner grid
    arrow_rescale = maze.WORLD_DIM / (
        inner_size * 1.8
    )  # Rescale arrow width and other properties to be relative to the size of the maze
    width = 0.005 * arrow_rescale
    if show_components:
        # A list of length-four lists of (x, y) tuples, one for each mouse position
        for idx, tile_arrows in enumerate(arrows):
            ax.quiver(
                [legal_mouse_positions[idx][1]] * len(tile_arrows),
                [legal_mouse_positions[idx][0]] * len(tile_arrows),
                [arr[1] for arr in tile_arrows],
                [arr[0] for arr in tile_arrows],
                color=color,
                scale=1,
                scale_units="xy",
                width=width,
            )

    else:
        arrows = [
            _tadd(*arr_list) for arr_list in arrows
        ]  # Add the arrows together to get a total vector for each mouse position
        ax.quiver(
            [pos[1] for pos in legal_mouse_positions],
            [pos[0] for pos in legal_mouse_positions],
            [arr[1] for arr in arrows],
            [arr[0] for arr in arrows],
            color=color,
            scale=1,
            scale_units="xy",
            width=width,
        )

    venv = maze.venv_from_grid(grid)
    visualize_venv(
        venv,
        ax=ax,
        mode="human" if human_render else "numpy",
        render_padding=render_padding,
        render_mouse=False,
        show_plot=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])


def forward_func_policy(network: nn.Module, inp: torch.Tensor):
    """Forward function for the policy network."""
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)


# %%
# Get vector field

# FIXME really stupid way to do this tbh, should use numpy somehow
def _tmul(tup: tuple, scalar: float):
    """Multiply a tuple by a scalar."""
    return tuple(scalar * x for x in tup)


def _tadd(*tups: List[Tuple[int, int]]):
    """Add a list of tuples elementwise."""
    return tuple(sum(axis) for axis in zip(*tups))


def _device(policy: nn.Module):
    return next(policy.parameters()).device


def vector_field(venv: ProcgenGym3Env, policy: nn.Module, idx: int = 0):
    """
    Get the vector field induced by the policy on the maze in the idx-th environment.
    """
    venv_idx = maze.copy_venv(venv, idx)
    venv_all, (
        legal_mouse_positions,
        grid,
    ) = maze.venv_with_all_mouse_positions(venv_idx)
    return vector_field_tup(venv_all, legal_mouse_positions, grid, policy)


def get_arrows_and_probs(
    legal_mouse_positions: List[Tuple[int, int]], c_probs: torch.Tensor
) -> List[dict]:
    """Get the arrows and probabilities for each mouse position.

    Args:
        legal_mouse_positions: A list of (x, y) tuples, each assumed to be an outer grid coordinate.
        c_probs: A tensor of shape (len(legal_mouse_positions), 15) of post-softmax probabilities, one for each mouse position.

    Returns:
        action_arrows: A list of lists of probability-weighted basis vectors -- an (x, y) tuple, one for each mouse position
        probs: A list of dicts of action -> probability, one for each mouse position
    """
    # FIXME: Vectorize this loop. It isn't as critical as the model though
    action_arrows, probs = [], []
    for i in range(len(legal_mouse_positions)):
        # Dict of action -> probability for this mouse position
        probs_dict = models.human_readable_actions(c_probs[i])
        # Convert to floats
        probs_dict = {k: v.item() for k, v in probs_dict.items()}

        # Multiply each basis vector by the probability of that action, and append this list of action component arrows
        action_arrows.append(
            [
                _tmul(models.MAZE_ACTION_DELTAS[act], p)
                for act, p in probs_dict.items()
            ]
        )
        # Append the {(action : str): (probability : float)} dict
        probs.append(tuple(probs_dict.values()))

    return action_arrows, probs


def vector_field_tup(
    venv_all: ProcgenGym3Env,
    legal_mouse_positions: List[Tuple[int, int]],
    grid: np.ndarray,
    policy: nn.Module,
):
    """
    Plot the vector field induced by the policy on the maze in venv env number 1.

    Args:
        venv_all: The venv to use to get the grid and legal mouse positions. Deleted after use.
        legal_mouse_positions: a list of (x, y) tuples, each assumed to be an outer grid coordinate.
        grid: The outer grid to use to compute the vector field.
        policy: The policy to use to compute the vector field.
    """
    # TODO: Hypothetically, this step could run in parallel to the others (cpu vs. gpu)
    batched_obs = torch.tensor(
        venv_all.reset(), dtype=torch.float32, device=_device(policy)
    )
    del venv_all

    # use stacked obs list as a tensor
    with torch.no_grad():
        categorical, _ = policy(batched_obs)

    action_arrows, probs = get_arrows_and_probs(
        legal_mouse_positions, categorical.probs
    )

    # make vfield object for returning
    return {
        "arrows": action_arrows,
        "legal_mouse_positions": legal_mouse_positions,
        "grid": grid,
        "probs": probs,
    }


# %%
# Plot vector field for every mouse position
def plot_vector_field(
    venv: ProcgenGym3Env,
    policy: nn.Module,
    ax: plt.Axes = None,
    env_num: int = 0,
):
    """
    Plot the vector field induced by the policy on the maze in venv env number i.
    """
    warn("Deprecated in favor of calling vector_field and plot_vf directly.")
    venv = maze.copy_venv(venv, env_num)
    vf = vector_field(venv, policy)
    return plot_vf(vf, ax=ax)


def map_vf_to_human(vf: dict, account_for_padding: bool = False):
    """Map the vector field vf to the human view coordinate system.

    Args:
        vf: A vector field dict with the maze coordinate system.
        account_for_padding: Whether to account for the padding in the human view coordinate system.

    Returns:
        vf: A vector field dict with the human view coordinate system.
    """
    legal_mouse_positions, arrows, grid = (
        vf["legal_mouse_positions"],
        vf["arrows"],
        vf["grid"],
    )

    # We need to transform the arrows to the human view coordinate system
    human_view = maze.render_outer_grid(grid)

    padding = maze.WORLD_DIM - grid.shape[0]
    assert padding % 2 == 0
    padding //= 2
    rescale = human_view.shape[0] / maze.WORLD_DIM

    legal_mouse_positions = [
        ((grid.shape[1] - 1) - row, col) for row, col in legal_mouse_positions
    ]  # flip y axis
    if account_for_padding:
        legal_mouse_positions = [
            (row + padding, col + padding)
            for row, col in legal_mouse_positions
        ]
    legal_mouse_positions = [
        ((row + 0.5) * rescale, (col + 0.5) * rescale)
        for row, col in legal_mouse_positions
    ]
    arrows = [[_tmul(arr, rescale) for arr in arr_list] for arr_list in arrows]

    return {
        "arrows": arrows,
        "legal_mouse_positions": legal_mouse_positions,
        "grid": grid,
    }


def plot_vf(
    vf: dict,
    ax=None,
    human_render: bool = True,
    render_padding: bool = False,
    show_components: bool = False,
):
    "Plot the vector field given by vf. If human_render is true, plot the human view instead of the raw grid np.ndarray."
    render_arrows(
        map_vf_to_human(vf, account_for_padding=render_padding)
        if human_render
        else vf,
        ax=ax,
        human_render=human_render,
        render_padding=render_padding,
        color="white" if human_render else "red",
        show_components=show_components,
    )


def get_vf_diff(vf1: dict, vf2: dict):
    """Get the difference "vf1 - vf2" between two vector fields."""

    def assert_compatibility(vfa, vfb):
        assert vfa["legal_mouse_positions"] == vfb["legal_mouse_positions"], (
            "Legal mouse positions must be the same to render the vf"
            " difference."
        )
        assert (
            vfa["grid"].shape == vfb["grid"].shape
        ), "Grids must be the same shape to render the vf difference."
        assert len(vfa["arrows"]) == len(
            vfb["arrows"]
        ), "Arrows must be the same length to render the vf difference."

    # Remove cheese from the legal mouse positions and arrows, if levels are otherwise the same
    for i in range(2):
        try:
            assert_compatibility(vf1, vf2)
        except:
            if (vf1["grid"] == maze.CHEESE).any():
                cheese_vf_idx = 0
            elif (vf2["grid"] == maze.CHEESE).any():
                cheese_vf_idx = 1
            else:
                raise ValueError(
                    "Levels are not the same, but neither has cheese."
                )

            vfs = [vf1, vf2]
            cheese_location = maze.get_cheese_pos(vfs[cheese_vf_idx]["grid"])

            # Remove cheese from the legal mouse positions and arrows
            other_vf_idx = 1 - cheese_vf_idx
            vfs[other_vf_idx]["arrows"] = [
                arr
                for pos, arr in zip(
                    vfs[other_vf_idx]["legal_mouse_positions"],
                    vfs[other_vf_idx]["arrows"],
                )
                if pos != cheese_location
            ]
            vfs[other_vf_idx]["legal_mouse_positions"] = [
                pos
                for pos in vfs[other_vf_idx]["legal_mouse_positions"]
                if pos != cheese_location
            ]

    arrow_diffs = [
        [(a1[0] - a2[0], a1[1] - a2[1]) for a1, a2 in zip(arrs1, arrs2)]
        for arrs1, arrs2 in zip(vf1["arrows"], vf2["arrows"])
    ]

    return {
        "arrows": arrow_diffs,
        "legal_mouse_positions": vf2["legal_mouse_positions"],
        "grid": vf2["grid"],
    }


def vf_diff_magnitude(vf_diff: dict) -> float:
    """Compute the average magnitude of the vector field difference."""
    return np.linalg.norm(vf_diff["arrows"]) / len(vf_diff["arrows"])


def vf_diff_magnitude_from_seed(seed: int, patches: dict):
    """Return average per-location probability change due to the given patches.
    """
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    vf1 = vector_field(venv, policy)
    with hook.use_patches(patches):
        vf2 = vector_field(venv, hook.network)
    vf_diff = get_vf_diff(vf1, vf2)

    # Average the vector diff magnitude over grid locations
    avg_diff = vf_diff_magnitude(vf_diff)
    # Compute TV distance so divide by 2, otherwise double-counting
    # probability shifts
    return avg_diff / 2


def plot_vf_diff(
    vf1: dict,
    vf2: dict,
    ax: plt.Axes = None,
    human_render: bool = True,
    render_padding: bool = False,
    show_components: bool = False,
):
    """Render the difference "vf1 - vf2" between two vector fields, plotting only the difference.
    """
    # Remove cheese from the legal mouse positions and arrows, if levels are otherwise the same
    vf_diff = get_vf_diff(vf1, vf2)

    render_arrows(
        map_vf_to_human(vf_diff, account_for_padding=render_padding)
        if human_render
        else vf_diff,
        ax=ax,
        human_render=human_render,
        render_padding=render_padding,
        color="lime" if human_render else "red",
        show_components=show_components,
    )

    return vf_diff


def plot_vfs(
    vf1: dict,
    vf2: dict,
    human_render: bool = True,
    render_padding: bool = False,
    ax_size: int = 5,
    show_diff: bool = True,
    show_original: bool = True,
    show_components: bool = False,
):
    """Plot two vector fields and, if show_diff is True, their difference vf2 - vf1. Plots three axes in total. Returns the figure, axes, and the difference vector field. If show_original is False, don't plot the original vector field.
    """
    num_cols = 1 + show_diff + show_original
    fontsize = 16
    fig, axs = plt.subplots(1, num_cols, figsize=(ax_size * num_cols, ax_size))

    idx = 0
    if show_original:
        plot_vf(
            vf1,
            ax=axs[idx],
            human_render=human_render,
            render_padding=render_padding,
            show_components=show_components,
        )
        axs[idx].set_xlabel("Original", fontsize=fontsize)
        idx += 1

    plot_vf(
        vf2,
        ax=axs[idx],
        human_render=human_render,
        render_padding=render_padding,
        show_components=show_components,
    )
    axs[idx].set_xlabel("Patched", fontsize=fontsize)
    idx += 1

    if show_diff:
        # Pass in vf2 first so that the difference is vf2 - vf1, or the difference between the patched and original vector fields
        vf_diff = plot_vf_diff(
            vf2,
            vf1,
            ax=axs[idx],
            human_render=human_render,
            render_padding=render_padding,
            show_components=show_components,
        )
        axs[idx].set_xlabel("Patched vfield minus original", fontsize=fontsize)
    return fig, axs, (vf_diff if show_diff else None)
