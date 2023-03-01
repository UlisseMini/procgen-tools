from procgen_tools.imports import *
from procgen_tools import maze, vfield, patch_utils
from typing import Dict


# LABEL HANDLING
def format_label(label : str):
    """Format a label for display in the """
    return label.replace("embedder.", "")

def expand_label(label : str):
    if not (label.startswith("fc_policy") or label.startswith("fc_value")):
        return "embedder." + label
    else:
        return label

def format_labels(labels : List[str]):
    """Format labels for display in the """
    return list(map(format_label, labels))

AGENT_OBS_WIDTH = 64
def get_impala_num(label : str):
    """Get the block number of a layer."""
    if not label.startswith("embedder.block"): raise ValueError(f"Not in the Impala blocks.")

    # The labels are formatted as embedder.block{blocknum}.{residual_block_num}
    return int(label.split(".")[1][-1])

def get_residual_num(label : str):
    """Get the residual block number of a layer."""
    if not label.startswith("embedder.block"): raise ValueError(f"Not in the Impala blocks.")

    # The labels are formatted as embedder.block{blocknum}.{residual_block_num}
    return int(label.split(".")[2][-1])

# Plotting
def is_internal_activation(label : str):
    """ Return True if the label is an internal activation, i.e. not an input or output. """
    # Check if 'in' is in the label
    if 'in' in label:
        return False
    # Check if 'out' is in the label
    if 'out' in label and 'embedder' not in label:
        return False
    return True

def plot_layer_stats(hook : cmh.ModuleHook, mode : str = "activations", fig : go.Figure = None):
    """ Create and show a plotly bar chart of the number of activations per layer of policy. The default mode is "activations", which plots the number of activations per layer. If mode is "parameters", then the number of parameters per layer is plotted. """
    if mode not in ("activations", "parameters"):
        raise ValueError(f"mode must be either 'activations' or 'parameters', got {mode}.")

    # Get the number of activations/parameters per layer
    quantities = {}
    zipped_list = filter(lambda tup: is_internal_activation(tup[0]), hook.values_by_label.items()) if mode == 'activations' else hook.network.named_parameters()
    for name, quantity in zipped_list:
        quantities[format_label(name)] = quantity.numel()
    total_quantity = sum(quantities.values())

    # Aggregate bias quantities
    if mode == 'parameters':
        bias_quants = {label: quantity for label, quantity in quantities.items() if 'bias' in label}
        for label, quantity in bias_quants.items():
            if 'bias' in label: # If the label is a bias, add the quantity to the corresponding weight
                quantities[label.replace('bias', 'weight')] += quantity
                del quantities[label]

    key_list = [key.replace('.weight', '') if mode == 'parameters' else key for key in quantities.keys()][::-1]
    values_list = list(quantities.values())[::-1]
    fig = go.Figure(data=[go.Bar(y=key_list, x=values_list, orientation='h')])

    # Format the total number of quantity to be e.g. 365M
    formatted_total = format(total_quantity, ',')

    # Set layout
    fig.update_layout(
        title={ 'text': f'Model {mode.title()} Per Layer (Total: {formatted_total})', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, 
        xaxis_title=mode.title() if mode == "parameters" else "Activation count", 
        yaxis_title="Layer" if mode == "parameters" else "Activations",
        )

    # Ensure there's enough height to see all the x labels
    fig.update_layout(height=max(500, 20 * len(quantities)))

    # Set mouse-over text to show the number of quantity
    fig.update_traces(text=values_list, hovertemplate='%{text:,}' + f' {mode}<extra></extra>', hoverlabel=dict(bgcolor='white'), )

    # Set x axis to be logarithmic
    fig.update_xaxes(type="log")

    fig.show()

# Navigating the feature maps
def get_stride(label : str):
    """Get the stride of the layer referred to by label. How many pixels required to translate a single entry in the feature maps of label. """
    if not label.startswith("embedder.block"): raise ValueError(f"Not in the Impala blocks.")

    block_num = get_impala_num(label)
    if 'conv_out' in label: # Before that Impala layer's maxpool has been applied
        block_num -= 1
    return 2 ** block_num

dummy_venv = patch_utils.get_cheese_venv_pair(seed=0) 
human_view = dummy_venv.env.get_info()[0]['rgb']
PIXEL_SIZE = human_view.shape[0] # width of the human view input image

def get_pixel_loc(val : int, channel_size : int = 16):
    """ Given a single channel position value, find the pixel location that corresponds to that channel. """
    assert val < channel_size, f"channel_pos {val} must be less than channel_size {channel_size}"
    assert val >= 0, f"channel_pos {val} must be non-negative"

    scale = PIXEL_SIZE / channel_size
    return int(scale * (val + .5))

def get_pixel_coords(channel_pos : Tuple[int, int], channel_size : int = 16, flip_y : bool = True):
    """ Given a channel position, find the pixel location that corresponds to that channel. If flip_y is True, the y-axis will be flipped from the underlying numpy coords to the conventional human rendering format. """
    row, col = channel_pos
    assert 0 <= row < channel_size and 0 <= col < channel_size, f"channel_pos {channel_pos} must be within the channel_size {channel_size}."

    if flip_y: 
        row = (channel_size - 1) - row

    return get_pixel_loc(row, channel_size), get_pixel_loc(col, channel_size)

def plot_pixel_dot(ax : plt.Axes, row : int, col : int, color : str = 'r', size : int = 50, hidden_padding : int = 0):
    """ Plot a dot on the pixel grid at the given row and column of the block2.res1.resadd_out channel. hidden_padding is the number of tiles which are not shown in the human view, presumably due to render_padding being False in some external call. """
    px_row, px_col = get_pixel_coords((row, col))
    padding_offset = (PIXEL_SIZE / maze.WORLD_DIM) * hidden_padding
    dot_rescale_from_padding =  maze.WORLD_DIM / (maze.WORLD_DIM - hidden_padding)
    
    # Ensure the dot is within the bounds of the pixel grid
    new_row, new_col = (coord - padding_offset for coord in (px_row, px_col))  
    if 0 <= new_row <= PIXEL_SIZE and 0 <= new_col <= PIXEL_SIZE:
        ax.scatter(y=new_row, x=new_col, c=color, s=size * dot_rescale_from_padding)

def plot_dots(axes : List[plt.Axes], coord : Tuple[int, int], color : str = 'red', flip_y : bool = True, is_grid : bool = False, hidden_padding : int = 0):
    """ Plot dots on the given axes, given a channel coordinate in the full grid. If flip_y, flips the y-axis. If is_grid, assumes the coord is an outer grid coordinate. 
    
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
        row = 15 - row # NOTE assumes that channel width is 16
    for ax in axes:
        plot_pixel_dot(ax, row=row, col=col, color=color, hidden_padding = hidden_padding)

def get_channel_from_grid_pos(pos : Tuple[ int, int ], layer : str = default_layer):
    """ Given a grid position, find the channel location that corresponds to that position. """
    # Ensure cheese_pos is valid
    row, col = pos
    assert 0 <= row < maze.WORLD_DIM and 0 <= col < maze.WORLD_DIM, f'Invalid position: {pos}'

    # Convert to pixel location
    px_row, px_col = ((row + .5) * maze.PX_PER_TILE, (col + .5) * maze.PX_PER_TILE)

    px_per_channel_idx = get_stride(layer) # How many pixels per channel index
    chan_row, chan_col = (px_row // px_per_channel_idx, px_col // px_per_channel_idx)
    return (int(chan_row), int(chan_col))

def visualize_venv(venv : ProcgenGym3Env, idx : int = 0, mode : str="human", ax : plt.Axes = None, ax_size : int = 3, show_plot : bool = True, flip_numpy : bool = True, render_padding : bool = True):
    """ Visualize the environment. Returns an img if show_plot is false. 
    
    Parameters: 
    venv: The environment to visualize
    idx: The index of the environment to visualize, in the vectorized environment.
    mode: The mode to visualize in. Can be "human", "agent", or "numpy"
    ax: The axis to plot on. If None, a new axis will be created.
    ax_size: The size of the axis to create, if ax is None.
    show_plot: Whether to show the plot. 
    flip_numpy: Whether to vertically flip the numpy view.
    render_padding: Whether to render the padding in the human or numpy view.
    """
    assert not (mode == "agent" and not render_padding), "This parameter combination is unsupported; must render padding in agent mode."
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(ax_size, ax_size))
    ax.axis('off')
    ax.set_title(mode.title() + " view")
    
    if mode == "human":
        if render_padding:
            inner_grid = maze.EnvState(venv.env.callmethod('get_state')[idx]).inner_grid() 
            img = maze.render_inner_grid(inner_grid)
        else:
            img = venv.env.get_info()[idx]['rgb']
    elif mode == "agent":
        img = venv.reset()[idx].transpose(1,2,0) # (C, H, W) -> (H, W, C)
    elif mode == "numpy":
        env_state = maze.EnvState(venv.env.callmethod('get_state')[idx])
        grid = env_state.full_grid() if render_padding else env_state.inner_grid()
        img = grid[::(-1 if flip_numpy else 1), :] # Flip the numpy view vertically
    else:
        raise ValueError(f"Invalid mode {mode}")

    ax.imshow(img)
    if show_plot:
        plt.show() 
    else:
        return img

def custom_vfield(policy : t.nn.Module, venv : ProcgenGym3Env = None, seed : int = 0, ax_size : int = 2, callback : Callable = None, show_components : bool = False):
    """ Given a policy and a maze seed, create a maze editor and a vector field plot. Update the vector field whenever the maze is edited. Returns a VBox containing the maze editor and the vector field plot. 
    
    Args:
        policy: The policy to use to compute the vector field.
        venv: The environment to use to compute the vector field. If None, a new environment will be created.
        seed: The seed to use to create the environment.
        ax_size: The size of the vector field plot.
        callback: A callback to call whenever the maze is edited.
        show_components: Whether to show the vectors for each action.
    """
    output = Output()
    fig, ax = plt.subplots(1,1, figsize=(ax_size, ax_size))
    plt.close('all')
    if venv is None: 
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)

    # We want to update ax whenever the maze is edited
    def update_plot():
        # Clear the existing plot
        with output:
            vf = vfield.vector_field(venv, policy)
            ax.clear()
            vfield.plot_vf(vf, ax=ax, show_components=show_components)

            # Update the existing figure in place 
            clear_output(wait=True)
            display(fig)

    update_plot()

    def cb(gridm): # Callback for when the maze is edited
        if callback is not None:
            callback(gridm)
        update_plot()

    # Then make a callback which updates the render in-place when the maze is edited
    editors = maze.venv_editors(venv, check_on_dist=False, env_nums=range(1), callback=cb) # NOTE can't block off initial mouse position? 
    # Set the editors so that they don't space out when the window is resized
    for editor in editors:
        editor.layout.height = "100%"

    widget_vbox = Box(children=editors + [output], layout=Layout(display='flex', flex_flow='row', align_items='stretch', width='100%'))

    return widget_vbox

### Activation management
def get_activations(obs : np.ndarray, hook: cmh.ModuleHook, layer_name: str):
    hook.run_with_input(obs) # Run the model with the given obs
    return hook.get_value_by_label(layer_name) # Shape is (b, c, h, w) at conv layers, (b, activations) at linear layers

### Plotters
def plot_activations(activations: np.ndarray, fig: go.FigureWidget):
    """ Plot the activations given a single (non-batched) activation tensor. """
    fig.update(data=[go.Heatmap(z=activations)])

def plot_nonzero_activations(activations: np.ndarray, fig: go.FigureWidget): 
    """ Plot the nonzero activations in a heatmap. """
    # Find nonzero activations and cast to floats
    nz = (activations != 0).astype(np.float32)
    fig.update(data=[go.Heatmap(z=nz)])

def plot_nonzero_diffs(activations: np.ndarray, fig: go.FigureWidget):
    """ Plot the nonzero activation diffs in a heatmap. """
    diffs = activations[0] - activations[1]
    plot_nonzero_activations(diffs, fig)

NUM_ACTIONS = 15
def format_plotter(fig : go.Figure, activations : np.ndarray, title : str = None, is_policy_out : bool = False, bounds : Tuple[int, int] = None, px_dims : Tuple[int, int] = None):
    """ Format the figure. Takes activations as input so that the x- and z-axes can be formatted according to the activations. 
    
    If is_policy_out is True, the x-axis will be formatted as a policy output. If bounds is not None, the z-axis will be formatted to show the activations in the given bounds. """
    h, w = px_dims or (500, 500)
    fig.update_layout(height=h, width=w, title_text=title)
    fig.update_xaxes(side="top")        
    
    if is_policy_out:
        # Transform each index into the corresponding action label, according to maze.py 
        fig.update_xaxes(ticktext=[models.human_readable_action(i).title() for i in range(NUM_ACTIONS)], tickvals=np.arange(activations.shape[3])) 
    else: # Reset the x-axis ticks to match the y-axis ticks
        yaxis_ticktext, yaxis_tickvals = fig.layout.yaxis.ticktext, fig.layout.yaxis.tickvals # TODO double the step of yaxis ticks?
        fig.update_xaxes(ticktext=yaxis_ticktext, tickvals=yaxis_tickvals)

    fig.update_xaxes(side="top") # Set the x ticks to the top
    fig.update_yaxes(autorange="reversed") # Reverse the row-axis autorange        
    
    # Set the min and max to be the min and max of all channels at this label
    if bounds is None: 
        max_act = np.abs(activations).max()
        bounds = (-1 * max_act, max_act)
    fig.update_traces(zmin=bounds[0], zmid=0, zmax=bounds[1])    
    
    # Change the colorscale to split red (negative) -- white (zero) -- blue (positive)
    fig.update_traces(colorscale='RdBu')

# To indicate that fig can be matplotlib or plotly, we use the type go.FigureWidget

def plot_patch(patch : dict, hook : cmh.ModuleHook, layer : str = default_layer, channel : int = 0, fig : go.FigureWidget = None, title : str = None, bounds : Tuple[int, int] = None, px_dims : Tuple[int, int] = None):
    """ Plot the activations of a single patch, at the given layer and channel. Returns a figure. """
    assert layer in patch, f"Layer {layer} not in patch {patch}"

    if fig is None: 
        fig = go.FigureWidget()
    activations = patch[layer](hook.values_by_label[default_layer])
    assert activations.shape[1] > channel, f"Channel {channel} not in activations {activations.shape}"

    activations = activations[0, channel] # Shape is (h, w)
    plot_activations(activations, fig)
    format_plotter(fig, activations, title=title, bounds=bounds, px_dims=px_dims)
    return fig

# Widget helpers
def create_save_button(prefix : str, fig : plt.Figure, descriptors : Dict[str, float]): 
    """ Create a button that saves fig to a file. 
    
    Args:
        prefix (str): The prefix of the filename. Typically "experiments/visualizations/" or "playground/visualizations".
        fig (plt.Figure): The figure to save.
        descriptors (defaultdict[str, float]): A dictionary of descriptors to add to the filename.
    """
    def save_fig(b):
        """ Save the figure to a file. """
        filename = prefix
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
    return button # TODO integrate with activationsPlotter

class ActivationsPlotter:
    def __init__(self, labels: List[str], plotter: Callable, activ_gen: Callable, hook, coords_enabled: bool=False, defaults : dict = None, save_dir='experiments/', **act_kwargs):
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
        self.label_widget = Dropdown(options=formatted_labels, value=formatted_labels[0], description="Layers")
        self.channel_slider = IntSlider(min=0, max=127, step=1, value=0, description="Channel")
        
        # Add channel increment and decrement buttons
        button_width = '10px'
        decrement_button, increment_button = [Button(description=descr_str, layout=Layout(width=button_width)) for descr_str in ("-", "+")]
        def add_to_slider(x : int):
            # Clip the value to the min and max
            self.channel_slider.value = np.clip(self.channel_slider.value + x, self.channel_slider.min, self.channel_slider.max)
            self.update_plotter()
        decrement_button.on_click(lambda _: add_to_slider(-1))
        increment_button.on_click(lambda _: add_to_slider(1))
        self.widgets = [self.fig, self.label_widget, HBox([self.channel_slider, decrement_button, increment_button])] # TODO make this a helper for converting arbitrary sliders

        # Add row and column sliders if enabled
        self.coords_enabled = coords_enabled
        if coords_enabled:
            self.col_slider, self.row_slider = (IntSlider(min=0, max=62, step=1, value=32, description="Column"), IntSlider(min=0, max=63, step=1, value=32, description="Row"))
            self.widgets.extend([self.col_slider, self.row_slider])

        # Add a custom filename widget
        self.filename_widget = Text(value="", placeholder="Custom filename", disabled=False)
        self.filename_widget.layout.width = '150px'
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
                widget.observe(self.update_plotter, names='value')
        
        # Set the initial plot
        self.update_plotter()

    def display(self):
        """ Display the elements; this function separates functionality from implementation. """
        display(self.fig)
        display(VBox(self.widgets[1:-1])) # Show a VBox of the label dropdown and the sliders, centered beneath the plot
        display(self.widgets[-1])

    def save_image(self, b): # Add a save button to save the image
        basename = self.filename_widget.value if self.filename_widget.value != "" else f"{self.label_widget.value}_{self.channel_slider.value}{f'_{self.col_slider.value}_{self.row_slider.value}' if self.coords_enabled else ''}"
        filepath = f"{self.save_dir}/{basename}.png" # NOTE For some reason, PATH_PREFIX isn't necessary? Unsure why

        # Annotate to the outside of the plot
        old_title = self.fig.layout.title
        self.fig.layout.title = f"{self.label_widget.value};\nchannel {self.channel_slider.value}{f' at ({self.col_slider.value}, {self.row_slider.value})' if self.coords_enabled else ''}"

        self.fig.update_yaxes(autorange="reversed") 
        self.fig.write_image(filepath)
        print(f"Saved image to {filepath}")

        self.fig.layout.title = old_title # Clear the title
        
        self.filename_widget.value = "" # Clear the filename_widget box

    def update_plotter(self, b=None):
        """ Update the plot with the current values of the widgets. """
        label = expand_label(self.label_widget.value)        

        if self.coords_enabled:
            col, row = self.col_slider.value, self.row_slider.value
            activations = self.activ_gen(row, col, label, self.hook, **self.act_kwargs)
        else:
            activations = self.activ_gen(label, self.hook, **self.act_kwargs) # shape is (b, c, h, w) at conv layers, (b, activations) at linear layers 

        self.channel_slider.max = patch_utils.num_channels(hook=self.hook, layer_name=label) - 1 
        channel = self.channel_slider.value = min(self.channel_slider.value, self.channel_slider.max)

        if len(activations.shape) == 2: # Linear layer (batch, hidden_dim)
            # Ensure shape[1] is a perfect square
            sqrt_act = int(math.sqrt(activations.shape[1]))
            if sqrt_act * sqrt_act == activations.shape[1]:
                activations = np.reshape(activations, newshape=(activations.shape[0], 1, sqrt_act, sqrt_act)) # Make a dummy channel dimension
                # Annotate that there is no spatial meaning to the activations
                self.fig.update_layout(title_text=f"{self.label_widget.value}; reshaped to 2D; no spatial meaning")
            else:
                activations = np.expand_dims(activations, axis=(1,2)) # Add a dummy dimension to the activations

        self.plotter(activations=activations[:, channel], fig=self.fig) # Plot the activations
        format_plotter(fig=self.fig, activations=activations, is_policy_out=self.label_widget.value == 'fc_policy_out', title=self.label_widget.value)


    
