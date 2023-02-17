from procgen_tools.imports import *
from procgen_tools import maze, vfield, patch_utils

NUM_ACTIONS = 15

# LABEL HANDLING
def format_label(label : str):
    """Format a label for display in the visualization."""
    return label.replace("embedder.", "")

def expand_label(label : str):
    if not (label.startswith("fc_policy") or label.startswith("fc_value")):
        return "embedder." + label
    else:
        return label

def format_labels(labels : List[str]):
    """Format labels for display in the visualization."""
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

def get_pixel_loc(channel_pos : int, channel_size : int = 16):
    assert channel_pos < channel_size, f"channel_pos {channel_pos} must be less than channel_size {channel_size}"
    assert channel_pos >= 0, f"channel_pos {channel_pos} must be non-negative"

    scale = PIXEL_SIZE // channel_size
    return scale * channel_pos + scale // 2

def plot_pixel_dot(ax, row, col, color='r', size=50):
    pixel_loc =  get_pixel_loc(col), get_pixel_loc(row)
    ax.scatter(pixel_loc[0], pixel_loc[1], c=color, s=size)

def visualize_venv(venv : ProcgenGym3Env, idx : int = 0, mode : str="human", ax : plt.Axes = None, ax_size : int = 3, show_plot : bool = True):
    """ Visualize the environment. 
    
    Parameters: 
    venv: The environment to visualize
    idx: The index of the environment to visualize, in the vectorized environment.
    mode: The mode to visualize in. Can be "human", "agent", or "numpy"
    ax: The axis to plot on. If None, a new axis will be created.
    ax_size: The size of the axis to create, if ax is None.
    show_plot: Whether to show the plot. 
    """
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(ax_size, ax_size))
    ax.axis('off')
    ax.set_title(mode.title() + " view")
    
    if mode == "human":
        img = venv.env.get_info()[idx]['rgb']
    elif mode == "agent":
        img = venv.reset()[idx].transpose(1,2,0)
    elif mode == "numpy":
        img = maze.EnvState(venv.env.callmethod('get_state')[idx]).full_grid()[::-1, :]
    else:
        raise ValueError(f"Invalid mode {mode}")

    ax.imshow(img)
    if show_plot:
        plt.show() 

def custom_vfield(policy : t.nn.Module, venv : ProcgenGym3Env = None, seed : int = 0, ax_size : int = 2, callback : Callable = None):
    """ Given a policy and a maze seed, create a maze editor and a vector field plot. Update the vector field whenever the maze is edited. Returns a VBox containing the maze editor and the vector field plot. """
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
            vfield.plot_vf(vf, ax=ax)

            # Update the existing figure in place 
            clear_output(wait=True)
            display(fig)

    update_plot()

    def cb(gridm): # Callback for when the maze is edited
        if callback is not None:
            callback(gridm)
        update_plot()
        


    # Then make a callback which updates the render in-place when the maze is edited
    editors = maze.venv_editors(venv, check_on_dist=False, env_nums=range(1), callback=cb)
    # Set the editors so that they don't space out when the window is resized
    for editor in editors:
        editor.layout.height = "100%"

    widget_vbox = Box(children=editors + [output], layout=Layout(display='flex', flex_flow='row', align_items='stretch', width='100%'))

    return widget_vbox

class ActivationsPlotter:
    def __init__(self, labels: List[str], plotter: Callable, activ_gen: Callable, hook, coords_enabled: bool=False, defaults : dict = None, **act_kwargs):
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

        # Remove the _out layer and "embedder." prefixes
        formatted_labels = format_labels(labels)
        self.label_widget = Dropdown(options=formatted_labels, value=formatted_labels[0], description="Layers")
        self.channel_slider = IntSlider(min=0, max=127, step=1, value=0, description="Channel")
        self.widgets = [self.fig, self.label_widget, self.channel_slider]

        self.coords_enabled = coords_enabled
        if coords_enabled:
            self.col_slider, self.row_slider = (IntSlider(min=0, max=62, step=1, value=32, description="Column"), IntSlider(min=0, max=63, step=1, value=32, description="Row"))
            self.widgets.extend([self.col_slider, self.row_slider])

        self.filename_widget = Text(value="", placeholder="Custom filename", disabled=False)
        self.filename_widget.layout.width = '150px'

        self.button = Button(description="Save image")
        self.button.on_click(self.save_image)
        self.widgets.append(HBox([self.filename_widget, self.button]))

        if defaults is not None:
            for key, value in defaults.items():
                getattr(self, key).value = value

        for widget in self.widgets:
            if widget != self.fig:
                widget.observe(self.update_plotter, names='value')
        self.update_plotter()

    def display(self):
        """ Display the elements; this function separates functionality from implementation. """
        display(self.fig)
        display(VBox(self.widgets[1:-1])) # Show a VBox of the label dropdown and the sliders, centered beneath the plot
        display(self.widgets[-1])

    def save_image(self, b): # Add a save button to save the image
        basename = self.filename_widget.value if self.filename_widget.value != "" else f"{self.label_widget.value}_{self.channel_slider.value}{f'_{self.col_slider.value}_{self.row_slider.value}' if self.coords_enabled else ''}"
        filepath = f"{PATH_PREFIX}experiments/visualizations/{basename}.png"

        # Annotate to the outside of the plot
        old_title = self.fig.layout.title
        self.fig.layout.title = f"{self.label_widget.value};\nchannel {self.channel_slider.value}{f' at ({self.col_slider.value}, {self.row_slider.value})' if self.coords_enabled else ''}"

        self.fig.write_image(filepath)
        print(f"Saved image to {filepath}")

        self.fig.layout.title = old_title # Clear the title
        
        self.filename_widget.value = "" # Clear the filename_widget box

    def update_plotter(self, b=None):
        """ Update the plot with the current values of the widgets. """
        label = expand_label(self.label_widget.value)        
        self.fig.update_layout(height=500, width=500, title_text=self.label_widget.value)
        if self.coords_enabled:
            col, row = self.col_slider.value, self.row_slider.value
            activations = self.activ_gen(row, col, label, **self.act_kwargs)
        else:
            activations = self.activ_gen(label, **self.act_kwargs) # shape is (b, c, h, w) at conv layers, (b, activations) at linear layers

        shap = self.hook.get_value_by_label(label).shape
        self.channel_slider.max = shap[1] - 1 if len(shap) > 2 else 0
        self.channel_slider.value = min(self.channel_slider.value, self.channel_slider.max)
        channel = self.channel_slider.value
        assert channel < activations.shape[1], "Channel doesn't exist at this layer"

        if len(activations.shape) == 2: # Linear layer (batch, hidden_dim)
            # Ensure shape[1] is a perfect square
            sqrt_act = int(math.sqrt(activations.shape[1]))
            if sqrt_act * sqrt_act == activations.shape[1]:
                activations = np.reshape(activations, newshape=(activations.shape[0], 1, sqrt_act, sqrt_act)) # Make a dummy channel dimension
                # Annotate that there is no spatial meaning to the activations
                self.fig.update_layout(title_text=f"{self.label_widget.value}; reshaped to 2D; no spatial meaning")
            else:
                activations = np.expand_dims(activations, axis=(1,2)) # Add a dummy dimension to the activations


        if label == 'fc_policy_out':
            # Transform each index into the corresponding action label, according to maze.py 
            self.fig.update_xaxes(ticktext=[models.human_readable_action(i).title() for i in range(NUM_ACTIONS)], tickvals=np.arange(activations.shape[3])) 
        else: # Reset the x-axis ticks to match the y-axis ticks
            yaxis_ticktext, yaxis_tickvals = self.fig.layout.yaxis.ticktext, self.fig.layout.yaxis.tickvals # TODO double the step of yaxis ticks?
            self.fig.update_xaxes(ticktext=yaxis_ticktext, tickvals=yaxis_tickvals)

        self.fig.update_xaxes(side="top") # Set the x ticks to the top
        self.fig.update_yaxes(autorange="reversed") # Reverse the row-axis autorange
        

        self.plotter(activations=activations[:, channel], fig=self.fig) # Plot the activations

        # Set the min and max to be the min and max of all channels at this label
        bounds = np.abs(activations).max()
        self.fig.update_traces(zmin=-1 * bounds, zmid=0, zmax=bounds)    
        
        # Change the colorscale to split red (negative) -- white (zero) -- blue (positive)
        self.fig.update_traces(colorscale='RdBu')