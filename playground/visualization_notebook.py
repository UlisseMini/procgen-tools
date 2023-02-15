# %% [markdown]
# # Visualizing details of the goal misgeneralization nets
# Let's understand lots of details about [the goal misgeneralization paper](https://arxiv.org/abs/2105.14111). In particular, we'll be looking at the cheese-maze task from the goal misgeneralization task, for which cheese was spawned in the 5x5 top-right corner of the maze. 
# 
# Key conclusions:
# 1. Convolutional layers limit speed of information propagation (_locality_). More precisely, ignoring the effect of `maxpool2D` layers, any portions of the state separated by `n` pixels take at least `ceil(n/2)` convolutional layers to interact.
# 2. In the maximal maze size of 64 x 64, there is at most **two** steps of computation involving information from e.g. opposite corners of the maze. 

# %%
# %% Don't have to restart kernel and reimport each time you modify a dependency
%reload_ext autoreload
%autoreload 2

# %%
# Imports
from typing import List, Tuple, Dict, Union, Optional, Callable
import re 

import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm import tqdm
from einops import rearrange
from IPython.display import *
from ipywidgets import *
import itertools
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt

# Install procgen tools if needed
try:
  import procgen_tools
except ImportError:
  get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

# %%
# Download data and create directory structure

import os, sys
from pathlib import Path
from procgen_tools.utils import setup

setup() # create directory structure and download data

# path this notebook expects to be in
if 'experiments' not in os.getcwd():
    Path('experiments').mkdir(exist_ok=True)
    os.chdir('experiments')


# %%
import circrl.module_hook as cmh
import procgen_tools.models as models
from procgen_tools.patch_utils import *
from procgen_tools.visualization import *

from procgen import ProcgenGym3Env
from ipywidgets import Text # Import this later because otherwise Text gets cast as str?

RAND_REGION = 5
NUM_ACTIONS = 15
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False
PATH_PREFIX = '../' if in_jupyter else ''

# %%
# Load model
model_path = PATH_PREFIX + f'trained_models/maze_I/model_rand_region_{RAND_REGION}.pth'
policy = models.load_policy(model_path, NUM_ACTIONS, t.device('cpu'))
hook = cmh.ModuleHook(policy)

# %% [markdown]
# Let's visualize the network structure. Here's a Mermaid diagram. 
# 
# ![](https://i.imgur.com/acsV4aD.png) 

# %% [markdown]
# And here's a more dynamic view; small nodes are activations, and large nodes are `nn.Module`s.

# %%
hook.run_with_input(np.zeros((1,3, 64, 64), dtype=np.float32))
hook.get_graph(include_parent_modules=False)

# %%
def dummy_obs_pair(color: str, location: Tuple[int, int]=(32,32)):
    """ Returns a mostly-black image pair, the first of which contains a red/green/blue pixel in the center. Returns obs of shape (2, 3, 64, 64). """
    
    assert color in ['R', 'G', 'B'], f'Color must be one of R, G, B, not {color}'
    assert len(location) == 2, 'Location must be a tuple of length 2'
    assert all(0 <= col < 64 for col in location), 'Location must be in [0, 64)'

    channel = {'R': 0, 'G': 1, 'B': 2}[color]
    obs = np.zeros((2, 3, 64, 64), dtype=np.float32)
    obs[0, channel, location[0], location[1]] = 1 # Have one pixel in the middle, in the given channel
    return obs
    
# Let's load a dummy observation with only one nonzero value
plt.imshow(dummy_obs_pair("G")[0].transpose(1,2,0))

# %%
# Get the available labels
hook.run_with_input(dummy_obs_pair('R'))
labels = list(hook.values_by_label.keys())[:-1] # Skip the "_out" layer and remove "embedder." prefixes
assert labels == list(map(expand_label, format_labels(labels)))

# %%
# Let's visualize the activations at each layer using plotly, using an interactive interface that lets us slide the R/G/B pixel around
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def get_activations(label: str, obs: np.ndarray):
    hook.run_with_input(obs) # Run the model with the given obs
    return hook.get_value_by_label(label) # Shape is (b, c, h, w) at conv layers, (b, activations) at linear layers

def activation_diff(label: str, obs: np.ndarray):
    assert obs.shape[0] == 2 # Must be a pair of observations
    
    activations = get_activations(label, obs)
    return activations[0] - activations[1] # Subtract in order to cancel out bias terms which don't behave differently in the presence of the differing inputs 

def plot_activations(activations: np.ndarray, fig: go.FigureWidget):
    """ Plot the activations given a single (non-batched) activation tensor. """
    fig.update(data=[go.Heatmap(z=activations)])

# %%
# Now let's turn the above into a class. We want to be able to call the plotter with a single function call, and have it update the plot automatically.
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
        if self.coords_enabled:
            col, row = self.col_slider.value, self.row_slider.value
            activations = self.activ_gen(row, col, label, **self.act_kwargs)
        else:
            activations = self.activ_gen(label, **self.act_kwargs) # shape is (b, c, h, w) at conv layers, (b, activations) at linear layers

        shap = self.hook.get_value_by_label(label).shape
        self.channel_slider.max = shap[1] - 1 if len(shap) > 2 else 0
        self.channel_slider.value = min(self.channel_slider.value, self.channel_slider.max)
        channel = self.channel_slider.value

        if len(activations.shape) == 2: # Linear layer (batch, hidden_dim) TODO check this 
            # Unsqueeze the np.ndarray
            activations = np.expand_dims(activations, axis=(1,2))
            # If there's only a single channel, display a 1D Heatmap, with a single rowvalue and the activation indices as the col values TODO remove columns
        else: 
            assert channel < activations.shape[1], "Channel doesn't exist at this layer"

        self.fig.update_layout(height=500, width=500, title_text=self.label_widget.value)
        if label == 'fc_policy_out':
            # Transform each index into the corresponding action label, according to maze.py 
            self.fig.update_xaxes(ticktext=[models.human_readable_action(i).title() for i in range(NUM_ACTIONS)], tickvals=np.arange(activations.shape[3])) # TODO is this correct, for 3 instead of 2?
        else: # Reset the indices so that there are no xticks
            self.fig.update_xaxes(ticktext=[], tickvals=[])

        self.fig.update_xaxes(side="top") # Set the x ticks to the top
        self.fig.update_yaxes(autorange="reversed") # Reverse the row-axis autorange
        
        self.plotter(activations=activations[:, channel], fig=self.fig) # Plot the activations

        # Set the min and max to be the min and max of all channels at this label
        bounds = np.abs(activations).max()
        self.fig.update_traces(zmin=-1 * bounds, zmid=0, zmax=bounds)    
        
        # Change the colorscale to split red (negative) -- white (zero) -- blue (positive)
        self.fig.update_traces(colorscale='RdBu')

# %%
# We can begin to see how information propagates across the net for a dummy single-pixel input
def activ_gen_px(col: int, row: int, label: str, color: str, pair: bool = False):
    """ Get the activations for running a forward pass on a dummy observation, in the given color. Returns shape of (batch, channels, rows, cols), where size batch=1 if pair=False, and batch=2 if pair=True. """
    return get_activations(label, dummy_obs_pair(color, (row, col)))

# Instantiate the plotter
activ_plotter = ActivationsPlotter(labels, plotter=lambda activations, fig: plot_activations(activations[0], fig=fig), activ_gen=activ_gen_px, hook=hook, coords_enabled=True, color="R") # TODO for some reason block2.maxpool_out has a different fig width?
activ_plotter.display()

# %%
# Plot the diff between a single-pixel input and a blank input
activ_plotter = ActivationsPlotter(labels, plotter=lambda activations, fig: plot_activations(activations[0] - activations[1], fig=fig), activ_gen=activ_gen_px, hook=hook, coords_enabled=True, color="R")
activ_plotter.display()

# %%
def plot_nonzero_activations(activations: np.ndarray, fig: go.FigureWidget): 
    """ Plot the nonzero activations in a heatmap. """
    # Find nonzero activations and cast to floats
    nz = (activations != 0).astype(np.float32)
    fig.update(data=[go.Heatmap(z=nz)])

def plot_nonzero_diffs(activations: np.ndarray, fig: go.FigureWidget):
    """ Plot the nonzero activation diffs in a heatmap. """
    diffs = activations[0] - activations[1]
    plot_nonzero_activations(diffs, fig)

# Instantiate the plotter
nonzero_plotter = ActivationsPlotter(labels, plotter=plot_nonzero_diffs, activ_gen=activ_gen_px, hook=hook, coords_enabled=True, color="R", pair=True)
nonzero_plotter.display()

# %% [markdown]
# # 1: Locality
# Consider `n` convolutional layers (3x3 kernel, stride=1, padding=1) which each preserve the height col width of the previous feature maps. The above demonstrates that after these layers, information can only propagate `n` L1 pixels. The network itself is composed of # TODO 

# %% [markdown]
# # Visualizing actual observation activations

# %%
default_settings = {'channel_slider': 55, 'label_widget': 'block2.res1.resadd_out'}
# %%
def activ_gen_cheese(label: str, venv : ProcgenGym3Env = None): # TODO dont use None
    """ Generate an observation with cheese at the given location. Returns a tensor of shape (1, 3, rows, cols)."""
    assert venv is not None
    cheese_obs = venv.reset() 
    cheese_obs = np.array(cheese_obs, dtype=np.float32)
    activations = get_activations(label, cheese_obs)
    return activations

# Show a maze editor side-by-side with the interactive plotter
SEED = 0
venv = create_venv(num=1, start_level=SEED, num_levels=1) # This has to be a single maze, otherwise the vfield wont work
custom_maze_plotter = ActivationsPlotter(labels, lambda activations, fig: plot_activations(activations[0], fig=fig), activ_gen_cheese, hook, defaults=default_settings, venv=venv)
# Set the default settings


widget_box = custom_vfield(policy, venv=venv, callback=custom_maze_plotter.update_plotter) 
display(widget_box)
    
custom_maze_plotter.display() 
# TODO make it so that it's easy to attach notes to files, load  

# %%