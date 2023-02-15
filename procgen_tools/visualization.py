from typing import List, Tuple, Dict, Union, Optional, Callable

import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from einops import rearrange
from IPython.display import *
from ipywidgets import *
import itertools
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt

import procgen_tools.models as models
import procgen_tools.maze as maze
from procgen_tools.vfield import *

NUM_ACTIONS = 15

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

def get_stride(label : str):
    """Get the stride of the layer referred to by label. How many pixels required to translate a single entry in the feature maps of label. """
    if not label.startswith("embedder.block"): raise ValueError(f"Not in the Impala blocks.")

    block_num = get_impala_num(label)
    if 'conv_out' in label: # Before that Impala layer's maxpool has been applied
        block_num -= 1
    return 2 ** block_num

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

def custom_vfield(policy : t.nn.Module, venv : ProcgenGym3Env = None, seed : int = 0, ax_size : int = 3, callback : Callable = None):
    """ Given a policy and a maze seed, create a maze editor and a vector field plot. Update the vector field whenever the maze is edited. Returns a VBox containing the maze editor and the vector field plot. """
    output = Output()
    fig, ax = plt.subplots(1,1, figsize=(ax_size, ax_size))
    plt.close('all')
    if venv is None: 
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    # else:
        # assert venv.num == 1, "Can only visualize a single environment at a time."

    # We want to update ax whenever the maze is edited
    def update_plot():
        # Clear the existing plot
        with output:
            vfield = vector_field(venv, policy)
            ax.clear()
            plot_vf(vfield, ax=ax)

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

    # Display the maze editor and the plot in an HBox
    widget_vbox = VBox(editors + [output])
    return widget_vbox