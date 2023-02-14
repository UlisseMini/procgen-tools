from typing import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, HTML
from ipywidgets import Output, VBox
import procgen_tools.maze as maze

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

# Visualization subroutines
def custom_vfield(policy : torch.nn.Module, seed : int = 0, ax_size : int = 3, callback : Callable = None):
    """ Given a policy and a maze seed, create a maze editor and a vector field plot. Update the vector field whenever the maze is edited. Returns a VBox containing the maze editor and the vector field plot. """
    output = Output()
    fig, ax = plt.subplots(1,1, figsize=(ax_size, ax_size))
    plt.close()
    single_venv = maze.create_venv(num=1, start_level=seed, num_levels=1)

    # We want to update ax whenever the maze is edited
    def update_plot():
        # Clear the existing plot
        with output:
            vfield = vector_field(single_venv, policy)
            ax.clear()
            plot_vf(vfield, ax=ax)

            # Update the existing figure in place 
            clear_output(wait=True)
            display(fig)

    update_plot()

    def cb(_): # Callback for when the maze is edited
        update_plot()
        if callback is not None:
            callback()

    # Then make a callback which updates the render in-place when the maze is edited
    editors = maze.venv_editors(single_venv, check_on_dist=False, env_nums=range(1), callback=cb)

    # Display the maze editor and the plot in an HBox
    widget_vbox = VBox(editors + [output])
    return widget_vbox