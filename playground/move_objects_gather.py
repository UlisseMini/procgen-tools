import os
import pickle
import random

import numpy as np
import pandas as pd
import scipy as sp
import torch as t
import xarray as xr
import plotly.express as px
import plotly as py
import plotly.subplots
import plotly.graph_objects as go
from einops import rearrange
from IPython.display import Video, display
from tqdm.auto import tqdm
from argparse import ArgumentParser

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr

import procgen_tools.models as models
import procgen_tools.maze as maze

def make_maze_and_move_object():
    # Pick a random maze
    start_level = random.randint(0, 1e6)
    venv = maze.create_venv(num=1, start_level=start_level, num_levels=1)    
    # Put the mouse on a random square chosen from the set of squares that are on either path-to-cheese, path-to-corner or both
    env_state = maze.EnvState(venv.env.callmethod('get_state')[0])
    inner_grid = env_state.inner_grid()
    graph = maze.maze_grid_to_graph(inner_grid)
    path_to_cheese = maze.get_path_to_cheese(inner_grid, graph)
    path_to_corner = maze.get_path_to_corner(inner_grid, graph)
    path_nodes = list(set(path_to_corner) + set(path_to_cheese))
    start_mouse_pos = path_nodes[random.randint(0, len(path_nodes)-1)]
    orig_mouse_pos = maze.get_object_pos_in_grid(inner_grid, maze.MOUSE)
    # TODO: reconcile mouse pos in grid and state bytes, need to set state bytes!
    env_state.
    inner_grid[start_mouse_pos] = maze.EMPTY
    inner_grid[start_mouse_pos] = maze.EMPTY
    # Get an observation from this maze state
    # Decide whether to move the mouse or the cheese (50-50)
    move_mouse = bool(random.randint(0, 1))
    #     If mouse, move it to another random square chosen from the above on-distribution paths
    #     If cheese, move it to a random open square
    # Get a new observation from the modified maze
    # Run both observations through a forward pass
    # Diff all the activations at all layers, and store the scaled norms of the activation diffs by channel
    #  (Scaled by number of elements in tensor)
    # Calculate some stats about the maze mod that might be relevant (how different the cheese path is, etc.)
    # Save the relevant info: (level_seed, initial maze state, modified maze state, 
    #   full set of activation diff norms, any maze stats that it's easier to calculate now)
    venv.close()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_file', type=str, 
        default='../trained_models/maze_I/model_rand_region_5.pth')
    parser.add_argument('--num-mazes', type=int, default=256, help='number of mazes to gether data for')
    parser.add_argument('--seed', type=int, default=42, help='seed for maze level generator')

    args = parser.parse_args()

    random.seed(args.seed)

    policy = models.load_policy(args.model_file, action_size=15, device=t.device('cpu'))
    model_name = os.path.basename(args.model_file)

    # Get all the data
    data_all = []
    for ii in tqdm(range(args.num_mazes)):
        data_all.append(make_maze_and_move_object())

    # Process and save it
    # ...