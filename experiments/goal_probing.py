# %%[markdown]
# The goal of this experiment is to identify (sparse?) directions in activation 
# space in different layers that can accurately predict whether the agent will
# take the cheese or go to the corner instead.  Steps:
# - Generate a bunch of data: 
#   - Iterate over a bunch of mazes (1000?)
#   - Position the cheese in a specific location that creates roughly 50%
#     expected chance that a specific agent (rand_region_5?) will pick the cheese
#   - Run a rollout, store all the information.
# - Then, create some specific data sets:
#   - Activation at layer N at timestep 0 on every maze as input features
#   - Whether the agent got the cheese as target variable
#   - Same for activations at decision square
#   - Maybe activations at step 0 predicting argmax action at decision square?
#     (That is, does the network "know what it's going to do" from the start?)


# %%
# Imports
%reload_ext autoreload
%autoreload 2

import os
import random
import glob

import numpy as np
import pandas as pd
import torch as t
import xarray as xr
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm.auto import tqdm
from einops import rearrange
from IPython.display import Video, display, clear_output
import networkx as nx

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro

import procgen_tools.models as models
import procgen_tools.maze as maze
import gatherdata
import gatherdata_rich

# %%
# Functions

def maze_grid_to_graph(inner_grid):
    '''Convert a provided maze inner grid to a networkX graph object'''
    def nodes_where(cond):
        return [(r, c) for r, c in zip(*np.where(cond))]
    # Create edges: each node may have an edge up, down, left or right, check
    # each direction for all nodes at the same time
    edges = []
    for dirs, g0, g1 in [
            ['RL', inner_grid[:,:-1], inner_grid[:,1:]],
            ['UD', inner_grid[:-1,:], inner_grid[1:,:]],]:
        # Find squares that are open in both g0 and g1, and add an edge
        node0s = nodes_where((g0!=maze.BLOCKED)&(g1!=maze.BLOCKED))
        node1s = [(r, c+1) if dirs=='RL' else (r+1, c) 
            for r, c in node0s]
        edges.extend([(n0, n1) for n0, n1 in zip(node0s, node1s)])
    graph = nx.Graph()
    graph.add_edges_from(edges)
    #nx.draw_networkx()
    # colors_by_node = {(0, 0): 'green', maze.get_cheese_pos(inner_grid): 'yellow',
    #     (inner_grid.shape[0]-1, inner_grid.shape[1]-1): 'red'}
    # node_colors = [colors_by_node.get(node, 'blue') for node in graph.nodes]
    # nx.draw_kamada_kawai(graph, node_color=node_colors, node_size=10)
    return graph

def grid_graph_has_decision_square(inner_grid, graph):
    cheese_node = maze.get_cheese_pos(inner_grid)
    corner_node = (inner_grid.shape[0]-1, inner_grid.shape[1]-1)
    pth = nx.shortest_path(graph, (0, 0), corner_node)
    return (not cheese_node in pth)

def get_decision_square_from_grid_graph(inner_grid, graph):
    cheese_node = maze.get_cheese_pos(inner_grid)
    corner_node = (inner_grid.shape[0]-1, inner_grid.shape[1]-1)
    path_to_cheese = nx.shortest_path(graph, (0, 0), cheese_node)
    path_to_corner = nx.shortest_path(graph, (0, 0), corner_node)
    for ii, cheese_path_node in enumerate(path_to_cheese):
        if ii >= len(path_to_corner):
            return cheese_path_node
        if cheese_path_node != path_to_corner[ii]:
            return path_to_cheese[ii-1]

def maze_has_decision_square(states_bytes):
    maze_env_state = maze.EnvState(states_bytes)
    inner_grid = maze_env_state.inner_grid()
    grid_graph = maze_grid_to_graph(inner_grid)
    return grid_graph_has_decision_square(inner_grid, grid_graph)

def setup_env():
    has_dec_sq = False
    while not has_dec_sq:
        start_level = random.randint(0, 1e6)
        venv = gatherdata.create_venv(start_level=start_level)    
        episode_metadata = dict(start_level=start_level, 
            level_seed=int(venv.env.get_info()[0]["level_seed"]))
        has_dec_sq = maze_has_decision_square(venv.env.callmethod('get_state')[0])
    return venv, episode_metadata

class NoDecisionSquareException(Exception):
    pass

class NotReachedDecisionSquareException(Exception):
    pass

def process_rollout(fn, hook, value_labels=
        ['embedder.reluflatten_out', 'embedder.fc_out']):
    '''Extract quantities of interest from a rollout file:
        - location of decision square
        - termination status of episode (just cheese or not for now)
        - layer activations at decision square'''
    # Load the data
    episode_data = cro.load_saved_rollout(fn)
    seq = episode_data['seq']
    # Get the decision square location
    maze_env_state = maze.EnvState(seq.custom['state_bytes'][0].values[()])
    inner_grid = maze_env_state.inner_grid()
    grid_graph = maze_grid_to_graph(inner_grid)
    #px.imshow(rearrange(episode_data['seq'].obs[0].values, 'c h w -> h w c')).show()
    if not grid_graph_has_decision_square(inner_grid, grid_graph):
        raise NoDecisionSquareException
    dec_node = get_decision_square_from_grid_graph(inner_grid, grid_graph)
    # Get the decision square timestep
    dec_step = None
    for step in seq.obs.coords['step']:
        mst = maze.EnvState(seq.custom['state_bytes'][step].values[()])
        if maze.get_mouse_pos(mst.inner_grid()) == dec_node:
            dec_step = step
    if dec_step is None:
        raise NotReachedDecisionSquareException
    # Probe the network
    hook.probe_with_input(seq.obs.sel(step=[dec_step]).astype(np.float32))
    values = {label: hook.get_value_by_label(label).squeeze(dim='step') 
        for label in value_labels}
    return dict(
        dec_node = dec_node,
        did_get_cheese = episode_data['seq'].rewards[-1].values[()]>0.,
        values = values,
    )


# %%
# Set up some stuff
if __name__ == "__main__":
    model_file = '../trained_models/maze_I/model_rand_region_5.pth'
    policy = models.load_policy(model_file, action_size=15, device=t.device('cpu'))
    model_name = os.path.basename(model_file)

# %%
# Create a bunch of data
if __name__ == "__main__":
    gatherdata_rich.get_maze_dataset(policy, model_name, 10, 200, env_setup_func=setup_env)
    
# %%
# Load some data
if __name__ == "__main__":
    dr = '../episode_data/20230130T204732' # Big run, but some don't have dec squares
    #dr = '../episode_data/20230130T221407' # Short test run
    
    hook = cmh.ModuleHook(policy)
    data_all = []
    fns = glob.glob(os.path.join(dr, '*.dat'))
    for fn in tqdm(fns):
        try:
            data_all.append(process_rollout(fn, hook))
        except (NoDecisionSquareException, NotReachedDecisionSquareException):
            pass #print(f'No decision square in rollout {ii}')

# %%
# Parse into a single set of tables and do some fitting!
if __name__ == "__main__":
    label = 'embedder.fc_out'
    activ = xr.concat([dd['values'][label] for dd in data_all], dim='batch')
    did_get_cheese = xr.DataArray([dd['did_get_cheese'] for dd in data_all],
        dims=['batch'])
