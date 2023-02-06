# Imports
import os
import glob
import pickle

import numpy as np
import pandas as pd
import scipy as sp
import torch as t
import xarray as xr
from tqdm.auto import tqdm
from einops import rearrange
from argparse import ArgumentParser

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro

import procgen_tools.maze as maze
import gatherdata_rich

# %%
# Functions and classes

# Exceptions
class PostprocException(Exception):
    pass

class NoDecisionSquareException(PostprocException):
    pass

class NotReachedDecisionSquareException(PostprocException):
    pass

# Get the decision square location
def get_cheese_and_dec_nodes(seq):
    maze_env_state = maze.EnvState(seq.custom['state_bytes'][0].values[()])
    inner_grid = maze_env_state.inner_grid()
    grid_graph = maze.maze_grid_to_graph(inner_grid)
    #px.imshow(rearrange(episode_data['seq'].obs[0].values, 'c h w -> h w c')).show()
    cheese_node = maze.get_cheese_pos(inner_grid)
    dec_node = maze.get_decision_square_from_grid_graph(inner_grid, grid_graph)
    return cheese_node, dec_node, inner_grid, grid_graph

# Get the decision square timestep
def get_dec_step(seq, dec_node):
    dec_step = None
    for step in seq.obs.coords['step']:
        mst = maze.EnvState(seq.custom['state_bytes'][step].values[()])
        if maze.get_mouse_pos(mst.inner_grid()) == dec_node:
            dec_step = step
    return dec_step

def get_all_mouse_pos(seq):
    step_coord = seq.custom['state_bytes'].coords['step']
    mouse_pos_arr = maze.get_mouse_pos_from_seq_of_states(
        seq.custom['state_bytes'].values)
    mouse_pos = xr.DataArray(
        data = mouse_pos_arr,
        dims = ['step', 'axis'],
        coords = {'step': step_coord, 'axis': ['y', 'x']})
    return mouse_pos

def proc_batch_of_obs(episode_data):
    '''Extract a big bunch of observations, with some other interesting data:
        - location of decision square, None if NA
        - location of cheese square
        - termination status of episode (just cheese or not for now)
        - all observations
        - all mouse locations
        - maze state bytes at initial timestep
        - level seed'''
    seq = episode_data['seq']
    cheese_node, dec_node, inner_grid, grid_graph = get_cheese_and_dec_nodes(seq)
    mouse_pos = get_all_mouse_pos(seq)
    return dict(
        dec_node = dec_node,
        cheese_node = cheese_node,
        did_get_cheese = episode_data['seq'].rewards[-1].values[()]>0.,
        obs = seq.obs.astype(np.float32),
        mouse_pos = mouse_pos,
        initial_state_bytes = seq.custom['state_bytes'].isel(step=0).values[()],
        level_seed = episode_data['episode_metadata']['level_seed'],
    )

def proc_probe_data(episode_data):
    '''Extract quantities of interest from a rollout file:
        - location of decision square
        - location of cheese square
        - termination status of episode (just cheese or not for now)
        - observations at decision square
        - maze state bytes at decision square
        '''
    seq = episode_data['seq']
    cheese_node, dec_node, inner_grid, grid_graph = get_cheese_and_dec_nodes(seq)
    if dec_node is None:
        raise NoDecisionSquareException
    dec_step = get_dec_step(seq, dec_node)
    if dec_step is None:
        raise NotReachedDecisionSquareException
    # Return stuff
    return dict(
        dec_node = dec_node,
        cheese_node = cheese_node,
        did_get_cheese = episode_data['seq'].rewards[-1].values[()]>0.,
        obs = seq.obs.sel(step=dec_step).astype(np.float32),
        dec_state_bytes = seq.custom['state_bytes'].sel(step=dec_step).values[()]
    )

def proc_probe_data_initial(episode_data):
    '''Extract quantities of interest from a rollout file:
        - level_seed
        - decision step
        - location of decision square
        - location of cheese square
        - termination status of episode (just cheese or not for now)
        - observations at initial square
        - maze state bytes at initial square
        - paths to cheese and top-right corner
        '''
    seq = episode_data['seq']
    cheese_node, dec_node, inner_grid, grid_graph = get_cheese_and_dec_nodes(seq)
    dec_step = get_dec_step(seq, dec_node)
    # Return stuff
    return dict(
        level_seed = episode_data['episode_metadata']['level_seed'],
        dec_step = dec_step,
        dec_node = dec_node,
        cheese_node = cheese_node,
        did_get_cheese = episode_data['seq'].rewards[-1].values[()]>0.,
        obs = seq.obs.isel(step=0).astype(np.float32),
        init_state_bytes = seq.custom['state_bytes'].isel(step=0).values[()],
        path_to_cheese = maze.get_path_to_cheese(inner_grid, grid_graph),
        path_to_corner = maze.get_path_to_corner(inner_grid, grid_graph),
    )

POSTPROC_FUNCS_BY_TYPE = {
    'batch_of_obs': proc_batch_of_obs,
    'probe_data':   proc_probe_data,
    'probe_data_initial': proc_probe_data_initial,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('postproc_type', type=str,
        choices=POSTPROC_FUNCS_BY_TYPE.keys())
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--max-files', type=int, default=None, 
        help='maximum timesteps per episode')
    
    args = parser.parse_args()
    
    # Get the post-processing function
    postproc_func = POSTPROC_FUNCS_BY_TYPE[args.postproc_type]
    data_all = []
    exceptions = []
    fns = glob.glob(os.path.join(args.data_dir, '*.dat'))
    if args.max_files is not None:
        fns = fns[:args.max_files]
    for fn in tqdm(fns):
        try:
            # Load the data for this file
            episode_data = cro.load_saved_rollout(fn)
            data_all.append(postproc_func(episode_data))
        except PostprocException as ex:
            exceptions.append((fn, ex.__repr__()))

    # Save the results in the same data directory, uncompressed for now
    with open(os.path.join(args.data_dir, 
            f'postproc_{args.postproc_type}.pkl'), 'wb') as fl:
        pickle.dump({'data': data_all, 'exceptions': exceptions}, 
            fl, protocol=pickle.HIGHEST_PROTOCOL)






