# Imports and initial setup
from typing import List, Tuple, Dict, Union, Optional, Callable
import random
import itertools
import copy
import pickle
import os
import datetime
import sys

import numpy as np
import numpy.linalg
import pandas as pd
from sklearn.model_selection import train_test_split
import xarray as xr
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm.auto import tqdm
from einops import rearrange
from IPython.display import Video, display, clear_output
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array, vfx
from argparse import ArgumentParser

import lovely_tensors as lt
lt.monkey_patch()

import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr
import procgen_tools.models as models
import procgen_tools.maze as maze
import procgen_tools.vfield as vfield
import procgen_tools.rollout_utils as rollout_utils

path_prefix = '../'

def load_value(value_label, value_cache_dr):
    with open(os.path.join(value_cache_dr, f'{value_label}.pkl'), 'rb') as fl:
        value = pickle.load(fl)
    return value

if __name__ == "__main__":
    print('Parsing arguments...')
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print(os.getcwd())

    parser = ArgumentParser()
    parser.add_argument('--rand-region', type=int, default=5)
    
    dft_value_labels = values_to_store = [
        'embedder.block1.conv_in0',
        'embedder.block1.res1.resadd_out',
        'embedder.block1.res2.resadd_out',
        'embedder.block2.res1.resadd_out',
        'embedder.block2.res2.resadd_out',
        'embedder.block3.res1.resadd_out',
        'embedder.block3.res2.resadd_out',
        'embedder.flatten_out',
        'embedder.relufc_out',
        'fc_policy_out',
    ]
    parser.add_argument('--value-labels', nargs='+', type=str, default=dft_value_labels)
    parser.add_argument('--obs-fn', type=str, default='action_probing_obs.pkl', help='file containing observations to process')
    
    utc = datetime.datetime.utcnow()
    dft_save_dr = utc.strftime('action_probing_proc_%Y%m%dT%H%M%S')
    parser.add_argument('--save-dr', type=str, default=dft_save_dr, help='directory to save data')

    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.save_dr):
        os.mkdir(args.save_dr)

    # Generate or load a large batch of observations, run through hooked network to get fc activations,
    # cache these as dataset along with "next cheese action" and "next corner action".
    print('Generating / Loading obs data...')

    num_obs_normal = 25000
    num_obs_dec = 5000
    obs_batch_size = 5000

    # num_obs_normal = 500
    # num_obs_dec = 500
    # obs_batch_size = 500

    hook_batch_size = 500
    value_labels = ['embedder.flatten_out', 'embedder.relufc_out']
    logits_value_label = 'fc_policy_out'

    REDO_OBS = False
    cache_fn = args.obs_fn

    policy = models.load_policy(path_prefix + 
            f'trained_models/maze_I/model_rand_region_{args.rand_region}.pth', 
        15, t.device('cpu'))
    hook = cmh.ModuleHook(policy)

    def get_action(curr_pos, next_pos):
        if next_pos[0] < curr_pos[0]: return 'D'
        if next_pos[0] > curr_pos[0]: return 'U'
        if next_pos[1] < curr_pos[1]: return 'L'
        if next_pos[1] > curr_pos[1]: return 'R'
        return 'N'

    if not os.path.isfile(cache_fn) or REDO_OBS:
        next_level_seed = 0
        
        # Get a bunch of obs not necessarily on dec square to get decent navigational basis
        print(f'Get {num_obs_normal} normal observations...')
        obs_list = []
        obs_meta_normal_list = []
        for batch_start_ind in tqdm(range(0, num_obs_normal, obs_batch_size)):
            obs_normal, obs_meta_normal, next_level_seed = maze.get_random_obs_opts(
                obs_batch_size, 
                start_level=next_level_seed, return_metadata=True, random_seed=next_level_seed, 
                deterministic_levels=True, show_pbar=True)
            obs_list.append(obs_normal)
            obs_meta_normal_list.extend(obs_meta_normal)
        obs_normal = np.concatenate(obs_list, axis=0)
        
        # Also get a bunch on dec squares to show diversity between cheese/corner actions
        print(f'Get {num_obs_dec} decision square observations...')
        obs_list = []
        obs_meta_dec_list = []
        for batch_start_ind in tqdm(range(0, num_obs_dec, obs_batch_size)):
            obs_dec, obs_meta_dec, next_level_seed = maze.get_random_obs_opts(
                obs_batch_size, 
                start_level=next_level_seed, return_metadata=True, random_seed=next_level_seed, 
                deterministic_levels=True, show_pbar=True, must_be_dec_square=True)
            obs_list.append(obs_dec)
            obs_meta_dec_list.extend(obs_meta_dec)
        obs_dec = np.concatenate(obs_list, axis=0)

        # Merge into a single batch of observations
        obs = np.concatenate([obs_normal, obs_dec], axis=0)
        obs_meta = obs_meta_normal_list + obs_meta_dec_list

        # Extract best action for cheese and corner paths
        next_action_cheese = np.array([get_action(md['mouse_pos_outer'], 
                md['next_pos_cheese_outer'])
            for md in obs_meta])
        next_action_corner = np.array([get_action(md['mouse_pos_outer'], 
                md['next_pos_corner_outer'])
            for md in obs_meta])
        
        with open(cache_fn, 'wb') as fl:
            pickle.dump((obs, obs_meta, next_action_cheese, next_action_corner), fl)

    else:
        with open(cache_fn, 'rb') as fl:
            obs, obs_meta, next_action_cheese, next_action_corner = pickle.load(fl)

    # Run observations through a hooked network, extract different activations.
    # Do it batches to avoid running out of RAM!
    print('Run observations through hooked network, in batches...')
    print('HACK!  Skip the first 10k obs to make everything smaller :(')
    num_skip = 15000
    value_labels = args.value_labels
    for value_label in tqdm(value_labels):
        value_list = []
        for batch_start_ind in tqdm(range(num_skip, obs.shape[0], hook_batch_size)):
        #for batch_start_ind in tqdm(range(0, 1000, hook_batch_size)):
            hook.run_with_input(obs[batch_start_ind:(batch_start_ind+hook_batch_size)], 
                values_to_store=[value_label])
            value_list.append(hook.get_value_by_label(value_label))
        value_np = np.concatenate(value_list, axis=0)
        del value_list
        with open(os.path.join(args.save_dr, f'{value_label}.pkl'), 'wb') as fl:
            pickle.dump(value_np, fl)
        del value_np
        


