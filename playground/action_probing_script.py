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

if __name__ == "__main__":
    print('Parsing arguments...')
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print(os.getcwd())
    
    parser = ArgumentParser()
    parser.add_argument('--rand-region', type=int, default=5)
    parser.add_argument('--src-value', type=str, default='embedder.flatten_out')
    parser.add_argument('--num-levels', type=int, default=200, help='number of mazes to run test rollouts on')
    parser.add_argument('--seed', type=int, default=42, help='seed for various randomizations')
    parser.add_argument('--obs-fn', type=str, default='action_probing_obs.pkl', help='seed for various randomizations')
    utc = datetime.datetime.utcnow()
    dft_save_fn = utc.strftime('action_probing_res_%Y%m%dT%H%M%S.pkl')
    parser.add_argument('--save-fn', type=str, default=dft_save_fn, help='seed for various randomizations')

    args = parser.parse_args()
    print(args)

    # Generate or load a large batch of observations, run through hooked network to get fc activations,
    # cache these as dataset along with "next cheese action" and "next corner action".
    print('Generating / Loading data...')

    num_obs_normal = 25000
    num_obs_dec = 5000
    obs_batch_size = 5000

    hook_batch_size = 100
    value_labels = ['embedder.flatten_out', 'embedder.relufc_out']
    logits_value_label = 'fc_policy_out'

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

    with open(cache_fn, 'rb') as fl:
        try:
            obs, values_by_label, logits, next_action_cheese, next_action_corner = pickle.load(fl)
        except:
            obs, obs_meta, next_action_cheese, next_action_corner = pickle.load(fl)

    # Train a probe!
    value_label = args.src_value
    print(f'Training probe on {value_label}')

    value = values_by_label[value_label]

    probe_result = cpr.linear_probe(value, next_action_cheese, 
        model_type='classifier', C=0.01, class_weight='balanced', test_size=0.3, random_state=42)

    model = probe_result['model']

    print(probe_result['train_score'], probe_result['test_score'])
    print(probe_result['conf_matrix'])

    # Test on a bunch of levels
    print(f'Testing trained probe on {args.num_levels} levels...')
    levels = range(args.num_levels)
    random_seed = args.seed
    rng = np.random.default_rng(random_seed)

    model_to_use = model

    def predict_cheese(obs, deterministic):
        obs = t.FloatTensor(obs)
        with hook.store_specific_values([value_label]):
            hook.network(obs)
            fc = hook.get_value_by_label(value_label)
        probs = np.squeeze(model_to_use.predict_proba(fc))
        if deterministic:
            act_sh_ind = probs.argmax()
        else:
            act_sh_ind = rng.choice(len(probs), 1, p=probs)
        act_sh = model_to_use.classes_[act_sh_ind][0]
        for act_name, inds in models.MAZE_ACTION_INDICES.items():
            if act_name[0] == act_sh:
                act = np.array([inds[0]])
                break
        return act, None

    predict_normal = rollout_utils.get_predict(policy)

    results = []
    for level in tqdm(levels):
        result = {'level': level}
        for desc, pred in {'normal': predict_normal, 'retrain': predict_cheese}.items():
            venv = maze.create_venv(1, start_level=level, num_levels=1)
            seq, _, _ = cro.run_rollout(pred, venv, max_episodes=1, max_steps=256, 
                                        show_pbar=False, seed=random_seed)
            result[f'{desc}_found_cheese'] = seq.rewards.sum().item() > 0.
        results.append(result)
    df = pd.DataFrame(results)
    print(df.mean())

    # Save results
    print('Saving results...')
    with open(args.save_fn, 'wb') as fl:
        pickle.dump((model, df, args), fl)

