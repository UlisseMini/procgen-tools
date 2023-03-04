# %%[markdown]
# # Training Probes to Predict Next Cheese Action
# 
# What would happen if we essentially re-trained the final fc-to-logits weights using a linear probe to predict next "cheese-direction" action?  Would this work?  
# 
# Start with the usual imports...


# %%
# Imports and initial setup
%reload_ext autoreload
%autoreload 2

from typing import List, Tuple, Dict, Union, Optional, Callable
import random
import itertools
import copy
import pickle
import os

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
import matplotlib.pyplot as plt 
plt.ioff() # disable interactive plotting, so that we can control where figures show up when refreshed by an ipywidget

import lovely_tensors as lt
lt.monkey_patch()

import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr
import procgen_tools.models as models
import procgen_tools.maze as maze
import procgen_tools.patch_utils as patch_utils
import procgen_tools.vfield as vfield
from procgen import ProcgenGym3Env

path_prefix = '../'

# %%
# Generate a large batch of observations, run through hooked network to get fc activations,
# cache these as dataset along with "next cheese action" and "next corner action".

num_obs_normal = 25000
num_obs_dec = 5000
obs_batch_size = 5000

hook_batch_size = 100
value_label = 'embedder.relufc_out'
logits_value_label = 'fc_policy_out'

REDO_OBS = False
cache_fn = 'action_probing_obs.pkl'

rand_region = 5
policy = models.load_policy(path_prefix + 
        f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 
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

    # Run observations through a hooked network, extract the fc layer activations 
    # as the training/test data.  Do it batches to avoid running out of RAM!
    print('Run observations through hooked network, in batches...')
    value_list = []
    logits_list = []
    for batch_start_ind in tqdm(range(0, obs.shape[0], hook_batch_size)):
        hook.run_with_input(obs[batch_start_ind:(batch_start_ind+hook_batch_size)], 
            values_to_store=[value_label, logits_value_label])
        value_list.append(hook.get_value_by_label(value_label))
        logits_list.append(hook.get_value_by_label(logits_value_label))
    value = np.concatenate(value_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)
    
    with open(cache_fn, 'wb') as fl:
        pickle.dump((obs, value, logits, next_action_cheese, next_action_corner), fl)

else:
    with open(cache_fn, 'rb') as fl:
        obs, value, logits, next_action_cheese, next_action_corner = pickle.load(fl)


# %%
# Train a probe!
inds_slice = slice(None)
probe_result = cpr.linear_probe(value[inds_slice], next_action_cheese[inds_slice], 
    model_type='classifier', C=0.01, class_weight='balanced', test_size=0.3)

model = probe_result['model']

print(probe_result['train_score'], probe_result['test_score'])
print(probe_result['conf_matrix'])

# %%
# What about a more complex probe?
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    value[inds_slice], next_action_cheese[inds_slice], 
    test_size=0.3, random_state=42)

mdl = RandomForestClassifier(n_estimators=5, max_features='sqrt',
    class_weight='balanced', random_state=1)
mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test)
print(mdl.score(X_train, y_train), mdl.score(X_test, y_test))





# %%
# See how the probe compares with the actual best actions chosen by the real network logits
logits_argmax = logits.argmax(axis=1)
next_action_logits = models.MAZE_ACTIONS_BY_INDEX[logits_argmax].astype('<U1')
logits_cheese_score = (next_action_logits == next_action_cheese).mean()
logits_corner_score = (next_action_logits == next_action_corner).mean()
print(logits_cheese_score, logits_corner_score)

# What about confirming we can learn a probe to the actual logits??
probe_result_logits = cpr.linear_probe(value, next_action_logits, model_type='classifier', 
    C=1., test_size=0.3)
print(probe_result_logits['train_score'], probe_result_logits['test_score'])
#print(probe_result_logits['conf_matrix'])
model_logits = probe_result_logits['model']



# %%
# Test an agent with the trained cheese-action weights?
# RESULT: so far, the resulting policy performs quite badly, which is suprising as
# it predicts the correct "next action towards cheese" better than the actual policy!
# I think this is worth some debugging...

level = 17
random_seed = 42
rng = np.random.default_rng(random_seed)

model_to_use = model

def predict(obs, deterministic):
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

venv = maze.create_venv(1, start_level=level, num_levels=1)
seq, _, _ = cro.run_rollout(predict, venv, max_episodes=1, max_steps=256)
vid_fn, fps = cro.make_video_from_renders(seq.renders)
display(Video(vid_fn, embed=True))

# %%
# Try comparing vfields

level = 17

def forward(obs):
    obs = t.FloatTensor(obs)
    with hook.store_specific_values([value_label]):
        hook.network(obs)
        fc = hook.get_value_by_label(value_label)
    probs_sh = np.squeeze(model.predict_proba(fc))
    probs = np.zeros((probs_sh.shape[0], 15))
    for ind_sh, act_sh in enumerate(model.classes_):
        for act_name, inds in models.MAZE_ACTION_INDICES.items():
            if act_name[0] == act_sh:
                probs[:,inds[0]] = probs_sh[:,ind_sh]
    p = Categorical(probs=t.from_numpy(probs))
    return p, None

# Vfield
print(f'Level: {level}')
venv = maze.create_venv(1, start_level=level, num_levels=1)
vf_original = vfield.vector_field(venv, policy)
policy_hacked = copy.deepcopy(policy)
policy_hacked.forward = forward
vf_patched = vfield.vector_field(venv, policy_hacked)
vfield.plot_vfs(vf_original, vf_patched)
plt.show()

# %%
# Get a batch of observations for this specific level, test the model on them
venv = maze.create_venv(1, start_level=level, num_levels=1)
state_bytes = venv.env.callmethod('get_state')[0]
env_state = maze.EnvState(state_bytes)
grid = env_state.inner_grid(with_mouse=False)
legal_mouse_positions = maze.get_legal_mouse_positions(grid)

obs_list = []
obs_meta_this_level = []
for mouse_pos in tqdm(legal_mouse_positions):
    obs, obs_meta, _ = maze.get_random_obs_opts(1, start_level=level, 
        mouse_pos_inner=mouse_pos, deterministic_levels=True, return_metadata=True)
    obs_list.append(obs)
    obs_meta_this_level.append(obs_meta[0])

obs_this_level = np.concatenate(obs_list, axis=0)
next_action_cheese_this_level = np.array([get_action(md['mouse_pos_outer'], 
            md['next_pos_cheese_outer'])
        for md in obs_meta_this_level])

hook.run_with_input(obs_this_level, 
    values_to_store=[value_label, logits_value_label])
value_this_level = hook.get_value_by_label(value_label)
logits_this_level = hook.get_value_by_label(logits_value_label)

next_action_cheese_this_level_pred = model.predict(value_this_level)

(next_action_cheese_this_level_pred == next_action_cheese_this_level).mean()

# %%
# Try fine-tuning a version of the actual model to be a bit more cheese-seeking

# TODO: not working for some reason, need to debug!

minibatch_size = 100
random_seed = 10
test_size = 0.3

input = t.from_numpy(value)
input.requires_grad = True
target = Categorical(F.log_softmax(t.from_numpy(logits), dim=1)).probs
target.requires_grad = True

# Simple network to train from FC relu outputs to logits
class FcNet(nn.Module):
    def __init__(self, init_policy):
        super().__init__()
        self.fc_policy = nn.Linear(256, 15)
        with t.no_grad():
            self.fc_policy.weight.copy_(init_policy.fc_policy.weight)

    def forward(self, x):
        logits = self.fc_policy(x)
        return logits
        # log_probs = F.log_softmax(logits, dim=1)
        # p = Categorical(logits=log_probs)
        # return p

class SimpleDataset(t.utils.data.Dataset):
    def __init__(self, input, target):
        super(SimpleDataset, self).__init__()
        assert input.shape[0] == target.shape[0]
        self.input = input
        self.target = target
    def __len__(self):
        return self.target.shape[0]
    def __getitem__(self, index):
        return self.input[index], self.target[index]


new_fc = FcNet(policy)
dataset = SimpleDataset(input, target)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(new_fc.parameters(), lr=0.001, momentum=0.9)
rng = np.random.default_rng(random_seed)

train_indices, test_indices, _, _ = train_test_split(range(len(dataset)),
    dataset.target, test_size=test_size, random_state=random_seed)

# generate subset based on indices
train_split = t.utils.data.Subset(dataset, train_indices)
test_split = t.utils.data.Subset(dataset, test_indices)

# create batches
train_batches = t.utils.data.DataLoader(train_split, batch_size=minibatch_size, shuffle=True)
test_batches = t.utils.data.DataLoader(test_split, batch_size=minibatch_size)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(train_batches, 0):
        # get the inputs; data is a list of [inputs, labels]
        input_this, target_this = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = new_fc(input_this)
        loss = criterion(target_this, target_this)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every X mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')
