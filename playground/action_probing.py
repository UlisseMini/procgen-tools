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
import procgen_tools.rollout_utils as rollout_utils
from procgen import ProcgenGym3Env

from action_probing_obsproc import load_value

path_prefix = '../'

# %%
# Load observations and related data

num_obs_normal = 25000
num_obs_dec = 5000
obs_batch_size = 5000

hook_batch_size = 100
value_labels = ['embedder.flatten_out', 'embedder.relufc_out']
logits_value_label = 'fc_policy_out'

cache_fn = '../episode_data/action_probing_current/action_probing_obs.pkl'
value_cache_dr = '../episode_data/action_probing_current'

rand_region = 5
policy = models.load_policy(path_prefix + 
        f'trained_models/maze_I/model_rand_region_{rand_region}.pth', 
    15, t.device('cpu'))
hook = cmh.ModuleHook(policy)

# A hack because storing activations in middle layers took too much RAM with full dataset
num_obs_to_ignore = 15000

with open(cache_fn, 'rb') as fl:
    obs, obs_meta, next_action_cheese, next_action_corner = pickle.load(fl)
    obs = obs[num_obs_to_ignore:]
    obs_meta = np.array(obs_meta[num_obs_to_ignore:])
    next_action_cheese = next_action_cheese[num_obs_to_ignore:]
    next_action_corner = next_action_corner[num_obs_to_ignore:]



# %%
# Train a probe!
#value_label = 'embedder.relufc_out'
value_label = 'embedder.flatten_out'

value = load_value(value_label, value_cache_dr)

probe_result = cpr.linear_probe(value, next_action_cheese, 
    model_type='classifier', C=0.005, class_weight='balanced', test_size=0.3, random_state=42)

model = probe_result['model']

print(probe_result['train_score'], probe_result['test_score'])
print(probe_result['conf_matrix'])

# # Load pre-saved probe and results
# results_fn = 'action_probing_res_20230306T230300.pkl'
# with open(results_fn, 'rb') as fl:
#     model, df, args = pickle.load(fl)
# df.mean()


# %%
# Look for patterns in observations where the next cheese action is assigned particularly low prob

# Get probabilities for all actions based on learned linear model
next_action_proba = model.predict_proba(value)

# Turn target into one-hot encoded version
actions, next_action_cheese_ind = np.unique(next_action_cheese, return_inverse=True)
assert (actions == model.classes_).all()
#next_action_cheese_1hot = np.eye(len(actions))[next_action_cheese_ind]

# Sort data points by prob of picking next cheese action, lowest being most interesting
cheese_action_proba = next_action_proba[np.arange(len(next_action_cheese_ind)),next_action_cheese_ind]
inds_worst_pred = np.argsort(cheese_action_proba)

# %%
# Take a look at a few of the worse obs!
num_worst = 16

worst_obs_da = xr.DataArray(rearrange(obs[inds_worst_pred[:num_worst],...], 'b c h w -> h w c b'),
    dims=['h', 'w', 'rgb', 'obs_index'], 
    #coords={'level': [md['level_seed'] for md in np.array(obs_meta)[inds_worst_pred[:num_worst]]]})
    coords={'obs_index': inds_worst_pred[:num_worst]})
fig = px.imshow(worst_obs_da, facet_col='obs_index',
    facet_col_wrap=4, facet_col_spacing=0.01, facet_row_spacing=0.01)
fig.update_layout(height=1200)
fig.show()


# %%
# Play with some specific levels
inds_to_check = [10441]

# Get level seed for reference
#level_seeds = obs_meta[index_to_check]['level_seed']

# Look at specific channels that we expect to code for cheese, how do they look?
value_label_to_check = 'embedder.block2.res1.resadd_out'
channels_to_check = [55, 42]
hook.run_with_input(obs[inds_to_check])
value_to_check = hook.get_value_by_label(value_label_to_check)


# %%
# Compare distributions of max values of cheese coding channels when the agent does pick the 
# cheese actions, vs when it doesn't.
value_label_to_check = 'embedder.block2.res1.resadd_out'
channels_to_check = [55, 42]

pred_action = model.predict(value)

value_to_check = load_value(value_label_to_check, value_cache_dr)
value_to_check_max = value_to_check.max(axis=-1).max(axis=-1)

chans_list = []
for ch in channels_to_check:
    chans_list.append(pd.DataFrame({'max_value': value_to_check_max[:,ch], 
        'did_pick_cheese_action': pred_action==next_action_cheese, 
        'channel': np.full(next_action_cheese.shape, ch),
        #'level_seed': np.concatenate([level_seeds, level_seeds]),
        }))
chans_df = pd.concat(chans_list, axis='index')
px.histogram(chans_df, title=f'{value_label} max values at "did pick cheese action" true/false',
    x='max_value', color='did_pick_cheese_action', opacity=0.5, 
    barmode='overlay', facet_col='channel', facet_col_wrap=2,
    histnorm='probability', marginal='box', 
    hover_data=list(chans_df.columns)).show()



# %%
# What about a more complex probe?
# from sklearn.ensemble import RandomForestClassifier

# X_train, X_test, y_train, y_test = train_test_split(
#     value[inds_slice], next_action_cheese[inds_slice], 
#     test_size=0.3, random_state=42)

# mdl = RandomForestClassifier(n_estimators=5, max_features='sqrt',
#     class_weight='balanced', random_state=1)
# mdl.fit(X_train, y_train)
# y_pred = mdl.predict(X_test)
# print(mdl.score(X_train, y_train), mdl.score(X_test, y_test))


# %%
# See how the probe compares with the actual best actions chosen by the real network logits
logits_argmax = logits.argmax(axis=1)
next_action_logits = models.MAZE_ACTIONS_BY_INDEX[logits_argmax].astype('<U1')
logits_cheese_score = (next_action_logits == next_action_cheese).mean()
logits_corner_score = (next_action_logits == next_action_corner).mean()
print(logits_cheese_score, logits_corner_score)


# %%
# What about confirming we can learn a probe to the actual model actions?
probe_result_logits = cpr.linear_probe(value, next_action_logits, model_type='classifier', 
    C=0.01, test_size=0.3)
print(probe_result_logits['train_score'], probe_result_logits['test_score'])


# %%
# What about the actual model logits?
probe_result_logits = cpr.linear_probe(value, logits, model_type='ridge', 
    alpha=100., test_size=0.3)
print(probe_result_logits['train_score'], probe_result_logits['test_score'])


# %%
# Test an agent with the trained cheese-action weights?
# RESULT: so far, the resulting policy performs quite badly, which is suprising as
# it predicts the correct "next action towards cheese" better than the actual policy!
# I think this is worth some debugging...
# 2023-03-04: we can get 90% cheese-action accuracy learning a single linear map from the 
# flattened conv layer output, but even so it looks like we still get stuck sometimes?
# Level 2: it seems like it's *trying* to get the cheese, but it can't quite
# overcome the resistence to going up the cheese-holding branch.  I wonder if there's 
# actually something in the conv net that basically obscures the cheese from view when
# the mouse is at that junction square?
# Level 17: it gets stuck on the decision square, which is actually pretty similar to
# level 2... is there a pattern here?
# Level 5: new policy finds the cheese, old one doesn't, so that's interesting...
# Level 12: stuck near the decision square, normal doesn't get cheese
# Level 13: get's cheese, normal doesn't... 
# Test this more systematically in script

# levels = range(200)
random_seed = 42
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

# results = []
# for level in tqdm(levels):
#     result = {'level': level}
#     for desc, pred in {'normal': predict_normal, 'retrain': predict_cheese}.items():
#         venv = maze.create_venv(1, start_level=level, num_levels=1)
#         seq, _, _ = cro.run_rollout(pred, venv, max_episodes=1, max_steps=256, show_pbar=False)
#         result[f'{desc}_found_cheese'] = seq.rewards.sum().item() > 0.
#     results.append(result)
# df = pd.DataFrame(results)
# display(df)


# %%
# Pick a few levels to show side-by-side rollouts on
# vid_levels = [level for level in 
    # df[(df.normal_found_cheese==False) & (df.retrain_found_cheese==False)].iloc[:4].level]
vid_levels = [17]

vid, seqs = rollout_utils.side_by_side_rollout({
    'normal': predict_normal, 'cheese-retrained': predict_cheese}, vid_levels)
display(vid)


# %%
# What about a sparse probe?
# Seems to work pretty well!  Can get almost the same prediction accuracy (89%) with 1000 activtions, only 1/8th total available
#f_test, _ = cpr.f_classif_fixed(value, next_action_cheese)

index_nums = np.array([10, 100, 300, 600, 1000])
f_test, _ = cpr.f_classif_fixed(value, next_action_cheese)
sort_inds = np.argsort(f_test)[::-1]
scores_list = []
for K in tqdm(index_nums):
    results = cpr.linear_probe(value[:,sort_inds[:K]], next_action_cheese, C=0.01, 
                               class_weight='balanced', random_state=42)
    scores_list.append({'K': K, 'train_score': results['train_score'],
                        'test_score': results['test_score']})
scores_df = pd.DataFrame(scores_list)
scores_df

# %%
# What about using sparse channels?
# Okay this is pretty interesting!  Seems like 10-20 channels is enough to get pretty solid prediction performance on for
# next cheese action!  And also enough to get next action selected by the logits at 96% accuracy with only 15 channels?
chan_nums = np.array([5, 10, 15, 20])
f_test_full = rearrange(f_test, '(c h w) -> c h w', h=8, w=8)
value_full = rearrange(value, 'b (c h w) -> b c h w', c=128, h=8, w=8)
f_test_sum_by_chan = f_test_full.sum(axis=-1).sum(axis=-1)
chan_sort_inds = np.argsort(f_test_sum_by_chan)[::-1]
scores_list = []
for ch_num in tqdm(chan_nums):
    results = cpr.linear_probe(value_full[:,chan_sort_inds[:ch_num],:,:], next_action_cheese, C=0.01, 
                               class_weight='balanced', random_state=42)
    scores_list.append({'ch_num': ch_num, 'train_score': results['train_score'],
                        'test_score': results['test_score']})
scores_df = pd.DataFrame(scores_list)
display(scores_df)
model_to_use = results['model']

# %%
# Try some rollouts with a model that uses sparse channel model
levels = range(10)
random_seed = 42
rng = np.random.default_rng(random_seed)

def predict_cheese_sparse(obs, deterministic):
    obs = t.FloatTensor(obs)
    with hook.store_specific_values([value_label]):
        hook.network(obs)
        value_sparse = rearrange(
            rearrange(
                hook.get_value_by_label(value_label),
                'b (c h w) -> b c h w', c=128, h=8, w=8)[:,chan_sort_inds[:ch_num],:,:],
            'b c h w -> b (c h w)')
    probs = np.squeeze(model_to_use.predict_proba(value_sparse))
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
    for desc, pred in {'normal': predict_normal, 'retrain': predict_cheese_sparse}.items():
        venv = maze.create_venv(1, start_level=level, num_levels=1)
        seq, _, _ = cro.run_rollout(pred, venv, max_episodes=1, max_steps=256, show_pbar=True)
        result[f'{desc}_found_cheese'] = seq.rewards.sum().item() > 0.
    results.append(result)
df = pd.DataFrame(results)
display(df)

# %%
# Some specific levels with sparse retrained agent
vid_levels = [9]

vid, seqs = rollout_utils.side_by_side_rollout({
    'normal': predict_normal, 'cheese-retrained-sparse': predict_cheese_sparse}, vid_levels)
display(vid)



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
