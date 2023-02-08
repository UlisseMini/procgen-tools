# %%
# Imports
%reload_ext autoreload
%autoreload 2

import os
import pickle

import numpy as np
import pandas as pd
import scipy as sp
import torch as t
import xarray as xr
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm.auto import tqdm
from einops import rearrange
from IPython.display import Video, display, clear_output

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr

import procgen_tools.models as models
import procgen_tools.maze as maze
import gatherdata
import gatherdata_rich

# %%
# Functions
def get_node_probe_targets(data_all, world_loc):
    node_data = [maze.get_node_type_by_world_loc(dd['dec_state_bytes'], world_loc)
        for dd in data_all]
    node_types = [tt for tt, ngb in node_data]
    node_ngb = [ngb for tt, ngb in node_data]
    return xr.Dataset(dict(
        node_type = xr.DataArray(node_types),
        node_ngb = xr.DataArray(node_ngb),
    ))



# %%
# Load postprocessed data and convert to required form
# Load data as list of dicts
dr = '../episode_data/20230131T224127/' # 10k run
with open(os.path.join(dr, 'postproc_probe_data.pkl'), 'rb') as fl:
    data_all = pickle.load(fl)['data']
# Pull out the observations into a single batch
batch_coords = np.arange(len(data_all))
obs_all = xr.concat([dd['obs'] for dd in data_all], 
    dim='batch').assign_coords(dict(batch=batch_coords))
# Pull out / create probe targets of interest
probe_targets = xr.Dataset(dict(
    did_get_cheese = xr.DataArray([dd['did_get_cheese'] for dd in data_all]),
))
probe_targets = probe_targets.merge(get_node_probe_targets(data_all, 
    (12, 12))).rename({'dim_0': 'batch'}).assign_coords(dict(batch=batch_coords))


# %%
# Set up model and hook it
model_file = '../trained_models/maze_I/model_rand_region_5.pth'
policy = models.load_policy(model_file, action_size=15, device=t.device('cpu'))
model_name = os.path.basename(model_file)
hook = cmh.ModuleHook(policy)

    
# %%
# Run obs through model to get all the activations
num_batch = 1000
obs_sub = obs_all[:num_batch]
hook.run_with_input(obs_sub)

probe_targets_sub = probe_targets.isel(batch=slice(num_batch))


# %% 
# Try the circrl probing
y = probe_targets_sub['node_ngb'][:,0].values
probe_results = cpr.run_probe(hook, y,
    value_labels = ['embedder.block1.res1.resadd_out',
                    'embedder.block1.res2.resadd_out',
                    'embedder.block2.res1.resadd_out',
                    'embedder.block2.res2.resadd_out',
                    'embedder.block3.res1.resadd_out',
                    'embedder.block3.res2.resadd_out'],
    value_nums_to_use = np.array([1, 5, 10, 20, 50]))


# %%
# Select a layer to probe over and do some fitting!
# value_labels = ['embedder.flatten_out', 'embedder.relufc_out',
#   'embedder.block2.res2.resadd_out']
value_label = 'embedder.block1.res2.resadd_out'
#value_label = 'embedder.relufc_out'
value = hook.get_value_by_label(value_label)

X = rearrange(value.values, 'b ... -> b (...)')
# cat_values, y = np.unique(probe_targets_sub['node_type'].values, 
    # return_inverse=True)
y = probe_targets_sub['node_ngb'][:,0].values
scaler = StandardScaler()
X_scl = scaler.fit_transform(X)

def f_classif_fixed(X, y, **kwargs):
    '''Handle columns with zero variance, hackily'''
    X_fixed = X
    X_fixed[0,:] += 1e-6
    return f_classif(X_fixed, y, **kwargs)

D_act = X.shape[1]
K = 5 #int(0.01*D_act)
num_seeds = 5
f_test_checks = []
sort_inds_train_all = np.zeros((num_seeds, D_act), dtype=int)
ranks_train_all = np.zeros((num_seeds, D_act), dtype=int)
f_test_train_all = np.zeros((num_seeds, D_act))
for ii, random_state in enumerate(np.arange(num_seeds)):

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_scl, y, 
        test_size=0.2, random_state=random_state)
    
    # Get f-statisitcs
    f_test_train, _ = f_classif_fixed(X_train, y_train)
    f_test_test, _ = f_classif_fixed(X_test, y_test)
    #px.scatter(x=f_test_train, y=f_test_test, opacity=0.3).show()

    # Ranking
    def get_ranks(x):
        sort_inds = x.argsort()
        ranks = np.empty_like(sort_inds)
        ranks[sort_inds] = np.arange(len(x))
        return ranks
    ranks_train = get_ranks(f_test_train)
    ranks_test = get_ranks(f_test_test)
    sort_inds_train = f_test_train.argsort()       

    sort_inds_train_all[ii,:] = sort_inds_train
    f_test_train_all[ii,:] = f_test_train
    ranks_train_all[ii,:] = ranks_train

    top_K_inds_train = sort_inds_train[-K:]

    # Try training classifier on top-K by f_test
    X_top = X_train[:,top_K_inds_train]
    # clf = LogisticRegression(random_state=0, multi_class='ovr', 
    #     solver='liblinear')
    clf = LogisticRegression(random_state=0)
    clf.fit(X_top, y_train)
    y_pred_train = clf.predict(X_top)
    y_pred_test = clf.predict(X_test[:,top_K_inds_train])
    def accur(y1, y2):
        return (y1==y2).mean()
    print(top_K_inds_train)
    print('Train accurary: {:.3f}'.format(accur(y_pred_train, y_train)))
    print('Test accurary: {:.3f}'.format(accur(y_pred_test, y_test)))
    print('Test baseline: {:.3f}'.format(y_test.mean()))
    print()

    f_test_checks.append({
        f'mean_traintopK_ftest_train': f_test_train[top_K_inds_train].mean(),
        f'mean_traintopK_ftest_test': f_test_test[top_K_inds_train].mean(),
        f'mean_ftest_test': f_test_test.mean(),
        f'std_ftest_test': f_test_test.std(),
        f'mean_traintopK_rank_train': ranks_train[top_K_inds_train].mean(),
        f'mean_traintopK_rank_test': ranks_test[top_K_inds_train].mean(),
        f'mean_rank_test': ranks_test.mean(),
        f'std_rank_test': ranks_test.std(),
    })
    #px.scatter(x=ranks_train[top_K_inds_train], y=ranks_test[top_K_inds_train], 
    #    opacity=0.3).show()
    
f_test_checks = pd.DataFrame(f_test_checks)
display(f_test_checks)

z_mean_ftest_test = (f_test_checks['mean_traintopK_ftest_test'] - 
    f_test_checks['mean_ftest_test']) / \
        (f_test_checks['std_ftest_test']/np.sqrt(K))
p_mean_ftest_test = sp.stats.norm.sf(z_mean_ftest_test)
display(p_mean_ftest_test)

# f_test_train_mean = f_test_train_all.mean(axis=0)
# sort_inds_train_mean = f_test_train_mean.argsort()
# f_test_train_all_sorted = f_test_train_all[:,sort_inds_train_mean]

# fig = go.Figure()
# for ii in range(f_test_train_all_sorted.shape[0]):
#     fig.add_trace(go.Scatter(y=f_test_train_all_sorted[ii,-K:]))
# fig.add_trace(go.Scatter(y=f_test_train_mean[sort_inds_train_mean][-K:]))
# fig.show()

# Find the indices with the maximin rank?
maximin_ranks_train = ranks_train_all.min(axis=0).argsort()

# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# %%
# Test some of these rankings
if __name__ == "__main__":
    for jj in range(1,5):
        #act_ind = maximin_ranks_train[-jj]
        act_ind = sort_inds_train_all[-1,-jj]
        X_try = X[:,act_ind]
        df = pd.DataFrame({'X_try': X_try, 'y': y})
        px.histogram(df, x='X_try', color='y', 
            title='{}: {}'.format(value_label, act_ind)).show()

# %% 
# A couple specific activations
if __name__ == "__main__":
    # act_inds = [1873, 24557]
    # for act_ind in act_inds:
    #     px.histogram(pd.DataFrame({'X_try': X[:,act_ind], 'y': y}), 
    #         x='X_try', color='y', 
    #         title='{}: {}'.format(value_label, act_ind)).show()
    # px.scatter(x=X_scl[:,act_inds[0]], y=X_scl[:,act_inds[1]], color=y,
    #     opacity=0.2).show()

    act_ind = 24557
    likely_true_range = [-0.38, -0.37]

    # px.histogram(pd.DataFrame({'X_try': X[:,act_ind], 'y': y}), 
    #         x='X_try', color='y', nbins=100, 
    #         title='{}: {}'.format(value_label, act_ind)).show()

    ch, row, col = np.unravel_index(act_ind, value.shape[1:])
    v2o_scl = obs_all.shape[2] / value.shape[2]
    obs_row, obs_col = np.array([row, col])*v2o_scl
    
    bis = (y & (X[:,act_ind] >= likely_true_range[0]) &
        (X[:,act_ind] < likely_true_range[1])).nonzero()[0]

    for bi in [9]: #[bis[4]]: #bis[:1]:
        print(y[bi])
        print((X[bi,act_ind] >= likely_true_range[0]) &
            (X[bi,act_ind] < likely_true_range[1]))
        fig = px.imshow(rearrange(obs_all[bi,:,:,:].values, 'c h w -> h w c'))
        fig.add_shape(type="rect",
            x0=obs_col, y0=obs_row, 
            x1=obs_col+v2o_scl, y1=obs_row+v2o_scl,
            line=dict(color="RoyalBlue"))
        fig.show()
        





