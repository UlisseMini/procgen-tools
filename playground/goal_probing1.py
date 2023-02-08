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

import procgen_tools.models as models
import procgen_tools.maze as maze
import gatherdata
import gatherdata_rich

# %%
# Functions

class NoDecisionSquareException(Exception):
    pass

class NotReachedDecisionSquareException(Exception):
    pass

def process_rollout(fn):
    '''Extract quantities of interest from a rollout file:
        - location of decision square
        - termination status of episode (just cheese or not for now)
        - observations at decision square'''
    # Load the data
    episode_data = cro.load_saved_rollout(fn)
    seq = episode_data['seq']
    # Get the decision square location
    maze_env_state = maze.EnvState(seq.custom['state_bytes'][0].values[()])
    inner_grid = maze_env_state.inner_grid()
    grid_graph = maze.maze_grid_to_graph(inner_grid)
    #px.imshow(rearrange(episode_data['seq'].obs[0].values, 'c h w -> h w c')).show()
    if not maze.grid_graph_has_decision_square(inner_grid, grid_graph):
        raise NoDecisionSquareException
    dec_node = maze.get_decision_square_from_grid_graph(inner_grid, grid_graph)
    # Get the decision square timestep
    dec_step = None
    for step in seq.obs.coords['step']:
        mst = maze.EnvState(seq.custom['state_bytes'][step].values[()])
        if maze.get_mouse_pos(mst.inner_grid()) == dec_node:
            dec_step = step
    if dec_step is None:
        raise NotReachedDecisionSquareException
    # Probe the network
    return dict(
        dec_node = dec_node,
        did_get_cheese = episode_data['seq'].rewards[-1].values[()]>0.,
        obs = seq.obs.sel(step=dec_step).astype(np.float32),
    )

def f_classif_fixed(X, y, **kwargs):
    '''Handle columns with zero variance, hackily'''
    X_fixed = X
    X_fixed[0,:] += 1e-6
    return f_classif(X_fixed, y, **kwargs)


# %%
# Set up some stuff
if __name__ == "__main__":
    model_file = '../trained_models/maze_I/model_rand_region_5.pth'
    policy = models.load_policy(model_file, action_size=15, device=t.device('cpu'))
    model_name = os.path.basename(model_file)
    
# %%
# Load some data
if __name__ == "__main__":
    dr = '../episode_data/20230130T204732' # Big run, but some don't have dec squares
    #dr = '../episode_data/20230130T221407' # Short test run
    
    data_all = []
    fns = glob.glob(os.path.join(dr, '*.dat'))
    for fn in tqdm(fns[:700]):
        try:
            data_all.append(process_rollout(fn))
        except (NoDecisionSquareException, NotReachedDecisionSquareException):
            pass #print(f'No decision square in rollout {ii}')

# %%
# Run obs through model
if __name__ == "__main__":
    batch_coords = np.arange(len(data_all))
    obs_all = xr.concat([dd['obs'] for dd in data_all], 
        dim='batch').assign_coords(dict(batch=batch_coords))
    hook = cmh.ModuleHook(policy)
    hook.run_with_input(obs_all)

# %%
# Select a layer to probe over and do some fitting!
if __name__ == "__main__":
    # value_labels = ['embedder.flatten_out', 'embedder.relufc_out',
    #   'embedder.block2.res2.resadd_out']
    #value_label = 'embedder.block2.res2.resadd_out'
    value_label = 'embedder.relufc_out'
    value = hook.get_value_by_label(value_label)
    did_get_cheese = xr.DataArray([dd['did_get_cheese'] for dd in data_all],
        dims=['batch'], coords=dict(batch=batch_coords))

    X = rearrange(value.values, 'b ... -> b (...)')
    y = did_get_cheese.values
    scaler = StandardScaler()
    X_scl = scaler.fit_transform(X)

    D_act = X.shape[1]
    K = 10 #int(0.01*D_act)
    num_seeds = 10
    f_test_checks = []
    sort_inds_train_all = np.zeros((num_seeds, D_act), dtype=int)
    ranks_train_all = np.zeros((num_seeds, D_act), dtype=int)
    f_test_train_all = np.zeros((num_seeds, D_act))
    for ii, random_state in enumerate(np.arange(num_seeds)):

        # Split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X_scl, y, 
            test_size=0.5, random_state=random_state)
        
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
        clf = LogisticRegression(random_state=0).fit(X_top, y_train)
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
        





