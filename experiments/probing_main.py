# %%[markdown]
# ## Sparse Linear Probing of Agent Model
#
# ### Summary of Results So Far
# - Sparse linear probes can be applied to this network, and can predict quantities assocaited with simple abstractions with high accuracy, including square open/closed state, cheese location, and mouse location.
# - Probe accuracies across layers provide some evidence that more complex representations are encoded in later layers.
# - A new probing technique is introduced called "convolutional probing", which results in much higher accuracy when predicting spatially-located targets (e.g. object position) vs standard linear probing.
# - Probing directly for "mouse did find cheese" is surprisingly accurate: able to achieve 78-85% accuracy depending on the layer, on a baseline "did find cheese" probability of 68%.
# - The information used by "mouse did find cheese" probes varies by layer: in the first IMPALA block, the probes are clearly learning a simple heuristic that mazes larger than 19x19 are much less likely to result in cheese finding. By contrast, in the third IMPALA block (and the FC layer), probes are learning something very different (and also getting higher accuracy).  I haven't tried to figure out what they are detecting here yet but I think it could be interesting.
#
# ### Intro to Sparse Linear Probing
#
# Probing is an interpretability technique whereby small models are trained in a supervised manner to predict some target variable from activations taken from inside the model being interpreted, over some dataset. The intent is to demonstrate that a given layer's activations contain a representation of some abstraction or concept by showing that a probe can be trained on these activations to recover a quantity associated with this representation with high accuracy.  Probing has been used extensively on language models to look for evidence that e.g. parts-of-speech, numbers, and other linguistic abstractions are learned and used.  
#
# Linear probing limits the structure of the learned probe model to a linear model.  The main benefits of linear probing vs allowing more complex probes are simplicity, speed of training, and resistance to overfitting.  This latter benefit is significant, as there exists evidence that more complex probes (e.g. MLPs) are sometimes able to simply "learn the task" themselves, resulting in misleading false-positives regarding the presence of a given representation in the actual model being studied.  The obvious drawback to linear probes is that they are unable to detect abstractions that are nonlinearly embedded.
#
# Sparse probing is an extension of the technique that attempts to train probes on small subsets of the activation in a given layer; when training is successful, it provides evidence not only that a layer represents a certain target, but that this target can be reconstructed using only a small number of neurons.  This alone does not demonstrate that the representation is highly localized, as it may be possible to train accurate probes for a certain target on many subsets of activations; however, sparse probes can show that small numbers of activations are *sufficient*, which is often highly useful in directing further interpretation efforts.
#
# Let's apply sparse linear probing to the maze agent from the goal misgeneralization paper and see if we can learn anything useful.  Fist, imports and all that...
#
# TODO: add links and references to this section

# %%
# Imports
%reload_ext autoreload
%autoreload 2

import os
import pickle

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_selection import f_classif
import torch as t
import torch.nn.functional as f
import xarray as xr
import plotly.express as px
import plotly as py
import plotly.subplots
import plotly.graph_objects as go
from einops import rearrange, repeat
from IPython.display import Video, display
from tqdm.auto import tqdm
import warnings

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr

import procgen_tools.models as models
import procgen_tools.maze as maze

warnings.filterwarnings("ignore", message=r'.*labels with no predicted samples.*')

# %%[markdown]
# Load the dataset we'll be working with, load and hook the model, run the batch of observations through a forward pass of the model...

# %%
# Setup / params

# Values to store activations for during forward pass over obs batch

# All conv layers, including obs
# values_to_store = [
#     'embedder.block1.conv_in0',
#     'embedder.block1.conv_out',
#     'embedder.block1.maxpool_out',
#     'embedder.block1.res1.relu1_out',
#     'embedder.block1.res1.conv1_out',
#     'embedder.block1.res1.relu2_out',
#     'embedder.block1.res1.conv2_out',
#     'embedder.block1.res1.resadd_out',
#     'embedder.block1.res2.relu1_out',
#     'embedder.block1.res2.conv1_out',
#     'embedder.block1.res2.relu2_out',
#     'embedder.block1.res2.conv2_out',
#     'embedder.block1.res2.resadd_out',
#     'embedder.block2.conv_out',
#     'embedder.block2.maxpool_out',
#     'embedder.block2.res1.relu1_out',
#     'embedder.block2.res1.conv1_out',
#     'embedder.block2.res1.relu2_out',
#     'embedder.block2.res1.conv2_out',
#     'embedder.block2.res1.resadd_out',
#     'embedder.block2.res2.relu1_out',
#     'embedder.block2.res2.conv1_out',
#     'embedder.block2.res2.relu2_out',
#     'embedder.block2.res2.conv2_out',
#     'embedder.block2.res2.resadd_out',
#     'embedder.block3.conv_out',
#     'embedder.block3.maxpool_out',
#     'embedder.block3.res1.relu1_out',
#     'embedder.block3.res1.conv1_out',
#     'embedder.block3.res1.relu2_out',
#     'embedder.block3.res1.conv2_out',
#     'embedder.block3.res1.resadd_out',
#     'embedder.block3.res2.relu1_out',
#     'embedder.block3.res2.conv1_out',
#     'embedder.block3.res2.relu2_out',
#     'embedder.block3.res2.conv2_out',
#     'embedder.block3.res2.resadd_out',
#     'embedder.relu3_out',
# ]

# Only input, resadd outputs, final conv layer flatten out, and FC + logits
values_to_store = [
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

# Most studied
# values_to_store = [
#     'embedder.block2.res1.resadd_out'
# ]

# This limits the number of mazes to look for faster exploration
num_batch = 1000

# 10k run, only mazes with dec square, obs saved on dec square
dr = '../episode_data/20230131T224127/'
fn = 'postproc_probe_data.pkl'
state_bytes_key = 'dec_state_bytes'

# Load data from post-processed pickled file
with open(os.path.join(dr, fn), 'rb') as fl:
    data_all = pickle.load(fl)['data']
    num_batch_to_use = min(num_batch, len(data_all))
    data_all = data_all[:num_batch_to_use]

# Pull out the observations into a single batch
batch_coords = np.arange(len(data_all))
obs_all = xr.concat([dd['obs'] for dd in data_all], 
    dim='batch').assign_coords(dict(batch=batch_coords))

# Pull out level seeds for later reference
level_seeds = np.array([maze.EnvState(dd[state_bytes_key]).state_vals['current_level_seed'].val for
    dd in data_all])

# Pull out the maze dim for later reference
maze_dims = np.array([maze.EnvState(dd[state_bytes_key]).state_vals['maze_dim'].val for
    dd in data_all])

# Set up model and hook it
model_file = '../trained_models/maze_I/model_rand_region_5.pth'
policy = models.load_policy(model_file, action_size=15, device=t.device('cpu'))
model_name = os.path.basename(model_file)
hook = cmh.ModuleHook(policy)

# Run obs through model to get all the activations
_ = hook.run_with_input(obs_all, values_to_store=values_to_store)


# %%[markdown]
# ### Feature Selection
# 
# To train a sparse probe, a feature selection algorithm is required.  Many exist, with various optimality guarantees (TODO: include links).  So far we have exclusively used the "f-test" metric, due to it's speed, simplicity and suitability for feature selection for linear models (TODO: link).
#
# ### Simple Example: Probing for Open/Closed Maze Square
#
# Possibly the simplest abstraction that we could probe for is the open/closed status of a single specific maze square.  In this case, we choose the square at absolute position (12, 13), since this is guaranteed to exist in mazes of every legal size, and is a "wall" square which will be open/closed with ~50% probability given the operation of the procgen maze generation algorithm (TODO: is this described somewhere?)
#
# Here we will demonstrate the feature selection approach, and then show the performance of sparse probes of various K-values (i.e. number of activations included in probe) over various layers throughout the network.
#
# First, generate the probe target as a bool variable "is open" for the chosen square...

# %%
square_to_probe = (12, 13)
square_is_open = []
for dd in tqdm(data_all):
    square_is_open.append(maze.EnvState(dd[state_bytes_key]).full_grid()[square_to_probe] != maze.BLOCKED)
square_is_open = np.array(square_is_open, dtype=bool)
    
# %%[markdown]
# Then, demonstrate the f-test as a feature selection metric by showing an example (grayscaled) maze observation with the f-score for each pixel of the observation over the full dataset overlaid in red.  We'd expect to see a blob of red over the (12, 13) square...

# %%
f_scores, p_values = f_classif(rearrange(obs_all.values, 'b c h w -> b (c h w)'), square_is_open)
def norm01(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
f_scores_max_over_clrs = norm01(rearrange(f_scores, '(c h w) -> h w c', c=3, w=obs_all.shape[2]).max(axis=-1))
obs_with_f_score = repeat(obs_all[0].values.mean(axis=0), 'h w -> h w c', c=3)*0.5
obs_with_f_score[:,:,0] += f_scores_max_over_clrs*0.5
px.imshow(obs_with_f_score, title=f'Example observation with {square_to_probe} "is open"<br>f-score for each pixel across dataset shown in red').show()

# %%[markdown]
# We can then demonstrate training a sparse linear probe with a range of K-values to predict the open/closed state of this specific square from a subset of pixels in the observation selected using this f-score metric.  The resulting plots demonstrate perfect accuracy once we include at least 4 pixels, which is roughly what we'd expect given the clear legibility of the square open/closed state from the input pixels.  (Note that the sparse probing code includes a train/test split, with the score being calculated as the fraction of correct predictions on the test set.)

# %%
index_nums = np.array([1, 2, 3, 4, 5])
results, _ = cpr.sparse_linear_probe(hook, ['embedder.block1.conv_in0'], square_is_open, 
    index_nums=index_nums, random_state=42, class_weight='balanced', max_iter=1000, C=10.)
px.line(x=index_nums, y=results.score.isel(value_label=0)).show()

# %%[markdown]
# Things get more interesting when we apply the same process to probing for square open/closed on the activations of various layers within the network.  We are constraining ourselves to looking at the residual-add outputs for simplicity (plus the observation), but probes can be trained on any layer.

# %%
value_labels_to_plot = [
    'embedder.block1.conv_in0',
    'embedder.block1.res1.resadd_out',
    'embedder.block1.res2.resadd_out',
    'embedder.block2.res1.resadd_out',
    'embedder.block2.res2.resadd_out',
    'embedder.block3.res1.resadd_out',
    'embedder.block3.res2.resadd_out']

index_nums = np.array([1, 10, 50, 100, 1000])

results, _ = cpr.sparse_linear_probe(hook, value_labels_to_plot, square_is_open,
    index_nums = index_nums, random_state=42, class_weight='balanced', max_iter=10000, C=10.)

def plot_sparse_probe_scores(results, y, index_nums, title, include_limits=True):
    scores_df = results.score.to_dataframe().reset_index()
    scores_df['K'] = index_nums[scores_df['index_num_step']]
    fig = px.line(scores_df, x='value_label', y='score', color='K', title=title)
    if include_limits:
        fig.add_hline(y=1., line_dash="dot", annotation_text="perfect", annotation_position="bottom right")
        baseline_score = abs(y.mean()-0.5) + 0.5
        fig.add_hline(y=baseline_score, line_dash="dot", annotation_text="baseline", 
                annotation_position="bottom right")
    fig.show()

plot_sparse_probe_scores(results, square_is_open, index_nums, 
    f'Probe score over layers and K-values for {square_to_probe} "is open"')

# %%[markdown]
# Here we can already see a somewhat interesting pattern: by probing the observation, or inside the first IMPALA block, we can perfectly reconstruct the open/closed value of a specific square with only a handlful of activations.  In the second IMPALA block, this accuracy is significantly reduced, and peak accuracy requires many more activations to be included.  By the third IMPALA block, linear probes cannot recover the open/closed state of this specific square much better than chance, even with 1000 activations included.
#
# This is consistent with a natural hypothesis that as we progress deeper into the network, more complex abstractions are being represented.  We hypothesize that the maxpool layers between the IMPALA blocks are key structural drivers of this phenomenon.
#
# ## Probing for Cheese and Mouse Location
#
# Two more "natural abstractions" that we hypothesize the model uses to solve mazes are the location of the cheese, and the location of the mouse/agent.  There a few ways we can think of representing probe targets for these specific quantities:
# 1. As a boolean target similar to the open/closed square that is true if the cheese is located at a specific maze square, false if not.
# 2. As a pair of continuous variables representing object x/y location, which can be probed for using a sparse regression rather than a classfier.
# 3. As a boolean target for a "convolutional probe" (see below for details).
#
# Let's start with the simple boolean target, probing for cheese at the most common cheese location in our dataset...

# %%
# Get the probe target for cheese location
def get_obj_loc_targets(data_all, obj_value, state_bytes_key):
    '''Get potential probe targets for y,x (row, col) location of an
    object, where the object is specified by obj_value as the value
    to match on in the maze grid array.'''
    pos_arr = maze.get_object_pos_from_seq_of_states(
        [dd[state_bytes_key] for dd in data_all], obj_value)
    pos = xr.Dataset({
        'y': xr.DataArray(pos_arr[:,0], dims=['batch']),
        'x': xr.DataArray(pos_arr[:,1], dims=['batch'])}).assign_coords(
            {'batch': np.arange(len(data_all))})
    return pos

# Get the cheese position for all data points
cheese_pos = get_obj_loc_targets(data_all, maze.CHEESE, state_bytes_key)

# %%
# Get the most common cheese position and make a probe target out of this
unique_cheese_pos, cheese_pos_counts = np.unique(cheese_pos.to_array().transpose('batch',...), axis=0, return_counts=True)
modal_cheese_pos = unique_cheese_pos[cheese_pos_counts.argmax(),:]
cheese_pos_is_modal = (cheese_pos.y.values == modal_cheese_pos[0]) & (cheese_pos.x.values == modal_cheese_pos[1])

# Probe for this target and show results
index_nums = np.array([1, 5, 10, 25, 100])

results, _ = cpr.sparse_linear_probe(hook, values_to_store, cheese_pos_is_modal,
    index_nums = index_nums, random_state=42, class_weight='balanced', max_iter=10000, C=10)

plot_sparse_probe_scores(results, cheese_pos_is_modal, index_nums, 
    f'Probe score over layers and K-values for {tuple(modal_cheese_pos)} has cheese')

# %%[markdown]
# While this plot shows some ability to exceed baseline accuracy using enough activations early in the network, the limitation to only probe for cheese in a single specific location results in poor data utilization and unbalanced classification classes.  Perhaps we'd do better if we probed for the actual x/y location of the cheese as a continuous variable?  Let's try cheese x position first.

# %%
cheese_pos_x = cheese_pos.x.values.astype(float)

# Probe for this target and show results
index_nums = np.array([1, 10, 100, 1000])

results, _ = cpr.sparse_linear_probe(hook, values_to_store, cheese_pos_x, model_type='ridge',
    index_nums = index_nums, random_state=42, max_iter=10000, alpha=100.)

plot_sparse_probe_scores(results, cheese_pos_x, index_nums, 
    f'Probe score over layers and K-values for cheese x-position',
    include_limits=False)

# %%[markdown]
# It certainly seems that the probe is picking up some signal here, and that more activations are in general producing a more accurate probe.  But how good are these scores?  Can we really say that the probe is finding the cheese x-position with any intuitively-compelling accuracy?  Let's take a look at what the x-pos predictions vs actual look like for one of the better-peforming probes...

# %%
value_label = 'embedder.block3.res1.resadd_out'
index_num_step = len(index_nums)-1
cheese_pos_x_pred = results.sel(value_label=value_label, index_num_step=index_num_step).y_pred_all
df = pd.DataFrame({'cheese pos x actual': cheese_pos_x, 'cheese pos x pred': cheese_pos_x_pred})
px.scatter(df, x='cheese pos x actual', y='cheese pos x pred', 
    title=f'Predicted vs actual cheese pos x using probe on {value_label}, K={index_nums[index_num_step]}')

# %%[markdown]
# Clearly there is a strong linear relationship here, but the errors are significant -- in the middle regions where the fit is best, the error is +/- 2 squares, and at the more extreme cheese positions there are numerous outliers.  Can we do better?
#
# ## Convolutional Probes
#
# The convolutional nature of the network imposes strict positional invariances on the computations performed.  For example, if a specific pixel in a specific convolutional channel picks up cheese at a specific maze square, then we know that we'll get exactly the same behavior if we translate the cheese location and the convolutional pixel location by the same amount.  This suggests a different type of probe that can leverage this invariance: instead of treating individual conv channel pixels as features, we unwrap the convolutional channels into the batch dimension, so that each pixel (or in an extended kernel-based version, each small patch) becomes a data point and we have only one such "feature" per convolutional channel.  Provided we also match our probe targets to the pixel locations appropriately, this dataset rearranging will give us many more data points with many less features, and also allow us to get the benefits of discrete probe targets while probing for targets at all spatial locations (TODO: explain this better!)
#
# To create datasets to test this approach (i.e. sets of convolution pixel values at the resadd layers of interest), we'll choose two pixels from each of our dataset samples: the pixel that most overlaps the object location, and another random pixel.  This keeps the dataset balanced.
#
# Let's focus on cheese first.  We'll build the dataset, then show f-test scores sorted over conv channels for each of our typical layers of interest, so we can see how relevance is distributed over channels, and also see how promising different layers are as probe soruces for cheese location...

# %%
# 
# Some helper functions
def grid_coord_to_value_ind(full_grid_coord, value_size):
    '''Pick the value index that covers the majority of the grid coord pixel'''
    # TODO: I'm pretty sure this is the best groudned approach, but ch 55 responds better to the approach in probing_main.py
    return np.floor((full_grid_coord+0.5) * value_size/maze.WORLD_DIM).astype(int)

def value_ind_to_grid_coord(value_ind, value_size):
    '''Pick the grid coordinate index whose center is closest to the center of the value pixel'''
    return np.floor((value_ind+0.5) * maze.WORLD_DIM/value_size).astype(int)

def get_obj_pos_data(value_label, object_pos):
    '''Pick the object location and a random other location without the object so we have a
    balanced dataset of pixels 2x the original size.'''
    rng = np.random.default_rng(15)
    # TODO: vectorize this!    
    value = hook.get_value_by_label(value_label).values
    value_size = value.shape[-1]
    num_pixels = num_batch * 2
    pixels = np.zeros((num_pixels, value.shape[1]))
    is_obj = np.zeros(num_pixels, dtype=bool)
    rows_in_value = np.zeros(num_pixels, dtype=int)
    cols_in_value = np.zeros(num_pixels, dtype=int)
    for bb in tqdm(range(obs_all.shape[0])):
        # Cheese location (transform from full grid row/col to row/col in this value)
        obj_pos_value = (grid_coord_to_value_ind(
                maze.WORLD_DIM-1 - object_pos.y[bb].item(), value_size),
            grid_coord_to_value_ind(object_pos.x[bb].item(), value_size))
        pixels[bb,:] = value[bb,:,obj_pos_value[0],obj_pos_value[1]]
        is_obj[bb] = True
        rows_in_value[bb] = obj_pos_value[0]
        cols_in_value[bb] = obj_pos_value[1]
        # Random pixel that isn't the object location
        bb_rand = bb + num_batch
        random_pos = obj_pos_value
        while random_pos == obj_pos_value:
            random_pos = (rng.integers(value_size), rng.integers(value_size))
        pixels[bb_rand,:] = value[bb,:,random_pos[0],random_pos[1]]
        is_obj[bb_rand] = False
        rows_in_value[bb_rand] = random_pos[0]
        cols_in_value[bb_rand] = random_pos[1]
    return pixels, is_obj, rows_in_value, cols_in_value

def show_f_test_results(pixels, target, target_name, rows_in_value, cols_in_value):
    f_test, _ = cpr.f_classif_fixed(pixels, target)
    f_test_df = pd.Series(f_test).sort_values(ascending=False)

    fig = px.line(y=f_test_df, title=f'Sorted {target_name} f-test scores for channels of<br>{value_label}',
        hover_data={'channel': f_test_df.index})
    fig.update_layout(
        xaxis_title="channel rank",
        yaxis_title="f-test score",)
    fig.show()

    print(list(f_test_df.index[:20]))

    for ch_ind in f_test_df.index[:2]:
        show_pixel_histogram(pixels, target, target_name, ch_ind)

value_labels_conv = [
    'embedder.block1.res1.resadd_out',
    'embedder.block1.res2.resadd_out',
    'embedder.block2.res1.resadd_out',
    'embedder.block2.res2.resadd_out',
    'embedder.block3.res1.resadd_out',
    'embedder.block3.res2.resadd_out']

f_test_list = []
pixel_data = {}
for value_label in value_labels_conv:
    pixels, is_obj, rows_in_value, cols_in_value = get_obj_pos_data(value_label, cheese_pos)
    f_test, _ = cpr.f_classif_fixed(pixels, is_obj)
    sort_inds = np.argsort(f_test)[::-1]
    pixel_data[value_label] = (pixels, is_obj, rows_in_value, cols_in_value, f_test, sort_inds)
    f_test_list.append(pd.DataFrame(
        {'layer': np.full(sort_inds.shape, value_label), 'rank': np.arange(len(sort_inds)),
         'channel': sort_inds, 'f-score': f_test[sort_inds]}))
    #show_f_test_results(pixels, is_obj, 'cheese', rows_in_value, cols_in_value)
f_test_df = pd.concat(f_test_list, axis='index')
px.line(f_test_df, x='rank', y='f-score', color='layer', hover_data=['channel'],
    title='Ranked f-test scores for "conv pixel contains cheese" over resadd layers').show()

# %%[markdown]
# From this we can observe that channels in the first two IMPALA blocks show strong peaks of cheese relevance, with the top 10 channels being significantly more relevant according to our f-test metric than the remainder.  This is some evidence that "cheese coding" channels exist in these layers, and that our "convolutional probing" approach may be able to use them.  We can also observe that f-scores decrease significantly in the third IMPALA block, implying that cheese location isn't encoded as clearly in this final block, providing some qualitative support for the ideas that (a) more complex abstractions are represented later in the network, and/or (b) later layers are more strongly influenced by mouse position since the final action logits need to apply to the specific position of the mouse, and information about mouse position has "had time" to propagate more widely through the convolutional layers.
#
# But what about the absolute values of these f-test scores?  How should we interpret those?  What does an f-score of 7000 actually mean in terms of the underlying distribution over classes?  Let's plot histograms of convolutional pixel values with and without cheese present for the top-5 f-score channels at a representative layer ('embedder.block2.res1.resadd_out') so we can get an intuitive feel for this...

# %%
# Show histograms
value_label = 'embedder.block2.res1.resadd_out'
pixels, is_obj, rows_in_value, cols_in_value, f_test, sort_inds = pixel_data[value_label]
chans_list = []
for ch in sort_inds[:5]:
    chans_list.append(pd.DataFrame({'pixel_value': pixels[:,ch], 'is_cheese_pixel': is_obj, 
        'channel': np.full(is_obj.shape, ch),
        'level_seed': np.concatenate([level_seeds, level_seeds]),
        'row_in_value': rows_in_value,
        'col_in_value': cols_in_value,}))
chans_df = pd.concat(chans_list, axis='index')
px.histogram(chans_df, title=f'{value_label} pixel values at "is cheese pixel" and "is not cheese pixel" locations',
    x='pixel_value', color='is_cheese_pixel', opacity=0.5, 
    barmode='overlay', facet_col='channel', facet_col_wrap=2,
    histnorm='probability', marginal='box', 
    hover_data=list(chans_df.columns)).show()

# %%[markdown]
# This shows us visually that these high-scoring channels do indeed show clearly different distributions of pixels values when cheese is present or absent.  Furthermore, the indices of these channels match up with channels that the team manually identified through visual inspection as appearing to code highly for cheese position.
# 
# So how would probes perform if we trained them on the top-K channels from these layers?  Let's find out...

# %%
# Train probes and show results
index_nums = np.arange(10)+1
scores_list = []
for value_label, (pixels, is_obj, rows_in_value, cols_in_value, f_test, sort_inds) in tqdm(pixel_data.items()):
    for K in index_nums:
        results = cpr.linear_probe(pixels[:,sort_inds[:K]], is_obj, C=10, random_state=42)
        scores_list.append({'layer': value_label, 'K': K, 'score': results['test_score']})
scores_df = pd.DataFrame(scores_list)

fig = px.line(scores_df, x='layer', y='score', color='K',
    title='Sparse probe scores for "conv pixel contains cheese" over resadd layers and K values')
fig.add_hline(y=1.,  line_dash="dot", annotation_text="perfect",  annotation_position="bottom right")
fig.add_hline(y=0.5, line_dash="dot", annotation_text="baseline", annotation_position="bottom right")
fig.show()

# %%[markdown]
# From this we can see near-perfect probe accuracy with K >= 2 throughout the first two IMPALA blocks, with still reasonable accuracy in the final IMPALA block.
#
# But knowing that the cheese is covered spatially by a specific convolutional channel pixel isn't the same as knowing the exact cheese location, since the convolutional pixels cover progressively more area of the observation as their resolution decreases due to maxpooling.  How does a naïve approach of predicting that the cheese is located at the center of the convolutional pixel projected into observation space do in terms of absolute accuracy for cheese x-position?

# %%
# Show accuracy of convolution pixel center over layers
cheese_pos_x_pred_list = []
for value_label, (pixels, is_obj, rows_in_value, cols_in_value, f_test, sort_inds) in pixel_data.items():
    value_size = hook.get_value_by_label(value_label, convert=False).shape[-1]
    cheese_pos_x_pred_list.append(pd.DataFrame(
        {'layer': value_label, 'x-pos actual': cheese_pos_x,
         'x-pos abs error': np.abs(cheese_pos_x-value_ind_to_grid_coord(cols_in_value[is_obj], value_size))}))
cheese_pos_x_pred_df = pd.concat(cheese_pos_x_pred_list, axis='index')
px.scatter(cheese_pos_x_pred_df, x='x-pos actual', y='x-pos abs error', facet_col='layer', facet_col_wrap=2,
    title='Error in cheese x-position using "center of convolution pixel" as prediction over layers').show()

# %%[markdown]
# From this, we can see that even block 3 layers result in a maximum error of one two mazes squares in cheese position if using the center of the convolutional pixel as the prediction, and block 1 layers would provide perfect accuracy.
#
# TODO: combine this with probe results to get a combined probe result!
#
# TODO: repeat for mouse position?

# %%[markdown]
# ## Probing for Goals
#
# Buoyed by some success probing for a couple of hypothesized primitive abstractions, one might be tempted to try probing directly for the high-level absraction that would be most directly relevant to the goals of this project: whether the mouse seeks the cheese or not.  
# 
# So let's try training a sparse linear probe with the target of predicting whether the mouse found the cheese at the end of each level in the dataset.  Take a minute to reflect on whether you would predict that a linear probe would show any success at predicting whether the mouse eventually takes the cheese in general?
#
# To find out, we train probes over the same resadd layers as before, plus the flattened output of the convolutional section of the model, the single fully connected hidden layer, and the final policy head action logits.

# %%
did_get_cheese = np.array([dd['did_get_cheese'] for dd in data_all], dtype=bool)

index_nums = np.array([1, 10, 20, 100, 1000])

results, _ = cpr.sparse_linear_probe(hook, values_to_store, did_get_cheese,
    index_nums = index_nums, random_state=42, class_weight='balanced', max_iter=1000, C=1)

plot_sparse_probe_scores(results, did_get_cheese, index_nums,
    f'Probe score over layers and K-values for "mouse did find cheese"')

# %%[markdown]
# I will personally admit to being pretty surprised that this probe works at all upon further reflection!  Even with only a single scalar activation value (i.e. one pixel in one convolutional channel), we can do measurably better than chance at predicting whether the mouse will get the cheese.  With larger numbers of activations (but still well below the full layer sizes until the final FC layer) we can get roughly half way from baseline to perfect accuracy.  What information are the probes picking up on here??
#
# To start with, let's visualize the relevance of pixels in different convolutional layers by summing f-scores over all channels over the usual layers...

# %%
# Visuzlize relevance of different pixel locations for "mouse did find cheese"
num_cols = 2
fig = py.subplots.make_subplots(rows=len(value_labels_conv)//num_cols, cols=num_cols,
    subplot_titles=value_labels_conv)
f_test_by_value = {}
for ii, value_label in tqdm(list(enumerate(value_labels_conv))):
    value = hook.get_value_by_label(value_label).values
    f_test_flat, _ = cpr.f_classif_fixed(rearrange(value, 'b ... -> b (...)'), did_get_cheese)
    f_test = rearrange(f_test_flat, '(c h w) -> c h w', h=value.shape[-2], w=value.shape[-1])
    f_test_by_value[value_label] = f_test
    f_test_sum = f_test.sum(axis=0)
    fig.add_trace(go.Heatmap(z=f_test_sum, coloraxis="coloraxis"), row=ii//num_cols+1, col=ii%num_cols+1)
    axis_id_str = '' if ii==0 else f'{ii+1}'
    fig.update_layout({f'yaxis{axis_id_str}_scaleanchor': f'x{axis_id_str}'})
fig.update_layout(height=600, title_text='f-test scores summed over channels for "mouse did find cheese", by layer')
fig.show()

# %%[markdown]
# Some interesting patterns jump out:
# - In the first IMPALA block, we can see a strong, symmetrical spatial pattern that seems to suggest a highly relevant square of activations.
# - In the second IMPALA block, we see irregular patterns conentrated at the edges, with some bulges at the corners.
# - In the final IMPALA block, we see something quite different: a big bloom of relevance in the center of the frame, with smaller blobs still at the corners.
# 
# The strong visual differences between layers suggest that the probes may be picking up different types of information at different layers.  Let's see if we can figure out what it is, starting with the first IMPALA block.
#
# ### IMPALA Block 1 Goal Probing
#
# What does that clear square of highly relevant activations correspond to in maze coordinates?

# %%
# Find relevant actvations in maze coords and plot
value_label = 'embedder.block1.res1.resadd_out'
f_test = f_test_by_value[value_label]
f_test_sum = f_test.sum(axis=0)
best_rows = f_test_sum.sum(axis=0).argsort()[-2:]
best_cols = f_test_sum.sum(axis=1).argsort()[-2:]
print(f'Most relevant rows/cols in layer pixel dimensions: {best_rows}, {best_cols}')
value_size = hook.get_value_by_label(value_label, convert=False).shape[-1]
best_rows_maze = value_ind_to_grid_coord(best_rows, value_size)
best_cols_maze = value_ind_to_grid_coord(best_cols, value_size)
print(f'Most relevant rows/cols in maze grid dimensions: {best_rows_maze}, {best_cols_maze}')

# %%[markdown]
# Given that the maze world size is 25x25, this implies that imformation about mazes squares +/-9 squares around the center of the maze (square 12,12) is relevant to whether the mouse finds the cheese.  Does the size of the maze inner grid have any predictive power on whether the mouse finds the cheese, and is this what the probe is picking up?  Specifically, do mazes of size 19x19 or larger affect cheese-finding significantly?

# %%
# Check how cheese finding is affected by maze size
get_cheese_frac_and_maze_dim = pd.DataFrame({'maze_dim': maze_dims, 
    'get_cheese': did_get_cheese}).groupby('maze_dim').mean().reset_index()
get_cheese_frac_and_maze_dim['source'] = 'actual'
px.line(get_cheese_frac_and_maze_dim, x='maze_dim',
    y='get_cheese', labels={'get_cheese': 'probability'}, 
    title='Probability of mouse finding cheese vs maze size over dataset').show()

# %%[markdown]
# Yes! Maze size seems to be a strong driver of cheese-finding likelihood, so it makes sense that probes would pick up on this.  Is this all that they are doing in this first IMPALA block?

# %%
# Train a single probe on a single block, see what it's predictions look like vs maze dim
K = 10
results, _ = cpr.sparse_linear_probe(hook, [value_label], did_get_cheese,
    index_nums = [K], random_state=42, class_weight='balanced', max_iter=1000, C=1)
pred_get_cheese = np.squeeze(results.y_pred_all.values)
pred_cheese_frac_and_maze_dim = pd.DataFrame({'maze_dim': maze_dims, 
    'get_cheese': pred_get_cheese}).groupby('maze_dim').mean().reset_index()
pred_cheese_frac_and_maze_dim['source'] = 'predicted'
fig = px.line(pd.concat([get_cheese_frac_and_maze_dim, pred_cheese_frac_and_maze_dim], axis='index'),
    y='get_cheese', labels={'get_cheese': 'probability'}, color='source',
    title=f'Actual/predictd probability of mouse finding cheese vs maze size over dataset, probing {value_label}, K={K}').show()

# %%[markdown]
# Yes! Okay, congrats logistic regression, you found a useful heuristic.
#
# But what's going on with that final IMPALA block?
#
# TODO: do this section!
#
# ## TODOs
# - Can we apply controls to these probes to confirm that they are picking up real representations?  https://nlp.stanford.edu/~johnhew/interpreting-probes.html
# - Finish TODO sections above
# - Probe for other hypothesized abstractions e.g. multi-square open/closes structures, dead-ends, square-is-on-path-to-cheese, etc.
#






# %%
# ARCHIVE ---------------------------------

# # What about probing for whether the mouse will pick the cheese direction at the decision square?
# to_cheese_is_best_dec_square_action = []
# action_logits = hook.get_value_by_label('fc_policy_out')
# for ii, dd in tqdm(list(enumerate(data_all))):
#     inner_grid = maze.EnvState(dd[state_bytes_key]).inner_grid()
#     padding = maze.get_padding(inner_grid)
#     graph = maze.maze_grid_to_graph(inner_grid)
#     path_to_cheese = maze.get_path_to_cheese(inner_grid, graph, dd['dec_node'])
#     if path_to_cheese[1][0] < path_to_cheese[0][0]: cheese_action = 'DOWN'
#     if path_to_cheese[1][0] > path_to_cheese[0][0]: cheese_action = 'UP'
#     if path_to_cheese[1][1] < path_to_cheese[0][1]: cheese_action = 'LEFT'
#     if path_to_cheese[1][1] > path_to_cheese[0][1]: cheese_action = 'RIGHT'
#     to_cheese_is_best_dec_square_action.append(
#         models.MAZE_ACTIONS_BY_INDEX[action_logits[ii,:].values.argmax()]==cheese_action)
# to_cheese_is_best_dec_square_action = np.array(to_cheese_is_best_dec_square_action, dtype=bool)

# results, _ = cpr.sparse_linear_probe(hook, values_to_store, to_cheese_is_best_dec_square_action,
#     index_nums = index_nums, random_state=42, class_weight='balanced', max_iter=1000, C=1)

# plot_sparse_probe_scores(results, to_cheese_is_best_dec_square_action, index_nums,
#     f'Probe score over layers and K-values for "best action was towards cheese at decision square"')


