# %%
# Imports
%reload_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from tqdm import tqdm
from einops import rearrange
from IPython.display import Video, display

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro

import procgen_tools.models as models
from procgen_tools.maze import create_venv

# %%
# The patching feature of ModuleHook can be used in two main ways:
# 1. In a single hook.run_with_input call, to apply patches during a single forward
#    pass on a batch of observations.  Likely the best way to test patches if
#    interaction with the actual environment isn't required.
# 2. By using a with hook.use_patches(...) context manager, which allows arbitrary
#    use of the hooked model with patches applied at all times inside the context
#    manager.  This can be used to e.g. apply patches during a roll-out in a real
#    environment.
#
# Both use cases are demonstrated below.

action_logits_label = 'fc_policy_out'

# First, load a pre-trained model and hook it
policy = models.load_policy('../trained_models/maze_I/model_rand_region_5.pth', 15,
    t.device('cpu'))
hook = cmh.ModuleHook(policy)

# Helper predict function that matches the interface for selecting # an action that is expected by run_rollout from circrl.
# Uses the hooked network so patching can be applied if needed,
# and activations can be accessed.
def predict(obs, deterministic):
    obs = t.FloatTensor(obs)
    dist, value = hook.network(obs)
    if deterministic:
        act = dist.mode.numpy()
    else:
        act = dist.sample().numpy()
    return act, None

# Define a helper function we'll use to run a rollout, make a video, and 
# return the observations and action logits.  Uses the above specified 
# predict function, and thus the hooked network.
def run_and_show_rollout(desc, patches={}):
    env = create_venv(num=1, start_level=3, num_levels=1)
    # Here we show the pattern of using a context manager to apply patches
    # while arbitrary code is executed.
    with hook.use_patches(patches):
        seq, _, _ = cro.run_rollout(predict, env, max_steps=50, deterministic=False)
    # Probe to get all the activations for this rollout, then return the action logits
    obs = seq.obs.astype(np.float32)
    # Note: this call has to happen outside the with block, since it also
    # sets patches (this is a bit confusing and should prob trigger a warning.)
    hook.run_with_input(obs, patches=patches)  
    action_logits = hook.get_value_by_label(action_logits_label)
    # Make a video of the renders so we can see what it looks like
    vid_fn, fps = cro.make_video_from_renders(seq.renders, fps=10)
    print(desc)
    display(Video(vid_fn, embed=True))
    return obs, action_logits

# Run a baseline rollout with the normal, unpached network
obs_orig, action_logits_orig = run_and_show_rollout('Baseline, unpatched rollout.')

# Create a patch object to zero-ablate a specific channel in a specific layer
# Takes shape from activation object generated during baseline rollout above.
# Note that convert=False below leaves the value object as a tensor, rather
# then converting to an xarray.
value_label = 'embedder.block2.res1.conv1_out'
channel = 123
# Note: shape should have 1 in batch dimension, so it can broadcast to any number
# of observations and thus be used in rollouts (with single obs) and in
# run_with_input passes when we might have a big batch of observations.
patch_shape = (1,) + hook.get_value_by_label(value_label, convert=False).shape[1:]
patch_value = t.zeros(patch_shape)
patch_mask = t.zeros(patch_shape, dtype=bool)
patch_mask[:,channel,:,:] = True
patches = {value_label: cmh.PatchDef(patch_mask, patch_value)}

# Run a forward pass with the patches to see how the action logits change
hook.run_with_input(obs_orig, patches=patches)
action_logits_patch1 = hook.get_value_by_label(action_logits_label)

# Show the original and patched action logits, and the diff
px.imshow(action_logits_orig.T, title='Original action logits').show()
px.imshow(action_logits_patch1.T, title=
    f'Patched action logits over batch of obs<br>channel {channel} in layer {value_label} zero-ablated').show()
px.imshow(action_logits_patch1.T-action_logits_orig.T, 
    title='Patched minus original action logits').show()


# Now demonstrate running patches online during a rollout...

# First with the same single-channel zero-ablation patch
obs_patch2, action_logits_patch2 = run_and_show_rollout(
    f'Patched rollout, channel {channel} in layer {value_label} zero-ablated.',
    patches=patches)
px.imshow(action_logits_patch2.T, title=
    f'Patched action logits over rollout<br>channel {channel} in layer {value_label} zero-ablated').show()

# Then, with a more extreme example: ddd a large value to the 'RIGHT' 
# logit (index 7) and show this affects behavior.  (This is patching
# directly on the action logits since it's easy to control behavior this
# way and makes for a clear demo.)
# This also demonstrated that we can pass a custom function taking an
# activation tensor and returning a modified tensor, instead of the 
# basic mask/value PatchDef if we need this for some reason.
patches = {action_logits_label: lambda outp: outp + (t.arange(15)==7)*10}
obs_patch3, action_logits_patch3 = run_and_show_rollout(
    f'Patched rollout, +10 added to RIGHT logit.',
    patches=patches)
px.imshow(action_logits_patch3.T, title=
    f'Patched action logits over rollout<br>, +10 added to RIGHT logit').show()

