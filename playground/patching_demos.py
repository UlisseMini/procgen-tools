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
# Demonstrate patching.  This should generate two sets of video + action logits heatmap,
# one for a normal unpatched rollout, and one for patched rollout where a large number is 
# added to the RIGHT final action logit.  (Any layer can be patched this way, just
# showing the final layer because it is easy to see the effect on behavior.)

# Load model and environment

env = create_venv(num=1, num_levels=1)
policy = models.load_policy('../trained_models/maze_I/model_rand_region_5.pth', 15,
    t.device('cpu'))

# Hook the network and demonstrate a custom patching function on a rollout
hook = cmh.ModuleHook(policy)

# Custom predict function to match rollout expected interface, uses
# the hooked network so it is patchable
def predict(obs, deterministic):
    obs = t.FloatTensor(obs)
    dist, value = hook.network(obs)
    if deterministic:
        act = dist.mode.numpy()
    else:
        act = dist.sample().numpy()
    return act, None

action_logits_label = 'fc_policy_out'

# Run a normal, unpatched roll-out
seq, _, _ = cro.run_rollout(predict, env, max_steps=30, deterministic=False)
# Probe to get all the activations for this rollout, then show the action logits
hook.probe_with_input(seq.obs.astype(np.float32))
action_logits_orig = hook.get_value_by_label(action_logits_label)
# Make a video of the renders so we can see what it looks like
vid_fn, fps = cro.make_video_from_renders(seq.renders, fps=10)
display(Video(vid_fn, embed=True))
px.imshow(action_logits_orig.T).show()

# Add a large value to the 'RIGHT' logit (index 7) and show this affects behavior
patches = {action_logits_label: lambda outp: outp + (t.arange(15)==7)*10}
with hook.use_patches(patches):
    seq_patched, _, _ = cro.run_rollout(predict, env, max_steps=30, deterministic=False)
# Probe to get all the activations for this rollout, then show the action logits
hook.probe_with_input(seq_patched.obs.astype(np.float32), patches=patches)
action_logits_patched = hook.get_value_by_label(action_logits_label)
# Make a video of the renders so we can see what it looks like
vid_fn, fps = cro.make_video_from_renders(seq_patched.renders, fps=10)
display(Video(vid_fn, embed=True))
px.imshow(action_logits_patched.T).show()

# %%
