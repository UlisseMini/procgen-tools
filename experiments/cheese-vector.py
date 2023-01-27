# %%
# Imports
import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
import plotly as py
import plotly.graph_objects as go

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
from procgen_tools import maze
from procgen_tools.models import load_policy

import gatherdata

# %%
# Load two levels
import pickle as pkl
from procgen import ProcgenGym3Env
import lovely_tensors as lt
lt.monkey_patch()

venv = ProcgenGym3Env(
    num=2, env_name='maze', num_levels=1, start_level=0,
    distribution_mode='hard', num_threads=1, render_mode="rgb_array",
)
venv = maze.wrap_venv(venv)

with open('mazes/2.pkl', 'rb') as f:
    state_bytes = pkl.load(f) 
venv.env.callmethod('set_state', state_bytes)

# %%
# Load model
modelpath = 'trained_models/maze_I/model_rand_region_5.pth'
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

num_actions = venv.action_space.n # lol

# Load model
policy = load_policy(modelpath, num_actions, device=device)

# Hook the network and run this observation through a custom predict-like function
hook = cmh.ModuleHook(policy)

# Custom probe function to evaluate the policy network
def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)

# Get initial observation, and show maze rendering
obs = venv.reset().astype(np.float32)  # Not sure why the venv is returning a float64 object?
render = venv.render(mode='rgb_array')
# px.imshow(render, title='Rendering').show()

# Do an initial run of this observation through the network
hook.probe_with_input(obs, func=forward_func_policy)

# Show the labels of all the intermediate activations
# print(hook.values_by_label.keys())

# Visualize a random intermediate activation, and the logits
label = 'embedder.fc_out'
value = hook.get_value_by_label(label)
action_logits = hook.get_value_by_label('fc_policy_out').squeeze()
# px.imshow(value[0,...], title=label).show()

# Demonstrate ablating some values to zero, show impact on action logits
# (Just ablate the first channel of the above activation as a test)
mask = t.from_numpy(np.ones(1,dtype=bool))

zero = {label: cmh.PatchDef(
    mask,
    t.from_numpy(np.array([0.], dtype=np.float32)))}
duplicate = {label: cmh.PatchDef(
    mask,
    t.from_numpy(value[0,...]))}

# Run the patched probes
for name, patches in zip(('zero patch', 'cheese patch'), (zero, duplicate)):
    hook.probe_with_input(obs,  func=forward_func_policy, patches=patches)
    value_patched = hook.get_value_by_label(label)
    action_logits_patched = hook.get_value_by_label('fc_policy_out').squeeze()
    if name == 'cheese patch': 
        assert np.allclose(action_logits_patched[0], action_logits_patched[1]), "Somehow the patched logits don't match the source logits!"

    # Plot results
    action_meanings = venv.env.combos
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=action_logits[1], name='original'))
    fig.add_trace(go.Scatter(y=action_logits_patched[1], name='patched'))
    fig.update_layout(title=f"No-cheese logits with {name}")
    fig.update_xaxes(tickvals=np.arange(action_logits.shape[-1]), ticktext=action_meanings)
    fig.show()

# %% 