
import sys, os
COLAB = 'google.colab' in sys.modules 
if COLAB and 'procgen-tools' not in os.getcwd():
    os.system("git clone https://github.com/UlisseMini/procgen-tools")
    os.chdir('procgen-tools')
    # %pip install -r requirements.txt


# %load_ext autoreload
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from procgen import ProcgenGym3Env
import envs.maze as maze
from models import load_policy
import torch as t
from torch.distributions.categorical import Categorical
from IPython import display
import pickle



# num is the number of environments to create in the vec env
# num_levels is the number of levels to sample from (0 for infinite, 1 for deterministic)
venv = ProcgenGym3Env(
    num=3, env_name='maze', num_levels=1, start_level=0,
    distribution_mode='hard', num_threads=1, render_mode="rgb_array",
)
venv = maze.wrap_venv(venv)


try:
    with open('./saved-mazes.pkl', 'rb') as f:
        state_bytes_list = pickle.load(f)
        if len(state_bytes_list) != venv.num_envs:
            print(f'WARN: saved {len(state_bytes_list)} envs but num_envs is {venv.num_envs}')
        else:
            venv.env.callmethod('set_state', state_bytes_list)

except FileNotFoundError:
    print('No ./saved-mazes.pkl file exists, using default envs')
        

 
# ## Check that maze is in-distribution
# 
# The maze is generated using a variant of [Kruskal's algorithm](https://weblog.jamisbuck.org/2011/1/3/maze-generation-kruskal-s-algorithm). The algorithm results in some constraints on generated mazes we want to uphold, in order to stay in-distribution.
# 1. Squares where the row and column indices are both odd, e.g. (1,1), must always be blocked.
# 2. Squares where the row and column indices are both even, e.g. (0,0), must always be open.
# 3. Squares where the row and column indices are (odd, even) or (even, odd) may be open or blocked.
# 4. The maze must be fully-connected, i.e. there must be a path from every open square to every other open square.
# 5. The maze must not have any loops or cycles, i.e. there must be exactly one path between any two open squares.
# 
# These are all checked by the `maze.on_distribution` function.


#assert maze.on_distribution(gridm), "Maze isn't on distribution!"

 
# ## Run the model on the new maze


if COLAB:
  from google.colab import files
  uploaded = files.upload()
  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))


# download from https://drive.google.com/file/d/1db1qJn_ZM49CUcA1k7uSMdsdiAQ4eR5a/view?usp=share_link
policy = load_policy('../models/maze_I/model_rand_region_5.pth', action_size=venv.action_space.n, device=t.device('cpu'))



done = np.zeros(venv.num_envs)
obs = venv.reset()

# import type-checking library
from typing import List
def get_movies(venv, policy: t.nn.Module, condition_names: List[str], action_probs: bool = True, max_steps: int = 50, basename='traj_probs'):
    """
    Roll out the policy in the virtual environments, displaying side-by-side videos of the agent's actions. If action_probs is True, also display the agent's action probabilities. Saves the figure in "../figures/{basename}.gif".
    """
    action_dict = {'left': 2, 'down': 3, 'up': 5, 'right': 6}
    num_envs = venv.num_envs
    assert num_envs == len(condition_names), "Number of environments must match number of condition names"

    obs = venv.reset() # reset the environments
    dones = np.zeros(num_envs, dtype=bool)

    # Initialize the figure
    plt.clf()    
    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(2, num_envs, i+1) for i in range(num_envs)]

    # Remove axis ticks and labels
    for i, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(condition_names[i])

    if action_probs: # Initialize the probability axes
        # There is a single action probability bar chart below the video axes
        logit_ax, prob_ax = [fig.add_subplot(2, num_envs, num_envs+i) for i in (1,2)]
        logit_ax.set_title('Logits')
        prob_ax.set_title('Probabilities')
        x = np.arange(action_dict.keys().__len__()) # x-axis for the bar chart
        width = .8 / num_envs # width of each bar; want whitespace in between actions
        
        bars = {'prob': [], 'logit': []}
        for i in range(num_envs):
            bars['logit'].append(logit_ax.bar(x + (i-1)*width, np.ones_like(x) * -10, width=width, label=condition_names[i]))
            bars['prob'].append(prob_ax.bar(x + (i-1)*width, np.ones_like(x), width=width, label=condition_names[i]))

        # Add a legend to the action probability axes
        prob_ax.legend()

        ax.set_xticks(x)
        ax.set_xticklabels(action_dict.keys())

    p_tens, v = t.zeros((num_envs, venv.action_space.n)), t.zeros((num_envs,))

    p = Categorical(logits=p_tens)
    last_info = np.empty((num_envs, 512, 512, 3), dtype=np.int32) # keep track of the last observation
    
    # Start recording a video of the figure 
    anim = animation.PillowWriter(fps=10)            
    with anim.saving(fig, f"./figs/{basename}.gif", 100):
        for step in range(max_steps):
            if dones.all(): break 

            # Plot the observations
            info = venv.env.get_info()
            for i in range(obs.shape[0]):
                if not dones[i]: last_info[i] = info[i]['rgb']
                axs[i].imshow(last_info[i])

            with t.no_grad():
                p_cat, v[~dones] = policy(t.FloatTensor(obs[~dones]))
            p_tens[~dones] = p_cat.logits
            p = Categorical(logits=p_tens)

            if action_probs: # Plot the action probabilities
                indices = list(action_dict.values())
                for i in range(num_envs):    
                    for act_ind in range(len(action_dict)): # set the height of each bar
                        bars['logit'][i][act_ind].set_height(p.logits[i][indices[act_ind]])
                        bars['prob'][i][act_ind].set_height(p.probs[i][indices[act_ind]])

            # Add a frame to the animation TODO convert to FFMPEGWriter 
            anim.grab_frame()
            
            # Sample actions 
            actions = p.sample().numpy()
            obs, rewards, dones_now, info = venv.step(actions)
            dones = np.logical_or(dones, dones_now)            

get_movies(venv, policy, condition_names=['far', 'near', 'vanished'])

