import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from procgen import ProcgenGym3Env
from procgen_tools import maze
from procgen_tools.models import *
import torch as t
from torch.distributions.categorical import Categorical
from IPython import display
import pickle
from typing import List

def save_movie(venv, policy: t.nn.Module, condition_names: List[str], action_probs: bool = True, max_steps: int = 50, basename='traj_probs'):
    """
    Roll out the policy in the virtual environments, displaying side-by-side videos of the agent's actions. If action_probs is True, also display the agent's action probabilities. Saves the figure in "../figures/{basename}.gif".
    """
    num_envs = venv.num_envs
    num_actions = 4 # We only care about cardinal directions
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
        prob_axs = [fig.add_subplot(2, num_envs, 1+num_envs + i) for i in range(num_envs)]
        x = np.arange(num_actions) # x-axis for the bar chart
        labels = list(MAZE_ACTION_INDICES.keys())[:-1] # labels for the bar chart
        labels = [action.lower().capitalize() for action in labels] # capitalize the labels

        for ax in prob_axs: 
            #ax.set_title('Probabilities')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
 
        bars = []
        for i in range(num_envs):
            bars.append(prob_axs[i].bar(x, np.ones_like(x), width=.5))

    p_tens, v = t.zeros((num_envs, venv.action_space.n)), t.zeros((num_envs,))

    p = Categorical(logits=p_tens)
    last_info = np.empty((num_envs, 512, 512, 3), dtype=np.int32) # keep track of the last observation
    
    # Start recording a video of the figure 
    anim = animation.PillowWriter(fps=4)            
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
                dist = human_readable_actions(Categorical(probs=p.probs)) 
                for i in range(num_envs):    
                    for idx, prob in enumerate(dist.values()):
                        if list(dist.keys())[idx] == 'NOOP': continue # don't plot NOOP
                        bars[i][idx].set_height(prob[i]) 


            # Add a frame to the animation TODO convert to FFMPEGWriter 
            anim.grab_frame()
            
            # Sample actions 
            actions = p.sample().numpy()
            obs, rewards, dones_now, info = venv.step(actions)
            dones = np.logical_or(dones, dones_now)           