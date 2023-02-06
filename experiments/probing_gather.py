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
import os
import random

import numpy as np
import pandas as pd
import torch as t
from argparse import ArgumentParser

import procgen_tools.models as models
import procgen_tools.maze as maze
import gatherdata
import gatherdata_rich

def setup_env():
    has_dec_sq = False
    while not has_dec_sq:
        start_level = random.randint(0, 1e6)
        venv = gatherdata.create_venv(num=1, start_level=start_level, num_levels=1)    
        episode_metadata = dict(start_level=start_level, 
            level_seed=int(venv.env.get_info()[0]["level_seed"]))
        has_dec_sq = maze.maze_has_decision_square(
            venv.env.callmethod('get_state')[0])
    return venv, episode_metadata

# Gather data
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_file', type=str, 
        default='../trained_models/maze_I/model_rand_region_5.pth')
    parser.add_argument('--num_timesteps', type=int, default=256, help='maximum timesteps per episode')
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to collect (agent finishes or times out)')

    args = parser.parse_args()

    policy = models.load_policy(args.model_file, action_size=15, device=t.device('cpu'))
    model_name = os.path.basename(args.model_file)

    gatherdata_rich.get_maze_dataset(policy, model_name, args.num_episodes, 
        args.num_timesteps, env_setup_func=setup_env)

