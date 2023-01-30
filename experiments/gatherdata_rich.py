import os
import random

import torch as t
from tqdm.auto import tqdm

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro

import procgen_tools.models as models
import procgen_tools.maze as maze
import gatherdata
from argparse import ArgumentParser

# Custom predict function to match rollout expected interface
def get_predict(policy):
    def predict(obs, deterministic):
        obs = t.FloatTensor(obs)
        dist, value = policy(obs)
        if deterministic:
            act = dist.mode.numpy()
        else:
            act = dist.sample().numpy()
        return act, None, dist.logits.detach().numpy()
    return predict

def setup_env():
    start_level = random.randint(0, 1e6)
    venv = gatherdata.create_venv(start_level=start_level)
    episode_metadata = dict(start_level=start_level, 
        level_seed=int(venv.env.get_info()[0]["level_seed"]))
    return venv, episode_metadata

def get_maze_dataset(policy, policy_desc, num_episodes, num_timesteps, seed=42,
        env_setup_func=setup_env):
    # Seed the RNG
    random.seed(seed)
    # Maze state getter function to pass to run_rollout
    # (make all args kwargs so calling order doesn't matter)
    def get_maze_state(env=None, **kwargs):
        return env.env.callmethod('get_state')[0]
    # Logit getter function to pass to run_rollout
    # (make all args kwargs so calling order doesn't matter)
    def get_action_logits(predict_extra=None, **kwargs):
        return predict_extra[0]
    # Function to remove renders from the seq object to save space
    # (Modifies in-place, returns ref)
    def remove_renders_from_seq(seq):
        seq.renders = None
        return seq
    # Get the dataset
    cro.make_dataset(get_predict(policy), 
        f'Rich rollouts including maze state at each timestep, policy: {policy_desc}', 
        '../episode_data', num_episodes, setup_env, seq_mod_func=remove_renders_from_seq,
        run_rollout_kwargs=dict(max_steps=num_timesteps,
            custom_data_funcs=dict(state_bytes=get_maze_state, action_logits=get_action_logits)))

# Generate a dataset
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_file', type=str, default='../trained_models/model_200015872.pth')
    parser.add_argument('--num_timesteps', type=int, default=256, help='maximum timesteps per episode')
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to collect (agent finishes or times out)')
    parser.add_argument('--argmax', action='store_true', help='argmax logits instead of sampling. often gets stuck, but when successful has less jittering')

    args = parser.parse_args()

    policy = models.load_policy(args.model_file, action_size=15, device=t.device('cpu'))
    model_name = os.path.basename(args.model_file)

    get_maze_dataset(policy, model_name, args.num_episodes, args.num_timesteps)