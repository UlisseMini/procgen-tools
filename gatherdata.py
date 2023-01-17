import matplotlib.pyplot as plt
from procgen import ProcgenGym3Env
import torch
import envs.maze as maze
from models import load_policy
from gym.spaces import Discrete
from tqdm import tqdm
import numpy as np
import pickle
from argparse import ArgumentParser
import random

def create_venv(num_levels = 1, start_level = 0):
    venv = ProcgenGym3Env(
        num=1,
        env_name='maze', num_levels=num_levels, start_level=start_level,
        distribution_mode='hard', num_threads=4, render_mode="rgb_array",
    )
    venv = maze.wrap_venv(venv)
    return venv


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_file', type=str, default='./models/model_200015872.pth')
    parser.add_argument('--num_timesteps', type=int, default=256)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--argmax', action='store_true')

    args = parser.parse_args()

    # num is the number of environments to create in the vec env
    # num_levels is the number of levels to sample from (0 for infinite, 1 for deterministic)
    venv = create_venv()

    # Don't actually need to save. seed is enough.
    # state_bytes_list = venv.env.callmethod('get_state')
    # state_vals_list = [maze.parse_maze_state_bytes(sb) for sb in state_bytes_list]
    # grids = [maze.get_grid(s) for s in state_vals_list]

    # venv.env.callmethod("set_state", [state_bytes])

    assert isinstance(venv.action_space, Discrete)
    policy = load_policy(args.model_file, venv.action_space.n, device=torch.device('cpu'))
    model_name = args.model_file.split('/')[-1][:-4]

    # determinism
    random.seed(0)

    # plt.ion()
    for episode in tqdm(range(args.num_episodes)):
        venv = create_venv(random.randint(0, 100000))
        obs = venv.reset()
        assert venv.num_envs == 1, 'Only one env supported (for now)'
        done = np.zeros(venv.num_envs)
        log = {"rewards": [], "actions": [], "steps": 0}

        policy.eval()

        for step in tqdm(range(args.num_timesteps)):
            p, v = policy(torch.FloatTensor(obs))
            if args.argmax:
                act = p.probs.argmax(dim=-1).numpy()
            else:
                act = p.sample().numpy()
            obs, rew, done_now, info = venv.step(act)
            log["rewards"].append(float(rew[0]))
            log["actions"].append(int(act[0]))
            done = np.logical_or(done, done_now)
            log["steps"] = step
            # plt.imshow(info[0]['rgb'])
            # plt.show()
            plt.pause(0.01)
            if done:
                break


        # level seed allows reproduction
        log["level_seed"] = int(info[0]["prev_level_seed"]) # type: ignore
        # get basename of model file
        sampler = 'argmax' if args.argmax else 'sample'
        with open(f'data/{model_name}-ep{episode}-seed{log["level_seed"]}-{sampler}-{log["steps"]}steps.pkl', 'wb') as f: # TODO: Compression, batch trajectories
            pickle.dump(log, f, protocol=pickle.HIGHEST_PROTOCOL)
