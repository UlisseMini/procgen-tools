from procgen import ProcgenGym3Env
import torch
from procgen_tools import maze
from procgen_tools.models import load_policy
from tqdm import tqdm
import numpy as np
import pickle
from argparse import ArgumentParser
import random
from data_util import Episode

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
    parser.add_argument('--num_timesteps', type=int, default=256, help='maximum timesteps per episode')
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to collect (agent finishes or times out)')
    parser.add_argument('--argmax', action='store_true', help='argmax logits instead of sampling. often gets stuck, but when successful has less jittering')

    args = parser.parse_args()

    policy = load_policy(args.model_file, action_size=15, device=torch.device('cpu'))
    model_name = args.model_file.split('/')[-1][:-4]

    # determinism
    random.seed(42)

    for ep in tqdm(range(args.num_episodes)):
        venv = create_venv(start_level=random.randint(0, 100000))
        assert venv.num_envs == 1, 'Only one env supported (for now)'

        # grab initial_state_bytes and initial info for episode object
        states_bytes = venv.env.callmethod('get_state')[0]
        states_vals = maze.parse_maze_state_bytes(states_bytes)
        info = venv.env.get_info()

        # init episode object
        mouse_positions_outer = [maze.get_mouse_pos_sv(states_vals)]
        actions, rewards = [], []
        sampler = "argmax" if args.argmax else "sample"
        episode = Episode(
            initial_state_bytes=states_bytes,
            mouse_positions_outer=mouse_positions_outer,
            actions=actions, rewards=rewards, sampler=sampler,
            level_seed=int(info[0]["level_seed"]),
        )

        policy.eval()
        done = np.zeros(venv.num_envs)
        obs = venv.reset()
        for step in tqdm(range(args.num_timesteps)):
            p, v = policy(torch.FloatTensor(obs))
            if args.argmax:
                act = p.probs.argmax(dim=-1).numpy()
            else:
                act = p.sample().numpy()
            obs, rew, done, info = venv.step(act)
            if done:
                # IMPORTANT: we don't update episode here. otherwise we'll log the last frame (a new level)
                break

            states_bytes = venv.env.callmethod('get_state')[0]
            states_vals = maze.parse_maze_state_bytes(states_bytes)

            rewards.append(float(rew[0]))
            actions.append(int(act[0]))
            mouse_positions_outer.append(maze.get_mouse_pos_sv(states_vals))


        # get basename of model file
        with open(f'../episode_data/{model_name}-ep{ep}-seed{episode.level_seed}-{sampler}-{episode.steps}steps.pkl', 'wb') as f: # TODO: Compression, batch trajectories
            state = episode.__getstate__()
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
