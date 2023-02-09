# %%
# Imports
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt

import circrl.module_hook as cmh

import procgen_tools.models as models
import torch.multiprocessing as mp
from queue import Empty, Full

import copy
import os

# %% 
# Load two levels and get values
from procgen_tools.patch_utils import *


def print_memory_info():
    from operator import itemgetter
    from pympler import tracker

    mem = tracker.SummaryTracker()
    print(sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10])

# %%
# Save vector fields and figures for a bunch of (seed, coeff) pairs


def _obs_worker(seed_queue: mp.Queue, obs_queue: mp.Queue):
    while True:
        seed = seed_queue.get(block=True)
        if seed == -1:
            print_memory_info()
            continue
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1, num_threads=0)
        venv_all, (legal_mouse_positions, grid) = maze.venv_with_all_mouse_positions(venv)
        obs_all = t.tensor(venv_all.reset(), dtype=t.float32)

        obs_queue.put((obs_all, (legal_mouse_positions, grid)))



# %%


if __name__ == '__main__':
    seed_queue = mp.Queue(maxsize=100)
    obs_queue = mp.Queue(maxsize=3)
    # pol_queue = mp.Queue(maxsize=10)

    for cpu in range(1):
        p = mp.Process(target=_obs_worker, args=(seed_queue, obs_queue))
        p.start()


    import random, time

    cnt = 0
    cnt_obs = 0
    start = time.monotonic()
    while cnt < 1000:
        # for debugging
        # if cnt % 100 == 1:
        #     seed_queue.put(-1, block=True)

        took = time.monotonic() - start
        print(f'{cnt} obs generated in {took:.2f} seconds [{cnt/took:.2f} seeds/s] [{cnt_obs/took:.2f} obs/s]')
        seed = random.randint(0, 1000000)
        seed_queue.put(seed, block=True)

        try:
            obs, _ = obs_queue.get(block=False)
            cnt += 1
            cnt_obs += len(obs)
            del obs
        except Empty:
            pass



# # %%



# # %%
# # Find memory leak

# import gc
# from operator import itemgetter
# from pympler import tracker

# mem = tracker.SummaryTracker()
# print(sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10])

# # %%

# num = 49199 // 10
# for obj in gc.get_objects():
#     if isinstance(obj, dict) and random.randint(0, num) == 0:
#         print(obj)
#         print('-'*80)

# # %%

# x = sum(isinstance(obj, dict) for obj in gc.get_objects())
# print(x)

# # %%

# exit()

# if __name__ == '__main__':
#     import itertools, pickle
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_file', type=str, default='trained_models/maze_I/model_rand_region_5.pth')
#     parser.add_argument('--label', type=str, default='embedder.block2.res1.resadd_out')
#     parser.add_argument('--device', type=str, default='cpu')
#     parser.add_argument('--num_levels', type=int, default=100)
#     parser.add_argument('--coeffs', type=str, default='-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3')
#     parser.add_argument('--save-figures', default=True, action='store_false')
#     args = parser.parse_args()

#     device = t.device(args.device)
#     seeds = range(args.num_levels)
#     coeffs = [float(c) for c in args.coeffs.split(',')]
#     label = args.label

#     policy = models.load_policy(path_prefix + args.model_file, 15, device)
#     hook = cmh.ModuleHook(policy)

#     for seed, coeff in tqdm(list(itertools.product(seeds, coeffs))):
#         fig, _, obj = plot_patched_vfields(seed, coeff, label, hook)
#         name = f"seed-{seed}_coeff-{coeff}_rr-{rand_region}_label-{label}"
#         with open(f'{path_prefix}data/vfields/{name}.pkl', 'wb') as fp:
#             pickle.dump(obj, fp)

#         if args.save_figures:
#             fig.savefig(f"{path_prefix}figures/{name}.png")
#         plt.clf()
#         plt.close()
# # %%

# %%
