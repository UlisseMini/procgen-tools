# %%
try:
    import procgen_tools
except ImportError or ModuleNotFoundError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup() # create directory structure and download data 

# %%
from procgen_tools.imports import *
# %matplotlib inline
from procgen_tools.procgen_imports import *
from procgen_tools import vfield_stats

# %%
# rand_region_5 model on all mazes, load 100 (uses a lot of memory)
from tqdm import tqdm
files = glob('episode_data/20230131T032642/*.dat')[:100] 

from circrl.rollouts import load_saved_rollout
rollouts = [load_saved_rollout(f) for f in tqdm(files)]

AX_SIZE = 5.5

# %% [markdown]
# # Introduction
# 
# We're looking at the maze solving agents from the [goal misgeneralization](https://arxiv.org/abs/2105.14111) paper. In particular, the agents were reinforced when they contacted cheese in the top-right corner of a guaranteed-solvable maze.
# 
# It's important to keep in mind the difference between the human-friendly high-resolution view, and what the agents actually observe. We also sometimes consult a "grid view" when doing backend work.

# %%
venv = maze.create_venv(num=1, start_level=0, num_levels=1)
fig, axs = plt.subplots(1,3, figsize=(AX_SIZE * 3, AX_SIZE))

for ax, mode in zip(axs, ['human', 'agent', 'numpy']):
    visualize_venv(venv, mode=mode, idx=0, ax=ax, show_plot=False, render_padding=mode != 'human')

plt.show()

# %%
venv = maze.create_venv(num=1, start_level=5, num_levels=1)
visualize_venv(venv, mode='human', idx=0, show_plot=True, render_padding=True, render_mouse=False)

# %%
