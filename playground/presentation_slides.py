# %%

import matplotlib.pyplot as plt
import random
import procgen_tools.maze as maze
from procgen_tools.maze import EnvState, create_venv, render_inner_grid
import numpy as np
import torch as t
import procgen_tools.patch_utils as pu
from procgen_tools.imports import hook, default_layer

AX_SIZE = 3.5

# %%
# Generate on dist vs. off dist figure

fig, ax = plt.subplots(1,2, figsize=(AX_SIZE * 2, AX_SIZE))

for i, (seed, title) in enumerate(zip((9, 0), ('Training: Top right 5x5', 'Deployment: Anywhere'))):
    img = render_inner_grid(EnvState(create_venv(1,seed,1).env.callmethod('get_state')[0]).inner_grid())
    ax[i].imshow(img)
    ax[i].set_title(title)
    ax[i].set_xticks([]); ax[i].set_yticks([])

# plt.savefig(f'mazes-on-off.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()


# %%
# Generate mechint figure, plot cheese encoding channels on the left
# and right, and plot the maze largely in the middle.

# CHEESE_CHANNELS = [42, 55, 99, 113, 77, 8, 70, 82, 46, 75, 39, 92, 61, 9, 88, 98, 44, 127, 95, 91] # Monte's probing
CHEESE_CHANNELS = [42, 55, 88, 77, 113, 8]


def plot_cheese_channels(hook):
    fig, axd = plt.subplot_mosaic(
        [
            ['L1', 'M', 'M', 'R1'],
            ['L2', 'M', 'M', 'R2'],
            # ['L3', 'M', 'M', 'R3'],
        ],
        figsize=(AX_SIZE * 4, AX_SIZE * 2),
        tight_layout=True,
    )

    # fig.suptitle('Cheese encoding channels', fontsize=24)


    i = 0
    for k in axd:
        axd[k].set_xticks([]); axd[k].set_yticks([])
        if k != 'M':
            activ = hook.get_value_by_label(default_layer)[0][CHEESE_CHANNELS[i]]
            axd[k].imshow(activ)
            # Use RdBu colormap for better contrast
            axd[k].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)
            axd[k].set_title(f'Channel {CHEESE_CHANNELS[i]}', fontsize=18)
            i += 1
        else:
            axd[k].imshow(render_inner_grid(EnvState(venv.env.callmethod('get_state')[0]).inner_grid()))

    return fig


venv = create_venv(1,0,1)
state = maze.EnvState(venv.env.callmethod('get_state')[0])
empty_space = maze.get_legal_mouse_positions(state.full_grid())

random.seed(42)
for i, cheese_pos in enumerate(random.sample(empty_space, 5)):
    maze.move_cheese_in_state(state, cheese_pos)
    venv.env.callmethod('set_state', [state.state_bytes])
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.set_hook_should_get_custom_data():
        hook.network(obs)

    fig = plot_cheese_channels(hook)
    plt.savefig(f'maze-{i}', bbox_inches='tight', pad_inches=0, dpi=300)

# Create gif with imageio 

import imageio
from glob import glob
filenames = sorted(glob('maze-*.png'), key=lambda x: int(x.split('-')[1].split('.')[0]))
with imageio.get_writer('maze.gif', mode="I", fps=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
# Delete pngs
os.system('rm maze-*.png')

# %%
# 