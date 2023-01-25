
import sys, os
COLAB = 'google.colab' in sys.modules 
if COLAB and 'procgen-tools' not in os.getcwd():
    os.system("git clone https://github.com/UlisseMini/procgen-tools")
    os.chdir('procgen-tools')
    # %pip install -r requirements.txt


# %load_ext autoreload
# %matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
from procgen import ProcgenGym3Env
import envs.maze as maze
from models import load_policy
import torch as t
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
    print('No ./saved-mazes.pkl file exists, generating venv from scratch')
        

obs = venv.reset()
# plot all the envs in the vectorized env
for i in range(obs.shape[0]):
    plt.imshow(obs[i].transpose(1, 2, 0))
    # plt.show()

 
# ## Edit the maze interactively
# 
# Using the magic `maze.grid_editor` function!
# Clicking in the maze changes walls to empty space and vise versa.
# If you click the cheese it'll disappear and reappear where you click next.


from ipywidgets import VBox, Text


editor_lst, grid_lst = [], []
vals_lst = []
for i in range(venv.num_envs):
    state_bytes = venv.env.callmethod('get_state')[i]
    vals_lst.append(maze.parse_maze_state_bytes(state_bytes))
    grid_lst.append(maze.get_grid_with_mouse(vals_lst[-1])) # gridm because it includes the mouse
    editor_lst.append(maze.grid_editor(grid_lst[-1], node_radius='8px'))
    editor_lst.append(Text("-----"))
VBox(editor_lst)


"""
Overwrite the env with the edited maze

If you don't do this venv will still be using the old state!
"""

for i in range(venv.num_envs):
    maze.set_grid_with_mouse(vals_lst[i], grid_lst[i])
    
state_bytes_list = [maze.serialize_maze_state(sv) for sv in vals_lst]
venv.env.callmethod("set_state", state_bytes_list)

obs = venv.reset()
info = venv.env.get_info()
# Show all three environments, with each titled "far", "near", and "vanished" respectively

plt.close()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i in range(obs.shape[0]):
    # plt.imshow(obs[i].transpose(1, 2, 0)) # agent view
    axes[i].imshow(info[i]['rgb'])
    axes[i].set_title(('far', 'near', 'vanished')[i])
# Remove axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.show()
plt.savefig('./figs/action.png')

with open('./saved-mazes.pkl', 'wb') as f:
    pickle.dump(state_bytes_list, f)

 
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
policy = load_policy('/home/turneale/Documents/Work/MATS/23-Winter/models/maze_I/model_rand_region_5.pth', action_size=venv.action_space.n, device=t.device('cpu'))



done = np.zeros(venv.num_envs)
obs = venv.reset()

# actions:
"""
2 left
3 down 

5 up
6 right
"""

policy.eval()
with t.no_grad():
    p, v = policy(t.FloatTensor(obs))

actions, labels = t.tensor([2, 3, 5, 6], dtype=t.long), ['left', 'down', 'up', 'right']
x=np.arange(len(labels))
width = .2

far, near, vanished = [p.logits[i] for i in range(venv.num_envs)]
farp, nearp, vanishedp = [p.probs[i] for i in range(venv.num_envs)]

# Clear previous plot
plt.clf()
plt.close()
# Make two barplot axes
fig, (ax1, ax2) = plt.subplots(1, 2)
# Make triple-barplot with matplotlib,
for i in range(venv.num_envs):
    ax1.bar(x+(i-1)*width, p.logits[i][actions], width=width, label=('far', 'near', 'vanished')[i])
    ax2.bar(x+(i-1)*width, p.probs[i][actions], width=width, label=('far', 'near', 'vanished')[i])

ax1.set_title('Action logits')
ax2.set_title('Action probabilities')
plt.xticks(labels, labels)
plt.legend()
plt.show()
plt.savefig('./figs/action_probs.png')




