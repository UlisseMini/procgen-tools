# %%

%matplotlib inline
import matplotlib.pyplot as plt
from procgen_tools import vfield, maze, models
from tqdm import tqdm
from glob import glob
from IPython.display import HTML, display, clear_output
import value_fn

# %%
# Plot vector fields for a bunch of models over time

# for model_file in glob('/home/uli/2023-02-02__17-29-21__seed_870/*.pth'):
#     policy = models.load_policy(model_file, 15, 'cpu')
#     venv = maze.create_venv(num=1, start_level=0, num_levels=1)

#     vf = vfield.vector_field(venv, policy)
#     vfield.plot_vf(vf)
#     plt.show()

# %%
# Create animated video of vector field using imageio

import imageio
from procgen_tools import vfield, maze, models
import numpy as np

with imageio.get_writer('vfield.mp4', fps=30) as writer:
    def _get_timestep(checkpoint: str):
        return int(checkpoint.split('_')[-1].split('.')[0])
    files = sorted(glob('/home/uli/2023-02-02__17-29-21__seed_870/*.pth'), key=lambda x: int(_get_timestep(x)))

    for model_file in tqdm(files):
        policy = models.load_policy(model_file, 15, 'cpu')
        venv = maze.create_venv(num=1, start_level=4, num_levels=1)

        # vf = vfield.vector_field(venv, policy)
        # vfield.plot_vf(vf)
        vf, (agree, total) = value_fn.plot(policy, venv)

        plt.axis('off')

        # preemptive fix for 'FigureCanvasAgg' object has no attribute 'renderer'
        plt.gcf().canvas.draw()
        # call writer.append_data on numpy data for current figure
        im_bytes = plt.gcf().canvas.tostring_rgb()
        writer.append_data(np.frombuffer(im_bytes, dtype=np.uint8).reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,)))
        plt.clf()
        plt.close()
# %%
# Show video in notebook


display(HTML("<video width='640' height='480' controls><source src='vfield.mp4' type='video/mp4'></video>"))

# %%
