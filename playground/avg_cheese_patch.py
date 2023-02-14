# %%

import matplotlib.pyplot as plt
from procgen_tools import models, maze, vfield
import procgen_tools.patch_utils as patch_utils
from glob import glob
import circrl.module_hook as cmh
import numpy as np
import torch as t
from ipywidgets import interact, IntSlider, Dropdown


def main(
    label = 'embedder.block2.res1.resadd_out',
    model_file = '../trained_models/model_rand_region_5.pth',
    coeff = -1.,
    seed = 1,
):
    policy = models.load_policy(model_file, 15, 'cpu')
    hook = cmh.ModuleHook(policy)

    venv_templ = maze.create_venv(1, seed, 1)
    venv_all, _ = maze.venv_with_all_mouse_positions(venv_templ)
    venv_all_no_cheese = maze.remove_all_cheese(maze.copy_venvs(venv_all))

    obs_cheese = t.tensor(venv_all.reset(), dtype=t.float32)
    obs_no_cheese = t.tensor(venv_all_no_cheese.reset(), dtype=t.float32)


    # Get all patches

    def _get_activations(hook, obs, label=label):
        with hook.set_hook_should_get_custom_data():
            hook.network(obs)
            return hook.get_value_by_label(label)

    cheese_act = _get_activations(hook, obs_cheese)
    no_cheese_act = _get_activations(hook, obs_no_cheese)
    print(cheese_act.shape, no_cheese_act.shape)

    # Mean patch

    mean_cheese_diff = (cheese_act - no_cheese_act).mean(0)

    patches = {label: lambda outp: outp + coeff*mean_cheese_diff} # can't pickle

    original_vfield = vfield.vector_field(venv_templ, hook.network)
    with hook.use_patches(patches):
        patched_vfield = vfield.vector_field(venv_templ, hook.network)

    fig, axs, vf_diff = vfield.plot_vfs(original_vfield, patched_vfield, render_padding=False, ax_size=4, show_diff=True)
    plt.show(fig)


# %%

@interact
def main_interact(seed = IntSlider(min=0, max=30, step=1, value=0)):
    main(seed=seed)


# %%
