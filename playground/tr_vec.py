# %%
try:
    import procgen_tools
except ImportError:
    commit = ""
    get_ipython().run_line_magic(
        magic_name="pip",
        line=(
            "install -U"
            f" git+https://github.com/ulissemini/procgen-tools.git@{commit}"
        ),
    )

from procgen_tools.utils import setup

setup()  # create directory structure

# %%
from procgen_tools.imports import *
from procgen_tools import visualization, patch_utils, maze, vfield

# %%
AX_SIZE = 5.5

# %% [markdown]
# # Generating top-right vectors
# We're going to generate a new kind of behavior-modifying
# activation-space vector. This vector will be a top-right vector. We
# compute it by diffing activations across two environments: one where
# the top-right corner is considerably higher up, and the original one.

# %%
@interact
def top_right_maze_pairs(seed=IntSlider(value=0, min=0, max=100)):
    venv = maze.get_top_right_venv_pair(seed=seed)
    fig, axs = plt.subplots(1, 2, figsize=(AX_SIZE, AX_SIZE * 2))
    for idx in (0, 1):
        visualization.visualize_venv(
            venv=venv,
            idx=idx,
            ax=axs[idx],
            ax_size=AX_SIZE,
            render_padding=True,
        )

    # Title each axis
    axs[0].set_title("Path to top-right")
    axs[1].set_title("Original maze")

    plt.show()


# %% [markdown]
# When the top-right-most reachable square is closer to the absolute
# top-right, the agent seems to have an increased tendency to go to
# there. (Arrow thickness is a technical artifact here, just look at directions.)


# %%


@interact
def top_right_vf_pairs(seed=IntSlider(value=0, min=0, max=100)):
    venv = maze.get_top_right_venv_pair(seed=seed)
    fig, axs = plt.subplots(1, 2, figsize=(10, 10 * 2))
    for idx in (0, 1):
        vfield = visualization.vector_field(venv, policy, idx=idx)
        visualization.plot_vf(vfield, ax=axs[idx], render_padding=True)

    # Title each axis
    axs[0].set_title("Path to top-right")
    axs[1].set_title("Original maze")

    plt.show()


# %%
@interact
def examine_tr_patch(
    target_seed=IntSlider(min=0, max=100, step=1, value=0),
    coeff=FloatSlider(min=-5, max=5, step=0.1, value=1),
):
    venv_pair = maze.get_top_right_venv_pair(seed=target_seed)
    patch = patch_utils.patch_from_venv_pair(
        venv_pair, layer_name=default_layer, hook=hook, coeff=coeff
    )

    # Show the effect of the patch
    target_venv = maze.create_venv(
        num=1, start_level=target_seed, num_levels=1
    )
    fig, axs, info = patch_utils.compare_patched_vfields(
        target_venv,
        patch,
        hook,
        render_padding=False,
        ax_size=AX_SIZE,
        show_components=False,
    )
    plt.show(fig)


# %%
@interact
def examine_tr_patch_transfer(
    target_seed=IntSlider(min=0, max=100, step=1, value=0),
    coeff=FloatSlider(min=-5, max=5, step=0.1, value=1),
):
    venv_pair = maze.get_top_right_venv_pair(seed=0)
    patch = patch_utils.patch_from_venv_pair(
        venv_pair, layer_name=default_layer, hook=hook, coeff=coeff
    )

    # Show the effect of the patch
    target_venv = maze.create_venv(
        num=1, start_level=target_seed, num_levels=1
    )
    fig, axs, info = patch_utils.compare_patched_vfields(
        target_venv,
        patch,
        hook,
        render_padding=False,
        ax_size=AX_SIZE,
        show_components=False,
    )
    plt.show(fig)


# %%
@interact
def compose_patches(
    target_seed=IntSlider(min=0, max=100, step=1, value=0),
    top_right_coeff=FloatSlider(min=-5, max=5, step=0.1, value=1),
    use_cheese_vector=Checkbox(value=True),
    use_tr_vector=Checkbox(value=True),
):
    patch_list = []
    if use_tr_vector:
        tr_venv_pair = maze.get_top_right_venv_pair(seed=target_seed)
        tr_patch = patch_utils.patch_from_venv_pair(
            tr_venv_pair,
            layer_name=default_layer,
            hook=hook,
            coeff=top_right_coeff,
        )
        patch_list.append(tr_patch)

    if use_cheese_vector:
        cheese_diff_values = patch_utils.cheese_diff_values(
            seed=target_seed, layer_name=default_layer, hook=hook
        )
        cheese_patch = patch_utils.get_values_diff_patch(
            values=cheese_diff_values, coeff=-1, layer_name=default_layer
        )
        patch_list.append(cheese_patch)

    patch = patch_utils.compose_patches(*patch_list) if patch_list else {}

    target_venv = maze.create_venv(
        num=1, start_level=target_seed, num_levels=1
    )
    fig, axs, info = patch_utils.compare_patched_vfields(
        target_venv, patch, hook, render_padding=False, ax_size=AX_SIZE
    )

    # Title which patches we're using
    title = "Patches: "
    if use_tr_vector:
        title += "Top-right vector, "
    if use_cheese_vector:
        title += "Cheese vector, "
    title = title[:-2]  # Remove trailing comma
    fig.suptitle(title, fontsize=20)

    plt.show(fig)

    # Generate a save button
    button = visualization.create_save_button(
        prefix=f"figures/vec_composition_gif/",
        fig=fig,
        descriptors=dict(
            target_seed=target_seed,
            top_right_coeff=top_right_coeff,
            use_cheese_vector=use_cheese_vector,
            use_tr_vector=use_tr_vector,
        ),
    )
    display(button)


# %%
