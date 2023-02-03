from ipywidgets import interact, IntSlider, FloatSlider
import matplotlib.pyplot as plt

def plot_vfs(vfs):
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    for i, vf in enumerate((vfs['original_vfield'], vfs['patched_vfield'])):
        legal_mouse_positions, arrows, grid = vf['legal_mouse_positions'], vf['arrows'], vf['grid']

        ax[i].set_xlabel("Original vfield" if i == 0 else "Patched vfield")
        ax[i].quiver(
            [x[1] for x in legal_mouse_positions], [x[0] for x in legal_mouse_positions],
            [x[1] for x in arrows], [x[0] for x in arrows], color='red',
        )
        ax[i].imshow(grid, origin='lower')

    plt.title(f"Seed {vfs['seed']}, coeff {vfs['coeff']}")
    return fig

