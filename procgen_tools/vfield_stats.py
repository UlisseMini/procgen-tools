from procgen_tools import maze, models, metrics, vfield
import numpy as np
import matplotlib.pyplot as plt

def _decision_square_index(vf):
    return vf['legal_mouse_positions'].index(metrics.decision_square(vf['grid']))


def _pathfind(grid: np.ndarray, start, end):
    cost, came_from, extra = maze.shortest_path(grid, start, stop_condition=lambda _, sq: sq == end)
    return maze.reconstruct_path(came_from, extra['last_square'])


def deltas_from(grid: np.ndarray, sq):
    path_to_cheese = _pathfind(grid, sq, maze.get_cheese_pos(grid))
    path_to_top_right = _pathfind(grid, sq, (grid.shape[0]-1, grid.shape[1]-1))
    delta_cheese = (path_to_cheese[1][0] - sq[0], path_to_cheese[1][1] - sq[1])
    delta_tr = (path_to_top_right[1][0] - sq[0], path_to_top_right[1][1] - sq[1])
    return delta_cheese, delta_tr


def get_decision_probs(vf):
    "From the decision square, return probabilities of going to the cheese or top right."

    grid = vf['grid']
    i = _decision_square_index(vf)
    probs_dict = {k: v for k,v in zip(models.MAZE_ACTION_INDICES.keys(), vf['probs'][i])}
    action_delta = models.MAZE_ACTION_DELTAS[max(probs_dict, key=probs_dict.get)]

    dsq = metrics.decision_square(grid)
    csq = maze.get_cheese_pos(grid)

    delta_cheese, delta_tr = deltas_from(grid, dsq)
    cheese_dir = models.MAZE_ACTION_DELTAS.inverse[delta_cheese]
    topright_dir = models.MAZE_ACTION_DELTAS.inverse[delta_tr]
    # if topright_dir in ('RIGHT', 'UP'):
    #     raise ValueError('filtering by mazes where the mouse has to go down/left to get to the top right')
    return probs_dict[cheese_dir], probs_dict[topright_dir]


def get_decision_probs_original_and_patched(vfields, coeff: float):
    assert coeff in set(vf['coeff'] for vf in vfields)

    decision_probs_original = []
    decision_probs_patched = []

    for vfs in vfields:
        if vfs['coeff'] == coeff and metrics.decision_square(vfs['original_vfield']['grid']) is not None:
            decision_probs_original.append(get_decision_probs(vfs['original_vfield']))
            decision_probs_patched.append(get_decision_probs(vfs['patched_vfield']))
    return np.array(decision_probs_original), np.array(decision_probs_patched)


def plot_decision_probs(decision_probs_original, decision_probs_patched, ax_size : int = 4):
    fig, ax = plt.subplots(1, 3, figsize=(3*ax_size, ax_size))
    dpo, dpp = decision_probs_original, decision_probs_patched

    dpo = np.stack([dpo[:,0], dpo[:,1], 1-dpo[:,0]-dpo[:,1]], axis=1)
    dpp = np.stack([dpp[:,0], dpp[:,1], 1-dpp[:,0]-dpp[:,1]], axis=1)

    for i, (a, label) in enumerate(zip(ax, ('cheese', 'top right', 'other'))):
        a.set_ylabel('count')
        a.hist(dpo[:,i], bins=20, label='original', alpha=0.5)
        a.hist(dpp[:,i], bins=20, label='patched', alpha=0.5)
        a.legend()
        a.set_title(f'{label} probability')

    return fig



_plot_labels = ('cheese', 'top-right', 'other')

def plot_dprobs_hist(dpo, dpp, ax=None):
    """ Plot histograms of the decision probabilities for the original and patched vfields. """
    assert len(ax) == len(_plot_labels) == 3

    ax[0].set_ylabel('count')
    for i in range(3):
        ax[i].set_title([f'P({label} | decision-square)' for label in _plot_labels][i])
        ax[i].set_xlabel('probability')
        ax[i].hist(dpo[:,i], bins=20, label='original', alpha=0.5)
        ax[i].hist(dpp[:,i], bins=20, label='patched', alpha=0.5)
        ax[i].legend()


def plot_dprobs_scatter(dpo, dpp, ax=None):
    """ Plot scatter plots of the decision probabilities for the original and patched vfields. """
    assert len(ax) == len(_plot_labels) == 3
    ax[0].set_ylabel('patched')
    for i in range(3):
        ax[i].set_title([f'P({label} | decision-square)' for label in _plot_labels][i])
        ax[i].scatter(dpo[:,i], dpp[:,i], alpha=0.5)
        ax[i].set_xlabel('original')


def plot_dprobs_box(dpo, dpp, ax=None):
    """ Plot box plots of the decision probabilities for the original and patched vfields. """
    assert len(ax) == len(_plot_labels) == 3
    ax[0].set_ylabel('probability')
    for i in range(3):
        ax[i].boxplot([dpo[:,i], dpp[:,i]], labels=['original', 'patched'])
        ax[i].set_xticks([1, 2])
        ax[i].set_xticklabels(['original', 'patched'])

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple

def plotly_fig_gen() -> go.Figure:
    return make_subplots(rows=1, cols=3, subplot_titles=['P(cheese | decision-square)', 'P(top-right | decision-square)', 'P(other | decision-square)'], shared_yaxes=True)

def get_probs_original_and_patched(vfields : List[dict], coeff : float) -> Tuple[np.ndarray, np.ndarray]:
    """ Get the original and patched decision probabilities from vfields, given a cheese vector coefficient. """
    dprobs_original, dprobs_patched = get_decision_probs_original_and_patched(vfields, coeff=coeff)
    probs_original, probs_patched = [np.stack([dprobs[:,0], dprobs[:,1], 1-dprobs[:,0]-dprobs[:,1]], axis=1) for dprobs in (dprobs_original, dprobs_patched)] # convert to 3-class probs

    # Clip all probabilities to [0,1]
    probs_original = np.clip(probs_original, 0, 1)
    probs_patched = np.clip(probs_patched, 0, 1)
    return probs_original, probs_patched

def format_fig(fig : go.Figure, coeff : float):
    fig.update_layout(showlegend=False)
    fig.update_layout(title_text=f'Cheese vector coefficient: {coeff}')

def histogram_plotly(coeff : float, vfields : List[dict], fig : go.Figure = plotly_fig_gen()):
    """ Plot decision probabilities, given a cheese vector coefficient. Plot the cheese, top-right, and other probabilities in three separate plots. """
    probs_original, probs_patched = get_probs_original_and_patched(vfields, coeff=coeff)
    
    for i in range(3):
        # Make the original colored blue and the patched colored orange
        fig.add_trace(go.Histogram(x=probs_original[:,i], name='original', marker_color='blue', histnorm='probability'), row=1, col=i+1)
        fig.add_trace(go.Histogram(x=probs_patched[:,i], name='patched', marker_color='orange', histnorm='probability'), row=1, col=i+1)

    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(title_text="Probability at decision square", row=1, col=2)

    # Bound x to [0,1]
    for i in range(3):
        fig.update_xaxes(range=[0,1], row=1, col=i+1)

    format_fig(fig=fig, coeff=coeff)
    return fig

def scatterplot_plotly(coeff : float, vfields : List[dict], fig : go.Figure = plotly_fig_gen()):
    """ Plot decision probabilities, given a cheese vector coefficient. Plot the cheese, top-right, and other probabilities in three separate plots. """
    probs_original, probs_patched = get_probs_original_and_patched(vfields, coeff=coeff)

    for i in range(3):
        # Make the original colored blue and the patched colored orange
        fig.add_trace(go.Scatter(x=probs_original[:,i], y=probs_patched[:,i], mode='markers', name='original', marker_color='blue', hovertext=[f'seed: {vfields[idx]["seed"]}, (original {probs_original[idx,i]:.3f}, patched {probs_patched[idx,i]:.3f})' for idx in range(len(probs_original))], opacity=0.6), row=1, col=i+1) 

        # Format the hovertext to show the seed, and also the original and patched probabilities to 3 decimal places
        fig.update_traces(hovertemplate='%{hovertext}<extra></extra>', hoverlabel=dict(bgcolor='white'), hoverlabel_align='left', hoverlabel_font_size=14, hoverlabel_font_family='monospace', hoverlabel_font_color='black') 

    # Label left-most y-axis as "patched" and central x-axis as "original"
    fig.update_yaxes(title_text='patched', row=1, col=1)
    fig.update_xaxes(title_text='original', row=1, col=2)

    format_fig(fig=fig, coeff=coeff)
    return fig

def boxplot_plotly(coeff : float, vfields : List[dict], fig : go.Figure = plotly_fig_gen()):
    """ Plot boxplots of decision probabilities, given a cheese vector coefficient. Plot the cheese, top-right, and other probabilities in three separate plots. """
    probs_original, probs_patched = get_probs_original_and_patched(vfields, coeff=coeff)

    for i in range(3):
        # Make the original colored blue and the patched colored orange
        fig.add_trace(go.Box(y=probs_original[:,i], name='original', marker_color='blue'), row=1, col=i+1)
        fig.add_trace(go.Box(y=probs_patched[:,i], name='patched', marker_color='orange'), row=1, col=i+1)
    
    format_fig(fig=fig, coeff=coeff)
    return fig