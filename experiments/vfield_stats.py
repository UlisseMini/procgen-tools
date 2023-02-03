# TODO: This should all be in a different file

from procgen_tools import maze, models, metrics
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


def get_decision_probs_patched_and_original(vfields, coeff: float):
    assert coeff in set(vf['coeff'] for vf in vfields)

    decision_probs_original = []
    decision_probs_patched = []

    for vfs in vfields:
        if vfs['coeff'] == coeff and metrics.decision_square(vfs['original_vfield']['grid']) is not None:
            decision_probs_original.append(get_decision_probs(vfs['original_vfield']))
            decision_probs_patched.append(get_decision_probs(vfs['patched_vfield']))
    return np.array(decision_probs_original), np.array(decision_probs_patched)


def plot_decision_probs(decision_probs_original, decision_probs_patched):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    dpo, dpp = decision_probs_original, decision_probs_patched

    dpo = np.stack([dpo[:,0], dpo[:,1], 1-dpo[:,0]-dpo[:,1]], axis=1)
    dpp = np.stack([dpp[:,0], dpp[:,1], 1-dpp[:,0]-dpp[:,1]], axis=1)

    for i, (a, label) in enumerate(zip(ax, ('cheese', 'top right', 'other'))):
        a.set_ylabel('count')
        a.set_xlabel(f'(original - patched) {label} probability')
        a.axvline(x=0, color='red')
        prob_diff = dpo[:,i] - dpp[:,i]
        a.hist(prob_diff, bins=20)
        a.set_title(f'{label} probability')

    return fig