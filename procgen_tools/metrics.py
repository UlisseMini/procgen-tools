import numpy as np
from typing import Optional, Tuple, Dict, Callable
from procgen_tools import maze

# Helpers

def decision_square(mgrid: np.ndarray) -> Optional[Tuple[int, int]]:
    "Get the decision square (square where agent can choose to go to cheese or top right) if it exists"

    _, came_from, _ = maze.shortest_path(mgrid, (0, 0))
    path_to_cheese = maze.reconstruct_path(came_from, maze.get_cheese_pos(mgrid))

    tr = (mgrid.shape[0]-1, mgrid.shape[1]-1)
    _, came_from, _ = maze.shortest_path(mgrid, (0, 0), stop_condition=lambda _, c: c == tr)
    path_to_top_right = maze.reconstruct_path(came_from, tr)
    # if the paths diverge they can never connect again, since mazes have no cycles
    for i, (n1, n2) in enumerate(zip(path_to_cheese, path_to_top_right)):
        # print(f'step {i}/{min(len(path_to_cheese), len(path_to_top_right))}: {n1} {n2}')
        if n1 != n2:
            return path_to_cheese[i - 1]
    return None


def shortest_path_to_nxn(grid: np.ndarray, start: Tuple[int, int], n: int):
    "Wrapper around shortest_path to find the shortest path to the top right NxN square"
    top_right_5x5 = {(grid.shape[0]-1-i, grid.shape[0]-1-j) for i in range(n) for j in range(n)}
    return maze.shortest_path(grid, start, stop_condition=lambda _,c: c in top_right_5x5, heuristic=lambda *_: 1)


def get_dsq(grid: np.ndarray) -> Tuple[int, int]:
    "Get the decision square and assert it exists"
    d_sq = decision_square(grid)
    assert d_sq is not None
    return d_sq


def path_len(cost, _, extra) -> int:
    "path_len(maze.shortest_path(...)) gives the path length of the shortest path."
    return cost[extra["last_square"]]


def distance(s1: Tuple[int,int], s2: Tuple[int, int], p=2) -> float:
    return (abs(s1[0]-s2[0])**p + abs(s1[1]-s2[1])**p)**(1/p)


def dist_cheese_nxn(grid: np.ndarray, start: Tuple[int, int], n: int, _filter=lambda g,c: g[c] != maze.BLOCKED, p=2) -> float:
    x, y = start
    return min(
        (abs(x-i)**p + abs(y-j)**p)**(1/p)
        for i in range(grid.shape[0]-n, grid.shape[0]) for j in range(grid.shape[1]-n, grid.shape[1])
        if _filter(grid, (i, j))
    )


"""
Metrics from Peli:
> Euclidean distance between cheese and decision-square
> Number of steps between cheese and decision-square
> Euclidean distance between cheese and top-right
> Number of steps between cheese and top-right
> Euclidean distance between decision-square and top-right
> Number of steps between decision-square and top-right
> Shortest Euclidean distance between cheese and a square in the top-right 5*5 region
> Smallest number of steps between cheese and a square in the top-right 5*5 region
> Shortest Euclidean distance between decision-square and a square in the top-right 5*5 region
> Smallest number of steps between decision-square and a square in the top-right 5*5 region
"""

MetricFn = Callable[[np.ndarray], float]
metrics: Dict[str, MetricFn]  = {}

def metric(fn: MetricFn):
    metrics[fn.__name__] = fn
    return fn

# All of Peli's metrics

@metric
def euc_dist_cheese_decision_square(grid: np.ndarray) -> float:
    "Euclidean distance between cheese and decision-square"
    return distance(maze.get_cheese_pos(grid), get_dsq(grid))


@metric
def taxi_dist_cheese_decision_square(grid: np.ndarray) -> float:
    "Taxicab (L1) distance between cheese and decision-square"
    return distance(maze.get_cheese_pos(grid), get_dsq(grid), p=1)


@metric
def steps_between_cheese_decision_square(grid: np.ndarray) -> int:
    "Number of steps between cheese and decision-square"
    return path_len(*maze.shortest_path(grid, get_dsq(grid)))


@metric
def steps_between_cheese_top_right(grid: np.ndarray) -> int:
    "Number of steps between cheese and top-right"
    return path_len(*maze.shortest_path(grid, start=(grid.shape[0]-1, grid.shape[1]-1)))


@metric
def euc_dist_cheese_top_right(grid: np.ndarray) -> float:
    "Euclidean distance between cheese and top-right"
    return distance(maze.get_cheese_pos(grid), (grid.shape[0]-1, grid.shape[1]-1))


@metric
def taxi_dist_cheese_top_right(grid: np.ndarray) -> float:
    "Taxicab (L1) distance between cheese and top-right"
    return distance(maze.get_cheese_pos(grid), (grid.shape[0]-1, grid.shape[1]-1), p=1)


@metric
def euc_dist_decision_square_top_right(grid: np.ndarray) -> float:
    "Euclidean distance between decision-square and top-right"
    return distance(get_dsq(grid), (grid.shape[0]-1, grid.shape[1]-1))


@metric
def taxi_dist_decision_square_top_right(grid: np.ndarray) -> float:
    "Taxicab (L1) distance between decision-square and top-right"
    return distance(get_dsq(grid), (grid.shape[0]-1, grid.shape[1]-1), p=1)


@metric
def steps_between_decision_square_top_right(grid: np.ndarray) -> int:
    "Number of steps between decision-square and top-right"
    dsq = get_dsq(grid)
    tr = (grid.shape[0]-1, grid.shape[1]-1)
    return path_len(*maze.shortest_path(grid, start=tr, stop_condition=lambda _,c: c == dsq))


@metric
def steps_between_cheese_5x5(grid: np.ndarray) -> int:
    "Smallest number of steps between cheese and a square in the top-right 5*5 region"
    return path_len(*shortest_path_to_nxn(grid, maze.get_cheese_pos(grid), n=5))


@metric
def steps_between_decision_square_5x5(grid: np.ndarray) -> int:
    "Smallest number of steps between decision-square and a square in the top-right 5*5 region"
    return path_len(*shortest_path_to_nxn(grid, get_dsq(grid), n=5))


@metric
def euc_dist_cheese_5x5(grid: np.ndarray, **kwargs) -> float:
    "Shortest Euclidean distance between cheese and a square in the top-right 5*5 region"
    return dist_cheese_nxn(grid, maze.get_cheese_pos(grid), n=5, **kwargs)


@metric
def taxi_dist_cheese_5x5(grid: np.ndarray, **kwargs) -> float:
    "Shortest Taxicab (L1) distance between cheese and a square in the top-right 5*5 region"
    return dist_cheese_nxn(grid, maze.get_cheese_pos(grid), n=5, p=1, **kwargs)


@metric
def euc_dist_decision_square_5x5(grid: np.ndarray, **kwargs) -> float:
    "Shortest Euclidean distance between decision-square and a square in the top-right 5*5 region"
    return dist_cheese_nxn(grid, get_dsq(grid), n=5, **kwargs)


@metric
def taxi_dist_decision_square_5x5(grid: np.ndarray, **kwargs) -> float:
    "Shortest Taxicab (L1) distance between decision-square and a square in the top-right 5*5 region"
    return dist_cheese_nxn(grid, get_dsq(grid), n=5, p=1, **kwargs)

@metric
def cheese_coord_norm(grid: np.ndarray, **kwargs) -> float:
    "Norm of the outer grid coordinates of cheese."
    outer_grid = maze.outer_grid(grid)
    return np.linalg.norm( maze.get_cheese_pos(outer_grid))
