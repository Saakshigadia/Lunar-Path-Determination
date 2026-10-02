"""A* path planning over a terrain cost map."""
import heapq
import math

import numpy as np

NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def build_cost_map(slope, hazard_prob, max_slope=25.0, slope_weight=4.0,
                   hazard_weight=8.0, hazard_block=0.8):
    """Cost multiplier per cell (always >= 1). np.inf marks cells the rover must never enter."""
    cost = 1.0 + slope_weight * (slope / max_slope) ** 2 + hazard_weight * hazard_prob
    cost[(slope > max_slope) | (hazard_prob > hazard_block)] = np.inf
    return cost


def astar(cost, start, goal, cell_size=1.0):
    """Return a list of (row, col) cells from start to goal, or [] if unreachable.

    Step cost = distance x average cost of the two cells. The heuristic is the
    straight-line distance, which never overestimates because every cost is >= 1,
    so A* returns the optimal path for this cost map.
    """
    h, w = cost.shape
    if not (np.isfinite(cost[start]) and np.isfinite(cost[goal])):
        return []
    g = {start: 0.0}
    parent = {}
    heap = [(0.0, start)]
    closed = set()
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        r, c = cur
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not np.isfinite(cost[nr, nc]):
                continue
            step = math.hypot(dr, dc) * cell_size * 0.5 * (cost[r, c] + cost[nr, nc])
            ng = g[cur] + step
            if ng < g.get((nr, nc), math.inf):
                g[(nr, nc)] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(heap, (ng + math.hypot(goal[0] - nr, goal[1] - nc) * cell_size, (nr, nc)))
    return []


def path_metrics(path, slope, hazard_truth, cell_size):
    if not path:
        return {"reachable": False}
    p = np.array(path)
    length = float(np.sum(np.hypot(*np.diff(p, axis=0).T)) * cell_size)
    s = slope[p[:, 0], p[:, 1]]
    return {
        "reachable": True,
        "length_m": round(length, 1),
        "max_slope_deg": round(float(s.max()), 1),
        "mean_slope_deg": round(float(s.mean()), 1),
        "hazard_cells_crossed": int(hazard_truth[p[:, 0], p[:, 1]].sum()),
    }


def nearest_free(cost, point):
    """Snap a requested start/goal to the closest safe cell in the largest connected
    traversable region, so the rover never starts inside an isolated pocket."""
    from scipy import ndimage
    free = np.isfinite(cost)
    labels, n = ndimage.label(free, structure=np.ones((3, 3)))
    if n == 0:
        raise ValueError("No traversable cells on this map")
    biggest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    cells = np.argwhere(labels == biggest)
    d = np.hypot(cells[:, 0] - point[0], cells[:, 1] - point[1])
    r, c = cells[np.argmin(d)]
    return int(r), int(c)
