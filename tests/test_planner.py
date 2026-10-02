import numpy as np
from fastapi.testclient import TestClient

from lunar.features import FEATURE_NAMES, compute_features
from lunar.planner import astar, build_cost_map, nearest_free
from lunar.terrain import generate_terrain


def test_straight_line_on_flat_ground():
    cost = np.ones((10, 10))
    path = astar(cost, (0, 0), (0, 9))
    assert path[0] == (0, 0) and path[-1] == (0, 9)
    assert len(path) == 10


def test_avoids_blocked_cells():
    cost = np.ones((10, 10))
    cost[:9, 5] = np.inf                      # wall with a gap at the bottom
    path = astar(cost, (0, 0), (0, 9))
    assert path and all(np.isfinite(cost[p]) for p in path)
    assert (9, 5) in path


def test_unreachable_returns_empty():
    cost = np.ones((5, 5))
    cost[:, 2] = np.inf
    assert astar(cost, (0, 0), (0, 4)) == []


def test_cost_map_blocks_steep_and_hazardous_cells():
    slope = np.array([[5.0, 30.0], [10.0, 5.0]])
    hazard = np.array([[0.0, 0.0], [0.95, 0.1]])
    cost = build_cost_map(slope, hazard)
    assert np.isinf(cost[0, 1]) and np.isinf(cost[1, 0])
    assert cost[0, 0] >= 1 and np.isfinite(cost[1, 1])


def test_nearest_free_snaps_out_of_obstacle():
    cost = np.ones((5, 5)); cost[0, 0] = np.inf
    assert np.isfinite(cost[nearest_free(cost, (0, 0))])


def test_features_shape_and_reproducible_terrain():
    a, b = generate_terrain(size=64, seed=3), generate_terrain(size=64, seed=3)
    assert np.allclose(a.dem, b.dem)
    assert compute_features(a.dem, a.cell_size).shape == (64, 64, len(FEATURE_NAMES))


def test_api_plan_endpoint():
    from api.main import app
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/plan", json={"seed": 42, "size": 64, "start": [2, 2], "goal": [60, 60]})
    assert r.status_code == 200
    body = r.json()
    assert body["safe_path"]["reachable"] and body["path"][0] == list(body["start"])
