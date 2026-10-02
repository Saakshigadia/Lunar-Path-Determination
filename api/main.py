"""FastAPI service: plan a hazard-aware rover path on a lunar terrain.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lunar.hazard_model import predict_hazard_map, train_model
from lunar.planner import astar, build_cost_map, nearest_free, path_metrics
from lunar.terrain import generate_terrain, slope_degrees

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hazard_rf.joblib"
app = FastAPI(title="Lunar Path Determination API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            _model = joblib.load(MODEL_PATH)
        else:
            _model = train_model()
            MODEL_PATH.parent.mkdir(exist_ok=True)
            joblib.dump(_model, MODEL_PATH)
    return _model


class PlanRequest(BaseModel):
    seed: int = Field(42, description="Which synthetic terrain to generate")
    size: int = Field(128, ge=32, le=256, description="Grid size in cells (5 m per cell)")
    start: Tuple[int, int] = (4, 4)
    goal: Tuple[int, int] = (123, 123)
    max_slope_deg: float = Field(25.0, gt=0, le=45)


class PlanResponse(BaseModel):
    start: Tuple[int, int]
    goal: Tuple[int, int]
    path: List[Tuple[int, int]]
    safe_path: dict
    shortest_path: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest):
    for name, (r, c) in (("start", req.start), ("goal", req.goal)):
        if not (0 <= r < req.size and 0 <= c < req.size):
            raise HTTPException(422, f"{name} {req.start if name == 'start' else req.goal} is outside a {req.size}x{req.size} map")
    t = generate_terrain(size=req.size, seed=req.seed)
    slope = slope_degrees(t.dem, t.cell_size)
    hazard_p = predict_hazard_map(get_model(), t.dem, t.cell_size)
    cost = build_cost_map(slope, hazard_p, max_slope=req.max_slope_deg)
    start, goal = nearest_free(cost, req.start), nearest_free(cost, req.goal)
    safe = astar(cost, start, goal, t.cell_size)
    shortest = astar(np.ones_like(slope), start, goal, t.cell_size)
    return PlanResponse(start=start, goal=goal, path=[(int(r), int(c)) for r, c in safe],
                        safe_path=path_metrics(safe, slope, t.hazard, t.cell_size),
                        shortest_path=path_metrics(shortest, slope, t.hazard, t.cell_size))
