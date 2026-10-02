"""Train the hazard model, benchmark safe vs shortest paths, and save figures.

Run from the project root:  python scripts/run_demo.py
Outputs go to ./outputs (figures, metrics.json) and ./models (trained model).
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lunar.hazard_model import evaluate, predict_hazard_map, train_model  # noqa: E402
from lunar.planner import astar, build_cost_map, nearest_free, path_metrics  # noqa: E402
from lunar.terrain import generate_terrain, slope_degrees  # noqa: E402
from lunar.render import lunar_surface, smooth_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, MODELS = ROOT / "outputs", ROOT / "models"
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)


def plan_both(model, seed, size=128):
    t = generate_terrain(size=size, seed=seed)
    slope = slope_degrees(t.dem, t.cell_size)
    hazard_p = predict_hazard_map(model, t.dem, t.cell_size)
    cost = build_cost_map(slope, hazard_p)
    start, goal = nearest_free(cost, (4, 4)), nearest_free(cost, (size - 5, size - 5))
    safe = astar(cost, start, goal, t.cell_size)
    shortest = astar(np.ones_like(slope), start, goal, t.cell_size)   # ignores terrain completely
    return t, slope, hazard_p, safe, shortest


def main():
    print("Training hazard model on 8 synthetic terrains...")
    model = train_model()
    joblib.dump(model, MODELS / "hazard_rf.joblib")

    model_metrics = evaluate(model)
    print("Hazard model on unseen terrains:", model_metrics)

    print("Benchmarking on 20 unseen terrains...")
    rows = []
    for seed in range(100, 120):
        t, slope, _, safe, short = plan_both(model, seed)
        rows.append({"seed": seed, "safe": path_metrics(safe, slope, t.hazard, t.cell_size),
                     "shortest": path_metrics(short, slope, t.hazard, t.cell_size)})
    ok = [r for r in rows if r["safe"]["reachable"]]

    def avg(k, which):
        return round(float(np.mean([r[which][k] for r in ok])), 1)

    summary = {
        "maps": len(rows), "safe_path_found": len(ok),
        "hazard_cells_crossed": {"shortest": avg("hazard_cells_crossed", "shortest"), "safe": avg("hazard_cells_crossed", "safe")},
        "max_slope_deg": {"shortest": avg("max_slope_deg", "shortest"), "safe": avg("max_slope_deg", "safe")},
        "length_m": {"shortest": avg("length_m", "shortest"), "safe": avg("length_m", "safe")},
    }
    summary["extra_distance_pct"] = round(100 * (summary["length_m"]["safe"] / summary["length_m"]["shortest"] - 1), 1)
    print(json.dumps(summary, indent=2))
    json.dump({"hazard_model": model_metrics, "benchmark": summary, "per_map": rows},
              open(OUT / "metrics.json", "w"), indent=2)

    # ---- figures for one example map
    t, slope, hazard_p, safe, short = plan_both(model, 42)
    s, sh = np.array(safe), np.array(short)

    sc = 4
    surface = lunar_surface(t.dem, t.cell_size, scale=sc)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.imshow(surface)
    sp, shp = smooth_path(s) * sc + sc / 2, smooth_path(sh) * sc + sc / 2
    ax.plot(shp[:, 1], shp[:, 0], color="#ff5a5a", lw=1.8, ls=(0, (5, 3)), label="Shortest path (drives through hazards)")
    ax.plot(sp[:, 1], sp[:, 0], color="black", lw=5, alpha=0.35)
    ax.plot(sp[:, 1], sp[:, 0], color="#39e58c", lw=2.6, label="Planned safe path")
    ax.scatter([sp[0, 1]], [sp[0, 0]], s=80, c="white", edgecolors="black", zorder=5, label="Start")
    ax.scatter([sp[-1, 1]], [sp[-1, 0]], s=80, c="#ffd84d", edgecolors="black", zorder=5, label="Goal")
    leg = ax.legend(loc="lower left", facecolor="black", edgecolor="#555", labelcolor="white", framealpha=0.6, fontsize=9)
    ax.text(0.98, 0.98, "640 m x 640 m lunar terrain", transform=ax.transAxes, color="white", fontsize=9, va="top", ha="right",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=4))
    ax.axis("off")
    fig.savefig(OUT / "path_comparison.png", dpi=130, bbox_inches="tight", pad_inches=0, facecolor="black"); plt.close(fig)

    shade = lunar_surface(t.dem, t.cell_size, scale=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(shade); axes[0].set_title("Terrain")
    axes[1].imshow(t.hazard, cmap="magma"); axes[1].set_title("True hazards")
    im = axes[2].imshow(hazard_p, cmap="magma", vmin=0, vmax=1); axes[2].set_title("Predicted hazard probability")
    for a in axes: a.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.savefig(OUT / "hazard_prediction.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("Saved figures and metrics to", OUT)


if __name__ == "__main__":
    main()
