"""Train and evaluate a Random Forest that predicts hazard probability per cell."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .features import FEATURE_NAMES, compute_features
from .terrain import generate_terrain

N = len(FEATURE_NAMES)


def build_dataset(seeds, size=128, sample_per_map=6000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    X, y = [], []
    for s in seeds:
        t = generate_terrain(size=size, seed=s)
        f = compute_features(t.dem, t.cell_size).reshape(-1, N)
        lab = t.hazard.ravel()
        idx = rng.choice(len(lab), size=min(sample_per_map, len(lab)), replace=False)
        X.append(f[idx]); y.append(lab[idx])
    return np.vstack(X), np.concatenate(y)


def train_model(train_seeds=range(1, 9), size=128) -> RandomForestClassifier:
    X, y = build_dataset(train_seeds, size)
    model = RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced",
                                   n_jobs=-1, random_state=0)
    model.fit(X, y)
    return model


def predict_hazard_map(model, dem, cell_size) -> np.ndarray:
    f = compute_features(dem, cell_size)
    h, w, n = f.shape
    return model.predict_proba(f.reshape(-1, n))[:, 1].reshape(h, w)


def evaluate(model, test_seeds=(42, 43, 44), size=128) -> dict:
    """Evaluate on terrains the model has never seen during training."""
    probs, labels = [], []
    for s in test_seeds:
        t = generate_terrain(size=size, seed=s)
        probs.append(predict_hazard_map(model, t.dem, t.cell_size).ravel())
        labels.append(t.hazard.ravel())
    p, y = np.concatenate(probs), np.concatenate(labels)
    pred = p > 0.5
    return {
        "precision": round(float(precision_score(y, pred)), 3),
        "recall": round(float(recall_score(y, pred)), 3),
        "f1": round(float(f1_score(y, pred)), 3),
        "roc_auc": round(float(roc_auc_score(y, p)), 3),
    }
