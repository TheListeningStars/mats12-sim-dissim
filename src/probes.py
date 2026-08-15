"""Deception probes: logistic-regression and difference-of-means.

A probe separates 'deceptive/framed' (label 1) from 'honest' (label 0) activations
within a given training cell. Trained probes are reused by transfer.py across cells.
Scores are affine (acts @ w + b) so a probe transfers to any activation matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config

# L2 strength for the probe. Recorded in run_meta.json rather than buried here:
# features are ~4k-d and training pools are ~40-90 rows, so this is a load-bearing
# hyperparameter, not a default.
LOGREG_C = 1.0


@dataclass
class Probe:
    w: np.ndarray            # weight direction (scaler folded in for logreg)
    b: float                 # bias
    layer: int
    kind: str                # "logreg" | "diffmean"

    def score(self, acts: np.ndarray) -> np.ndarray:
        """Decision scores; higher = more deceptive/framed."""
        return acts @ self.w + self.b


def train_logreg(acts: np.ndarray, labels: np.ndarray, layer: int) -> Probe:
    """L2 logistic regression on standardized features, fixed C, seeded.
    The scaler is folded into (w, b) so scoring needs no fit-time state."""
    scaler = StandardScaler().fit(acts)
    clf = LogisticRegression(C=LOGREG_C, max_iter=2000, random_state=config.SEED)
    clf.fit(scaler.transform(acts), labels)
    coef = clf.coef_[0]
    w = coef / scaler.scale_
    b = float(clf.intercept_[0] - (scaler.mean_ / scaler.scale_) @ coef)
    return Probe(w=w, b=b, layer=layer, kind="logreg")


def train_diffmean(acts: np.ndarray, labels: np.ndarray, layer: int) -> Probe:
    """Difference-of-means direction between classes; often more robust/causal."""
    mu1, mu0 = acts[labels == 1].mean(0), acts[labels == 0].mean(0)
    w = mu1 - mu0
    w = w / np.linalg.norm(w)
    b = float(-(mu1 @ w + mu0 @ w) / 2)
    return Probe(w=w, b=b, layer=layer, kind="diffmean")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def random_direction_auroc(acts: np.ndarray, labels: np.ndarray, n: int = 20,
                           seed: int = config.SEED) -> float:
    """Baseline floor (PLAN §8): mean AUROC of random unit-vector 'probes'.
    Expected ~0.5; reported raw (not folded), so it is an honest floor."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        v = rng.standard_normal(acts.shape[1])
        vals.append(roc_auc_score(labels, acts @ (v / np.linalg.norm(v))))
    return float(np.mean(vals))


if __name__ == "__main__":
    print("probes.py is a library; used by transfer.py / baselines.py")
