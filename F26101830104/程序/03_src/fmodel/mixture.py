"""Training-only ridge baseline for a fixed ordered simplex."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error


def validate_simplex(p, dimension=17, tolerance=1e-8):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != dimension or p.shape[0] == 0:
        raise ValueError(f"expected nonempty n by {dimension} mixture table")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("mixtures must be finite and nonnegative")
    if not np.allclose(p.sum(axis=1), 1, atol=tolerance, rtol=0):
        raise ValueError("mixture rows must sum to one")
    return p


class MixtureRidge:
    def __init__(self, dimension=17):
        self.dimension = dimension
        self.model = None

    def fit(self, p, losses, groups, alphas=(0.0001, 0.01, 1.0), folds=5):
        p = validate_simplex(p, self.dimension)
        y, groups = np.asarray(losses, float), np.asarray(groups)
        if y.shape != (len(p),) or groups.shape != y.shape or not np.isfinite(y).all():
            raise ValueError("invalid targets or group identifiers")
        unique = len(np.unique(groups))
        if unique < 2 or folds < 2:
            raise ValueError("need at least two independent groups for tuning")
        if not alphas or not all(np.isfinite(a) and a > 0 for a in alphas):
            raise ValueError("positive finite ridge penalties required")
        # Drop a reference domain so an intercept does not duplicate sum(p)=1.
        x = p[:, :-1]
        splits = list(GroupKFold(n_splits=min(folds, unique)).split(x, y, groups))
        history = []
        for alpha in alphas:
            predictions = np.empty_like(y)
            for train, valid in splits:
                model = Ridge(alpha=alpha).fit(x[train], y[train])
                predictions[valid] = model.predict(x[valid])
            history.append({"alpha": float(alpha),
                            "internal_group_cv_mae": float(mean_absolute_error(y, predictions))})
        best = min(history, key=lambda r: r["internal_group_cv_mae"])
        self.model = Ridge(alpha=best["alpha"]).fit(x, y)
        self.history = history
        self.selected_alpha = best["alpha"]
        return self

    def predict(self, p):
        if self.model is None:
            raise ValueError("model not fitted")
        return self.model.predict(validate_simplex(p, self.dimension)[:, :-1])


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise ValueError("matching nonempty vectors required")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("metrics cannot silently discard missing values")
    return {"n": len(y), "mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction)))}
