"""Explicit, reproducible scoring primitives; official field rules are required."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Indicator:
    name: str
    direction: int
    lower: float
    upper: float
    weight: float = 1.0

    def __post_init__(self):
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        if not np.isfinite([self.lower, self.upper, self.weight]).all():
            raise ValueError("indicator parameters must be finite")
        if self.upper <= self.lower or self.weight <= 0:
            raise ValueError("require upper > lower and positive weight")


def score_quality(values, indicators, min_coverage=1.0, clip=False):
    """Fixed scale scoring. Never fits normalization on a held-out data set.

    Missing = NaN; insufficient rows receive NaN, not zero or a silent deletion.
    If incomplete scoring is enabled, observed weights are renormalized and the
    coverage is reported. Bounds and clipping are choices requiring justification.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not indicators or x.shape[1] != len(indicators):
        raise ValueError("expected rows by configured indicators")
    if len({i.name for i in indicators}) != len(indicators):
        raise ValueError("duplicate indicator names")
    if not 0 < min_coverage <= 1 or np.isinf(x).any():
        raise ValueError("invalid coverage or infinite observation")
    lo = np.array([i.lower for i in indicators])
    hi = np.array([i.upper for i in indicators])
    weights = np.array([i.weight for i in indicators])
    weights = weights / weights.sum()
    observed = np.isfinite(x)
    out_of_bounds = observed & ((x < lo) | (x > hi))
    if out_of_bounds.any() and not clip:
        raise ValueError("observed value outside fixed normalization bounds")
    z = (x - lo) / (hi - lo)
    if clip:
        z = np.clip(z, 0, 1)
    z[:, np.array([i.direction < 0 for i in indicators])] = (
        1 - z[:, np.array([i.direction < 0 for i in indicators])]
    )
    coverage = observed @ weights
    total = np.nansum(z * weights, axis=1)
    scores = np.full(x.shape[0], np.nan)
    valid = (coverage >= min_coverage - 1e-12) & (coverage > 0)
    scores[valid] = total[valid] / coverage[valid]
    status = np.where(valid, "scored", "insufficient_coverage")
    return {"scores": scores, "coverage": coverage, "status": status,
            "normalized": z, "clipped_values": int(out_of_bounds.sum())}


def semantic_conflict(normalized, positive_index, cleanliness_index,
                      high_threshold=0.8, low_threshold=0.2):
    """One explicit candidate: high value AND low cleanliness, not generic variance."""
    x = np.asarray(normalized, dtype=float)
    if x.ndim != 2 or positive_index == cleanliness_index:
        raise ValueError("two distinct indicator indices required")
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("invalid thresholds")
    a, b = x[:, positive_index], x[:, cleanliness_index]
    valid = np.isfinite(a) & np.isfinite(b)
    return {"evaluable": valid,
            "conflict": valid & (a >= high_threshold) & (b <= low_threshold)}


def summarize_domains(scores, domains, weights=None):
    q = np.asarray(scores, dtype=float)
    domains = np.asarray(domains, dtype=str)
    if q.ndim != 1 or domains.shape != q.shape or np.isinf(q).any():
        raise ValueError("invalid scores or domain labels")
    w = np.ones(len(q)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != q.shape or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("aggregation weights must be positive and finite")
    rows = []
    for domain in sorted(set(domains)):
        mask = domains == domain
        valid = mask & np.isfinite(q)
        rows.append({"domain": domain, "n_total": int(mask.sum()),
                     "n_scored": int(valid.sum()), "n_missing": int((mask & ~valid).sum()),
                     "quality_mean": float(np.average(q[valid], weights=w[valid]))
                     if valid.any() else None})
    return rows
