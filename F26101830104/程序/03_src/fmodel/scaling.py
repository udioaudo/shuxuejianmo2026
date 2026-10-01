"""Classical scaling law baseline with fixed units and local diagnostics."""

import numpy as np
from scipy.optimize import least_squares


def classical_loss(N, D, params, Nref=1e8, Dref=1e9):
    N, D = np.asarray(N, float), np.asarray(D, float)
    pars = np.asarray(params, float)
    if pars.shape != (5,) or not np.isfinite(pars).all() or (pars < 0).any():
        raise ValueError("expected nonnegative E,a,b,alpha,beta")
    if (pars[1:] <= 0).any() or Nref <= 0 or Dref <= 0:
        raise ValueError("positive coefficients, exponents and reference scales required")
    if not np.isfinite([Nref, Dref]).all():
        raise ValueError("reference scales must be finite")
    if not np.isfinite(N).all() or not np.isfinite(D).all() or (N <= 0).any() or (D <= 0).any():
        raise ValueError("N,D must be positive finite counts")
    E, a, b, alpha, beta = pars
    return E + a * (N/Nref)**(-alpha) + b * (D/Dref)**(-beta)


def fit_classical(N, D, loss, *, Nref=1e8, Dref=1e9, starts=12, seed=2026):
    N, D, y = np.asarray(N, float), np.asarray(D, float), np.asarray(loss, float)
    if N.ndim != 1 or N.shape != D.shape or y.shape != N.shape or len(y) < 8:
        raise ValueError("at least eight matched observations required")
    if starts < 1 or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("invalid starts or loss")
    classical_loss(N, D, [0, 1, 1, .3, .3], Nref, Dref)
    # Bounds are baseline implementation choices and must be reviewed for real data.
    lower = [0, 1e-10, 1e-10, .001, .001]
    upper = [float(y.min())*(1-1e-10), np.inf, np.inf, 3, 3]
    rng = np.random.default_rng(seed)
    runs, fits = [], []
    for _ in range(starts):
        initial = [rng.uniform(.05,.8)*y.min(), rng.uniform(.1,2)*y.mean(),
                   rng.uniform(.1,2)*y.mean(), rng.uniform(.05,.8), rng.uniform(.05,.8)]
        result = least_squares(
            lambda p: classical_loss(N,D,p,Nref,Dref)-y, initial,
            bounds=(lower,upper), x_scale="jac", max_nfev=4000,
            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        runs.append({"success": bool(result.success), "cost": float(result.cost),
                     "nfev": int(result.nfev)})
        if result.success and np.isfinite(result.cost):
            fits.append(result)
    if not fits:
        raise RuntimeError("all fitting starts failed")
    result = min(fits, key=lambda x: x.cost)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    tol = np.finfo(float).eps * max(result.jac.shape) * singular[0]
    condition = float(singular[0]/singular[-1]) if singular[-1] > tol else None
    return {"model_type": "classical_baseline", "params": result.x.tolist(),
            "Nref": Nref, "Dref": Dref, "runs": runs,
            "training_rmse": float(np.sqrt(2*result.cost/len(y))),
            "jacobian_rank": int((singular > tol).sum()), "jacobian_condition": condition,
            "at_parameter_bound": bool(np.any(result.active_mask != 0)),
            "evidence_status": "fit_only_requires_external_validation",
            "limits": "No quality or mixture effects; no uncertainty or causal identification."}


# ---------------------------------------------------------------------------
# Q2 quality term. Main form (selected on B6 by BIC, see q2/quality_scenario.py):
#     h(N,D,Q) = c (N/Nref)^-theta_N (D/Dref)^-theta_D (1-Q)^kappa
# theta_N = theta_D = 0 recovers the scale-free additive term c (1-Q)^kappa,
# and h = 0 at Q = 1, so the generalized law reduces to the classical law.

def quality_params(quality):
    """Normalise a saved quality-model JSON to (c, kappa, theta_N, theta_D, Nref, Dref)."""
    kappa = quality.get('kappa', quality.get('gamma'))
    return (float(quality['c']), float(kappa), float(quality.get('theta_N', 0.0)),
            float(quality.get('theta_D', 0.0)), float(quality.get('Nref', 1e9)),
            float(quality.get('Dref', 1e9)))


def quality_penalty(N, D, Q, quality):
    """Loss added by imperfect data quality; N, D in raw counts."""
    c, kappa, tN, tD, Nref, Dref = quality_params(quality)
    N, D, Q = np.asarray(N, float), np.asarray(D, float), np.asarray(Q, float)
    return c * (N/Nref)**(-tN) * (D/Dref)**(-tD) * np.maximum(1-Q, 0)**kappa


def quality_penalty_grads(N, D, Q, quality):
    """Return (N dh/dN, D dh/dD, dh/dQ): scale derivatives in log form, Q derivative plain."""
    c, kappa, tN, tD, Nref, Dref = quality_params(quality)
    h = quality_penalty(N, D, Q, quality)
    one_minus = np.maximum(1-np.asarray(Q, float), 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        dQ = np.where(one_minus > 0,
                      -c*kappa*(np.asarray(N, float)/Nref)**(-tN)*(np.asarray(D, float)/Dref)**(-tD)
                      * one_minus**(kappa-1), 0.0)
    return -tN*h, -tD*h, dQ


def generalized_loss(N, D, Q, params, quality, Nref=1e9, Dref=1e9):
    """Classical B1 law plus the quality term (mixture fixed at p_ref)."""
    return classical_loss(N, D, params, Nref, Dref) + quality_penalty(N, D, Q, quality)
