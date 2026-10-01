"""Problem-defined costs. No claim of trained-model performance."""

import math
from scipy.optimize import minimize_scalar

ETA = 2e-4


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def quality_cost(Q, kind):
    if not math.isfinite(Q) or not 0 < Q <= 1:
        raise ValueError("Q must be in (0,1]")
    if kind == "exponential":
        return 1e7 * math.exp(6*Q)
    if kind == "power":
        return 5e9 * Q**4
    if kind == "logarithmic":
        return 2e9 * math.log1p(10*Q)
    raise ValueError("unknown quality cost family")


def compute_costs(N, D, Q, Q0, Lctx, kind):
    for name,value in [("N",N),("D",D),("Lctx",Lctx)]:
        _positive(value,name)
    ctrain = 6*N*D
    cq = D * max(quality_cost(Q,kind)-quality_cost(Q0,kind),0)
    cattn = ETA*N*D*Lctx
    result = {"train": ctrain, "quality": cq, "attention": cattn,
              "total": ctrain+cq+cattn}
    if not all(math.isfinite(x) for x in result.values()):
        raise ValueError("cost overflow")
    return result


def critical_context_length():
    return 6/ETA


def analytical_classic_allocation(C, a, b, alpha, beta):
    """Interior solution in ORIGINAL units for E+a*N^-alpha+b*D^-beta.

    Training-only cost 6ND=C; no quality, context or box constraints.
    """
    for name,value in [("C",C),("a",a),("b",b),("alpha",alpha),("beta",beta)]:
        _positive(value,name)
    logK = math.log(C/6)
    logN = (math.log(alpha*a/(beta*b)) + beta*logK)/(alpha+beta)
    return {"N": math.exp(logN), "D": math.exp(logK-logN)}


def optimize_classic_fixed_context(C, Lctx, a, b, alpha, beta, N_bounds, D_bounds):
    """Numerical check only: fixed Q=Q0 and fixed p; not full Q3 solver."""
    for name,value in [("C",C),("Lctx",Lctx),("a",a),("b",b),("alpha",alpha),("beta",beta)]:
        _positive(value,name)
    for bounds in (N_bounds,D_bounds):
        if len(bounds)!=2 or not 0 < bounds[0] < bounds[1] or not all(map(math.isfinite,bounds)):
            raise ValueError("finite positive increasing bounds required")
    K = C/(6+ETA*Lctx)
    low = max(N_bounds[0],K/D_bounds[1])
    high = min(N_bounds[1],K/D_bounds[0])
    if low > high:
        raise ValueError("no feasible budget-saturating configuration in supplied bounds")
    def objective(logN):
        N = math.exp(logN)
        return a*N**(-alpha)+b*(K/N)**(-beta)
    candidates = [math.log(low), math.log(high)]
    if low < high:
        fit = minimize_scalar(objective,bounds=(math.log(low),math.log(high)),
                              method="bounded",options={"xatol":1e-12})
        if not fit.success:
            raise RuntimeError("scalar optimizer failed")
        candidates.append(float(fit.x))
    logN = min(candidates,key=objective)
    N,D = math.exp(logN),K/math.exp(logN)
    return {"N":N,"D":D,"excess_loss":objective(logN),
            "relative_budget_residual":((6+ETA*Lctx)*N*D-C)/C,
            "scope":"classical fixed-quality fixed-mixture numerical check"}
