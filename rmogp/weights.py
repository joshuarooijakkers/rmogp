"""Robustness weights for R-MOGP (Section 3.1 of the paper).

Each observation ``y_{i,t}`` gets the inverse multi-quadratic (IMQ) weight

    w_t(x_i, y_i) = beta_t * (1 + ((y_{i,t} - gamma_{i,t}) / c_t)^2)^(-1/2),

with ``beta_t = sigma_t / sqrt(2)``. Everything here works with the *scaled*
weight ``w / beta`` in ``(0, 1]``, which is what enters the posterior.

Weights are computed in NumPy: they are held fixed while hyperparameters are
optimised, so they never need gradients. Missing observations (NaN) get NaN weights.
"""

import numpy as np
from sklearn.covariance import MinCovDet


def conditional_mean(Y, mean, cov, jitter=1e-3):
    """Centering function of R-MOGP: ``E[y_{i,t} | y_{i,-t}]`` under ``N(mean, cov)``.

    Args:
        Y: ``(N, T)`` outputs, may contain NaN. Only observed entries are conditioned on.
        mean: scalar or ``(T,)`` prior mean.
        cov: ``(T, T)`` covariance of ``y_i`` (e.g. ``B + Sigma`` under the ICM).

    Returns:
        ``(N, T)`` conditional means (NaN where ``Y`` is NaN).
    """
    Y = np.asarray(Y, dtype=np.float64)
    N, T = Y.shape
    mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), (T,))
    cov = np.asarray(cov, dtype=np.float64)
    gamma = np.full((N, T), np.nan)
    observed = ~np.isnan(Y)
    for t in range(T):
        rows = np.where(observed[:, t])[0]
        others = np.array([s for s in range(T) if s != t], dtype=int)
        patterns = observed[np.ix_(rows, others)]
        # Rows with the same pattern of observed "other" outputs share one solve.
        for pattern in np.unique(patterns, axis=0):
            idx = rows[np.all(patterns == pattern, axis=1)]
            cond = others[pattern]
            if cond.size == 0:
                gamma[idx, t] = mean[t]
                continue
            C_cc = cov[np.ix_(cond, cond)] + jitter * np.eye(cond.size)
            coef = np.linalg.solve(C_cc, cov[cond, t])
            gamma[idx, t] = mean[t] + (Y[np.ix_(idx, cond)] - mean[cond]) @ coef
    return gamma


def quantile_scale(Y, center, epsilons):
    """Decay rate ``c_t``: the ``(1 - eps_t)`` quantile of ``|y_{i,t} - gamma_{i,t}|`` over i."""
    T = Y.shape[1]
    epsilons = np.broadcast_to(np.asarray(epsilons, dtype=np.float64), (T,))
    resid = np.abs(Y - center)
    return np.array([np.nanquantile(resid[:, t], 1.0 - epsilons[t], method="lower") for t in range(T)])


def imq_weights(Y, center, scale):
    """Scaled IMQ weight ``w / beta`` and ``d/dy log(w^2)`` for every entry of ``Y``.

    Args:
        Y: ``(N, T)`` outputs.
        center: ``(N, T)`` centering values ``gamma``.
        scale: ``(T,)`` or ``(N, T)`` decay rates ``c``.

    Returns:
        ``(w01, grad_log_w2)``, both ``(N, T)``.
    """
    r2 = ((Y - center) / scale) ** 2
    w01 = (1.0 + r2) ** -0.5
    grad_log_w2 = 2.0 * (center - Y) / scale**2 / (1.0 + r2)
    return w01, grad_log_w2


def compute_weights(Y, mean, epsilons, cov=None, weight="rmogp"):
    """Weights used by R-MOGP.

    Args:
        Y: ``(N, T)`` outputs, may contain NaN.
        mean: scalar or ``(T,)`` prior mean.
        epsilons: scalar or ``(T,)`` expected proportion of outliers per output.
        cov: ``(T, T)`` output covariance; required for ``weight="rmogp"``.
        weight: ``"rmogp"`` centres each output on its conditional expectation given the
            other outputs (the weight proposed in the paper). ``"rcgp"`` centres on the
            prior mean, as in the scalar RCGP of Altamirano et al. (2024).

    Returns:
        dict with ``w01``, ``grad_log_w2``, ``center`` (all ``(N, T)``) and ``scale`` (``(T,)``).
    """
    Y = np.asarray(Y, dtype=np.float64)
    T = Y.shape[1]
    if weight == "rmogp":
        if cov is None:
            raise ValueError('weight="rmogp" needs the output covariance `cov`.')
        center = conditional_mean(Y, mean, cov)
    elif weight == "rcgp":
        center = np.where(np.isnan(Y), np.nan, np.broadcast_to(mean, (T,)))
    else:
        raise ValueError(f'Unknown weight {weight!r}; use "rmogp" or "rcgp".')
    scale = quantile_scale(Y, center, epsilons)
    w01, grad = imq_weights(Y, center, scale)
    return {"w01": w01, "grad_log_w2": grad, "center": center, "scale": scale}


def robust_covariance(Y, support_fraction=None, random_state=0):
    """Robust ``T x T`` covariance of the rows of ``Y`` with FastMCD.

    Rows with missing values are dropped. Used to initialise the weights before
    hyperparameter optimisation (Appendix C.1).
    """
    Y = np.asarray(Y, dtype=np.float64)
    complete = Y[~np.isnan(Y).any(axis=1)]
    mcd = MinCovDet(support_fraction=support_fraction, random_state=random_state).fit(complete)
    return mcd.covariance_
