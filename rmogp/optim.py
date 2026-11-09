"""Optimisers: SciPy L-BFGS-B with TensorFlow gradients, and Adam."""

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from .base import DTYPE


def lbfgs(loss_fn, variables, maxiter=None, tol=None):
    """Minimise ``loss_fn()`` over ``variables`` with L-BFGS-B.

    Gradients come from TensorFlow autodiff. If a Cholesky factorisation fails at
    some trial point, the objective returns a large value so the line search backs off.

    Returns:
        The ``scipy.optimize.OptimizeResult`` (e.g. ``res.nfev`` for the number of
        function evaluations).
    """
    shapes = [v.shape for v in variables]
    sizes = [int(np.prod(s)) for s in shapes]

    def unpack(x):
        out, i = [], 0
        for shape, size in zip(shapes, sizes):
            out.append(np.reshape(x[i : i + size], shape))
            i += size
        return out

    @tf.function
    def value_and_grad():
        with tf.GradientTape() as tape:
            loss = loss_fn()
        return loss, tape.gradient(loss, variables)

    def func(x):
        for var, value in zip(variables, unpack(x)):
            var.assign(value)
        try:
            loss, grads = value_and_grad()
        except tf.errors.InvalidArgumentError:  # e.g. matrix not positive definite
            return 1e10, np.zeros_like(x)
        grads = [np.zeros(s) if g is None else g.numpy() for g, s in zip(grads, shapes)]
        flat = np.concatenate([np.ravel(g) for g in grads])
        if not np.isfinite(loss.numpy()) or not np.all(np.isfinite(flat)):
            return 1e10, np.zeros_like(x)
        return float(loss.numpy()), flat

    x0 = np.concatenate([np.ravel(v.numpy()) for v in variables])
    options = {} if maxiter is None else {"maxiter": maxiter}
    res = minimize(func, x0, jac=True, method="L-BFGS-B", tol=tol, options=options)
    for var, value in zip(variables, unpack(res.x)):
        var.assign(value)
    return res


def adam(loss_fn, variables, steps=1000, learning_rate=1e-2):
    """Minimise ``loss_fn()`` with full-batch Adam. Returns the loss history."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    return np.array([float(step().numpy()) for _ in range(steps)], dtype=np.float64)


__all__ = ["lbfgs", "adam", "DTYPE"]
