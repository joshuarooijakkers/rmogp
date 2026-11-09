"""Evaluation metrics (Appendix C.2). Missing targets (NaN) are ignored."""

import numpy as np


def rmse(Y_true, Y_pred):
    """Root mean squared error over all observed entries."""
    err = np.asarray(Y_true) - np.asarray(Y_pred)
    return float(np.sqrt(np.nanmean(err**2)))


def nlpd(Y_true, mean, var, min_var=1e-10):
    """Average negative log predictive density under independent Gaussian predictives."""
    var = np.maximum(np.asarray(var), min_var)
    nll = 0.5 * np.log(2.0 * np.pi * var) + 0.5 * (np.asarray(Y_true) - np.asarray(mean)) ** 2 / var
    return float(np.nanmean(nll))
