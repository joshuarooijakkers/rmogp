"""Outlier-robust multi-output Gaussian processes (R-MOGP)."""

from .kernels import ICM, LMC, ProcessConvolution
from .models import MOGP, RMOGP
from .sparse import RSVMOGP, SVMOGP
from .weights import compute_weights, robust_covariance

__all__ = ["MOGP", "RMOGP", "SVMOGP", "RSVMOGP", "ICM", "LMC", "ProcessConvolution",
           "compute_weights", "robust_covariance"]
__version__ = "0.1.0"
