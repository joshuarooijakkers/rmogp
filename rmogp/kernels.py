"""Multi-output kernels.

All kernels return covariance matrices in *output-major* order: for inputs
``X1`` (N1 points) and ``X2`` (N2 points) and T outputs, ``kernel(X1, X2)`` has
shape ``(T * N1, T * N2)`` and block ``(t, t')`` holds ``Cov(f_t(X1), f_t'(X2))``.
This matches ``vec(Y)`` for an ``(N, T)`` output matrix ``Y`` (columns stacked).
"""

import numpy as np
import tensorflow as tf

from .base import DTYPE, Module, Parameter, to_tensor


def squared_distance(X1, X2):
    s1 = tf.reduce_sum(tf.square(X1), axis=1, keepdims=True)
    s2 = tf.reduce_sum(tf.square(X2), axis=1, keepdims=True)
    return tf.maximum(s1 + tf.transpose(s2) - 2.0 * tf.matmul(X1, X2, transpose_b=True), 0.0)


def rbf(X1, X2, lengthscale):
    """Unit-variance squared exponential kernel ``exp(-|x - x'|^2 / (2 l^2))``.

    ``lengthscale`` is a scalar or a vector with one entry per input dimension (ARD).
    """
    return tf.exp(-0.5 * squared_distance(X1 / lengthscale, X2 / lengthscale))


def kron(a, b):
    """Kronecker product of two matrices."""
    m, n = tf.shape(a)[0], tf.shape(a)[1]
    p, q = tf.shape(b)[0], tf.shape(b)[1]
    return tf.reshape(a[:, None, :, None] * b[None, :, None, :], [m * p, n * q])


class MultiOutputKernel(Module):
    num_outputs: int

    def __call__(self, X1, X2=None):
        raise NotImplementedError

    def diag(self, X):
        """Diagonal of ``self(X, X)``, shape ``(T * N,)``."""
        raise NotImplementedError

    def output_covariance(self, input_dim=1):
        """``K(x, x)``: the ``T x T`` covariance across outputs at a single input."""
        raise NotImplementedError


class ICM(MultiOutputKernel):
    """Intrinsic coregionalisation model ``K(x, x') = B k(x, x')`` with ``B = A A^T``.

    Args:
        A: ``(T, R)`` mixing matrix, where R is the number of latent functions.
        lengthscale: scalar or ``(d,)`` lengthscale of the squared exponential kernel ``k``.
    """

    def __init__(self, A, lengthscale=1.0):
        A = np.atleast_2d(np.asarray(A, dtype=np.float64))
        self.num_outputs, self.num_latent = A.shape
        self.A = Parameter(A, name="A")
        self.lengthscale = Parameter(lengthscale, positive=True, name="lengthscale")

    @property
    def B(self):
        A = self.A()
        return tf.matmul(A, A, transpose_b=True)

    def base(self, X1, X2):
        return rbf(X1, X2, self.lengthscale())

    def __call__(self, X1, X2=None):
        X2 = X1 if X2 is None else X2
        return kron(self.B, self.base(X1, X2))

    def diag(self, X):
        n = tf.shape(X)[0]
        return tf.repeat(tf.linalg.diag_part(self.B), n)

    def output_covariance(self, input_dim=1):
        return self.B


class LMC(MultiOutputKernel):
    """Linear model of coregionalisation ``K(x, x') = sum_g B_g k_g(x, x')`` with ``B_g = A_g A_g^T``.

    Args:
        As: list of ``(T, R_g)`` mixing matrices, one per component.
        lengthscales: list of lengthscales, one per component.
    """

    def __init__(self, As, lengthscales):
        if len(As) != len(lengthscales):
            raise ValueError("Need one lengthscale per mixing matrix.")
        self.components = [ICM(A, ls) for A, ls in zip(As, lengthscales)]
        self.num_outputs = self.components[0].num_outputs

    def __call__(self, X1, X2=None):
        return tf.add_n([c(X1, X2) for c in self.components])

    def diag(self, X):
        return tf.add_n([c.diag(X) for c in self.components])

    def output_covariance(self, input_dim=1):
        return tf.add_n([c.B for c in self.components])


class ProcessConvolution(MultiOutputKernel):
    """Process convolution kernel with Gaussian smoothing kernels ``G_t(x) = v_t exp(-w_t x^2 / 2)``.

    ``[K(x, x')]_{t,t'} = v_t v_t' (2 pi / (w_t + w_t'))^(d/2) exp(-w_t w_t' |x - x'|^2 / (2 (w_t + w_t')))``.

    Args:
        v: ``(T,)`` amplitudes.
        w: ``(T,)`` positive precisions.
    """

    def __init__(self, v, w):
        self.v = Parameter(np.asarray(v, dtype=np.float64), name="v")
        self.w = Parameter(np.asarray(w, dtype=np.float64), positive=True, name="w")
        self.num_outputs = len(np.atleast_1d(v))

    def _magnitude_and_precision(self, input_dim):
        v, w = self.v(), self.w()
        w_sum = w[:, None] + w[None, :]
        magnitude = v[:, None] * v[None, :] * (2.0 * np.pi / w_sum) ** (input_dim / 2.0)
        precision = w[:, None] * w[None, :] / w_sum
        return magnitude, precision

    def __call__(self, X1, X2=None):
        X2 = X1 if X2 is None else X2
        T = self.num_outputs
        n1, n2 = tf.shape(X1)[0], tf.shape(X2)[0]
        magnitude, precision = self._magnitude_and_precision(X1.shape[1])
        d2 = squared_distance(X1, X2)
        blocks = magnitude[:, :, None, None] * tf.exp(-0.5 * precision[:, :, None, None] * d2[None, None])
        return tf.reshape(tf.transpose(blocks, [0, 2, 1, 3]), [T * n1, T * n2])

    def diag(self, X):
        magnitude, _ = self._magnitude_and_precision(X.shape[1])
        return tf.repeat(tf.linalg.diag_part(magnitude), tf.shape(X)[0])

    def output_covariance(self, input_dim=1):
        return self._magnitude_and_precision(input_dim)[0]


__all__ = ["ICM", "LMC", "ProcessConvolution", "MultiOutputKernel", "rbf", "kron", "to_tensor", "DTYPE"]
