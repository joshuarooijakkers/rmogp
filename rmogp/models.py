"""Exact multi-output GP regression: the standard MOGP and the robust R-MOGP.

Both models take outputs as an ``(N, T)`` array ``Y``; missing entries (NaN) are
allowed, which covers heterotopic data. Internally the outputs are stacked as
``vec(Y)`` (output-major), matching the kernels in :mod:`rmogp.kernels`.

The R-MOGP posterior (Proposition 3.1) is

    mu_*    = m_* + k_*^T (K + Sigma J_W)^{-1} (y - m_W),
    Sigma_* = k_** - k_*^T (K + Sigma J_W)^{-1} k_*,

with ``J_W = diag(w01^-2)`` and ``m_W = m + Sigma * d/dy log(w^2)``. With all
weights equal to one (``J_W = I``, ``m_W = m``) it reduces to the standard MOGP,
which is how :class:`MOGP` is implemented.
"""

import numpy as np
import tensorflow as tf

from . import optim
from .base import DTYPE, Module, Parameter, to_tensor
from .weights import compute_weights, robust_covariance


class _ExactMOGP(Module):
    def __init__(self, kernel, noise_var=0.1, mean=0.0, shared_noise=False, jitter=1e-6):
        self.kernel = kernel
        self.num_outputs = T = kernel.num_outputs
        self.shared_noise = shared_noise
        noise_var = np.asarray(noise_var, dtype=np.float64)
        noise_var = np.atleast_1d(noise_var.mean() if shared_noise else np.broadcast_to(noise_var, (T,)))
        self.noise_var = Parameter(noise_var, positive=True, name="noise_var")
        self.mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), (T,)).copy()
        self.jitter = jitter

    # ----------------------------------------------------------------- helpers
    def noise_variances(self):
        """Noise variance per output, shape ``(T,)``."""
        return tf.broadcast_to(self.noise_var(), [self.num_outputs])

    def _set_data(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[1] != self.num_outputs:
            raise ValueError(f"Y has {Y.shape[1]} outputs but the kernel has {self.num_outputs}.")
        self.X, self.Y = X, Y
        self.N = X.shape[0]
        y_vec = Y.T.reshape(-1)
        self._obs = np.where(~np.isnan(y_vec))[0]
        self._X = to_tensor(X)
        self._y = to_tensor(y_vec[self._obs])
        self._mean_vec = to_tensor(np.repeat(self.mean, self.N)[self._obs])

    def _vec_obs(self, A):
        """``(N, T)`` array -> tensor of its observed entries in ``vec`` order."""
        return to_tensor(np.asarray(A, dtype=np.float64).T.reshape(-1)[self._obs])

    def _system(self, w01, grad_log_w2):
        """``K + Sigma J_W`` and ``m_W`` restricted to the observed entries."""
        K = tf.gather(tf.gather(self.kernel(self._X), self._obs, axis=0), self._obs, axis=1)
        noise = tf.gather(tf.repeat(self.noise_variances(), self.N), self._obs)
        J = w01 ** -2
        Kw = K + tf.linalg.diag(noise * J + self.jitter)
        m_w = self._mean_vec + noise * grad_log_w2
        return Kw, m_w, noise, J

    def _unit_weights(self):
        return tf.ones_like(self._y), tf.zeros_like(self._y)

    def _model_weights(self):
        """Weights used by the posterior; overridden by R-MOGP."""
        return self._unit_weights()

    def _update_posterior(self):
        w01, grad = self._model_weights()
        Kw, m_w, _, _ = self._system(w01, grad)
        self._L = tf.linalg.cholesky(Kw)
        self._alpha = tf.linalg.cholesky_solve(self._L, (self._y - m_w)[:, None])

    # --------------------------------------------------------------- public API
    def fit(self, X, Y):
        """Condition the model on inputs ``X`` ``(N, d)`` and outputs ``Y`` ``(N, T)``."""
        self._set_data(X, Y)
        self._update_posterior()
        return self

    def predict_f(self, Xs, full_cov=False):
        """Posterior of the latent functions at ``Xs``.

        Returns:
            ``mean`` of shape ``(Ns, T)`` and either the marginal variances ``(Ns, T)``
            or, with ``full_cov=True``, the full ``(T * Ns, T * Ns)`` covariance in
            output-major order.
        """
        Xs = to_tensor(np.atleast_2d(np.asarray(Xs, dtype=np.float64).reshape(len(Xs), -1)))
        T, Ns = self.num_outputs, Xs.shape[0]
        Ks = tf.gather(self.kernel(self._X, Xs), self._obs, axis=0)
        mean = tf.matmul(Ks, self._alpha, transpose_a=True)[:, 0] + tf.repeat(to_tensor(self.mean), Ns)
        v = tf.linalg.triangular_solve(self._L, Ks, lower=True)
        if full_cov:
            cov = self.kernel(Xs) - tf.matmul(v, v, transpose_a=True)
            return tf.transpose(tf.reshape(mean, [T, Ns])).numpy(), cov.numpy()
        var = self.kernel.diag(Xs) - tf.reduce_sum(tf.square(v), axis=0)
        to_matrix = lambda a: tf.transpose(tf.reshape(a, [T, Ns])).numpy()
        return to_matrix(mean), np.maximum(to_matrix(var), 0.0)

    def predict_y(self, Xs):
        """Predictive mean and variance of new observations (latent variance + noise)."""
        mean, var = self.predict_f(Xs)
        return mean, var + self.noise_variances().numpy()[None, :]

    def loo_log_predictive(self, w01=None, grad_log_w2=None, objective_weights=None):
        """Closed-form leave-one-out log predictive densities (Lemma B.3), summed.

        Args:
            w01, grad_log_w2: ``(N, T)`` weights used inside the posterior. Defaults to
                unit weights (standard MOGP).
            objective_weights: ``(N, T)`` weight per LOO term. Defaults to one.
        """
        w = self._unit_weights() if w01 is None else (self._vec_obs(w01), self._vec_obs(grad_log_w2))
        u = 1.0 if objective_weights is None else self._vec_obs(objective_weights)
        return self._loo(w[0], w[1], u)

    def _loo(self, w01, grad, objective_weights):
        Kw, m_w, noise, J = self._system(w01, grad)
        L = tf.linalg.cholesky(Kw)
        n = tf.shape(Kw)[0]
        L_inv = tf.linalg.triangular_solve(L, tf.eye(n, dtype=DTYPE), lower=True)
        inv_diag = tf.reduce_sum(tf.square(L_inv), axis=0)
        z = self._y - m_w
        inv_z = tf.linalg.cholesky_solve(L, z[:, None])[:, 0]
        loo_mean = z + self._mean_vec - inv_z / inv_diag
        loo_var = 1.0 / inv_diag - noise * J + noise
        log_p = -0.5 * tf.math.log(2.0 * np.pi * loo_var) - 0.5 * tf.square(self._y - loo_mean) / loo_var
        return tf.reduce_sum(objective_weights * log_p)

    def _optimize(self, loss_fn, maxiter, tol):
        res = optim.lbfgs(loss_fn, self.trainable_variables(), maxiter=maxiter, tol=tol)
        self._update_posterior()
        return res


class MOGP(_ExactMOGP):
    """Standard multi-output GP with Gaussian noise (independent across outputs).

    Args:
        kernel: a multi-output kernel from :mod:`rmogp.kernels`.
        noise_var: initial noise variance, scalar or ``(T,)``.
        mean: constant prior mean, scalar or ``(T,)`` (fixed, not optimised).
        shared_noise: use a single noise variance for all outputs.
    """

    def log_marginal_likelihood(self):
        Kw, m_w, _, _ = self._system(*self._unit_weights())
        L = tf.linalg.cholesky(Kw)
        r = (self._y - m_w)[:, None]
        alpha = tf.linalg.cholesky_solve(L, r)
        n = tf.cast(tf.shape(r)[0], DTYPE)
        return (
            -0.5 * tf.reduce_sum(r * alpha)
            - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
            - 0.5 * n * np.log(2.0 * np.pi)
        )

    def optimize(self, objective="mll", loo_weights=None, maxiter=None, tol=None):
        """Optimise kernel hyperparameters and noise.

        Args:
            objective: ``"mll"`` for the log marginal likelihood or ``"loo"`` for the
                leave-one-out log predictive.
            loo_weights: optional ``(N, T)`` weights for the LOO terms (``objective="loo"`` only).

        Returns:
            The SciPy optimisation result.
        """
        if objective == "mll":
            loss = lambda: -self.log_marginal_likelihood()
        elif objective == "loo":
            u = 1.0 if loo_weights is None else self._vec_obs(loo_weights)
            loss = lambda: -self._loo(*self._unit_weights(), u)
        else:
            raise ValueError(f'Unknown objective {objective!r}; use "mll" or "loo".')
        return self._optimize(loss, maxiter, tol)


class RMOGP(_ExactMOGP):
    """Robust and conjugate multi-output GP (R-MOGP).

    Args:
        kernel: a multi-output kernel from :mod:`rmogp.kernels`.
        noise_var: initial noise variance, scalar or ``(T,)``.
        mean: constant prior mean, scalar or ``(T,)`` (fixed, not optimised).
        epsilons: expected proportion of outliers per output, scalar or ``(T,)``.
        weight: ``"rmogp"`` (default) centres the IMQ weight of each output on its
            conditional expectation given the other outputs. ``"rcgp"`` centres it on the
            prior mean, which is the naive extension of RCGP.
        shared_noise: use a single noise variance for all outputs.
    """

    def __init__(self, kernel, noise_var=0.1, mean=0.0, epsilons=0.05, weight="rmogp",
                 shared_noise=False, jitter=1e-6):
        super().__init__(kernel, noise_var=noise_var, mean=mean, shared_noise=shared_noise, jitter=jitter)
        self.epsilons = np.broadcast_to(np.asarray(epsilons, dtype=np.float64), (self.num_outputs,)).copy()
        self.weight = weight

    def output_covariance(self):
        """Covariance of ``y_i`` across outputs: ``K(x, x) + Sigma``."""
        C = self.kernel.output_covariance(self.X.shape[1]) + tf.linalg.diag(self.noise_variances())
        return C.numpy()

    def compute_weights(self, cov=None):
        """Weights for the current data. ``cov`` defaults to :meth:`output_covariance`."""
        if self.weight == "rmogp" and cov is None:
            cov = self.output_covariance()
        return compute_weights(self.Y, self.mean, self.epsilons, cov=cov, weight=self.weight)

    def _model_weights(self):
        self.weights_ = self.compute_weights()
        return self._vec_obs(self.weights_["w01"]), self._vec_obs(self.weights_["grad_log_w2"])

    @property
    def weights(self):
        """Scaled weights ``w / beta`` in ``(0, 1]`` from the last fit, shape ``(N, T)``."""
        return self.weights_["w01"]

    def optimize(self, init_cov=None, weight_power=2, maxiter=None, tol=None):
        """Optimise hyperparameters with the weighted LOO-CV objective (Section 3.3).

        Following Appendix C.1, the weights are first computed with a robust estimate of
        the output covariance and kept fixed during optimisation; afterwards they are
        recomputed with the optimised hyperparameters.

        Args:
            init_cov: ``(T, T)`` covariance used for the fixed weights. Defaults to a
                FastMCD estimate on ``Y`` (see :func:`rmogp.weights.robust_covariance`).
                Not used with ``weight="rcgp"``.
            weight_power: the LOO term of ``y_{i,t}`` is weighted by ``(w_{i,t} / beta_t)^power``.
                ``2`` is the paper's choice, ``0`` gives the unweighted LOO.

        Returns:
            The SciPy optimisation result.
        """
        if self.weight == "rmogp" and init_cov is None:
            init_cov = robust_covariance(self.Y)
        fixed = compute_weights(self.Y, self.mean, self.epsilons, cov=init_cov, weight=self.weight)
        w01 = self._vec_obs(fixed["w01"])
        grad = self._vec_obs(fixed["grad_log_w2"])
        u = w01 ** weight_power
        return self._optimize(lambda: -self._loo(w01, grad, u), maxiter, tol)


__all__ = ["MOGP", "RMOGP"]
