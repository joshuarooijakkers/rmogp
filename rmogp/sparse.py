"""Sparse variational (R-)MOGP with inducing points (Section 3.4, Proposition 3.3).

The model uses the ICM: ``f_t(x) = sum_r A_{t,r} u_r(x)`` with R independent latent
GPs ``u_r``. Each latent GP gets the same Q inducing inputs ``Z``, which are kept fixed.
The optimal variational distribution is available in closed form, so only the
kernel hyperparameters and the noise are optimised, by maximising the ELBO.
"""

import numpy as np
import tensorflow as tf

from . import optim
from .base import DTYPE, Module, Parameter, to_tensor
from .kernels import ICM, kron, rbf
from .weights import compute_weights


class SVMOGP(Module):
    """Sparse variational MOGP with an ICM kernel and Gaussian noise.

    Args:
        kernel: an :class:`rmogp.kernels.ICM` kernel.
        Z: ``(Q, d)`` inducing inputs, shared by the R latent functions.
        noise_var: initial noise variance, scalar or ``(T,)``.
        mean: constant prior mean, scalar or ``(T,)`` (fixed).
        shared_noise: use a single noise variance for all outputs.
    """

    def __init__(self, kernel, Z, noise_var=0.1, mean=0.0, shared_noise=False, jitter=1e-6):
        if not isinstance(kernel, ICM):
            raise TypeError("The sparse models need an ICM kernel.")
        self.kernel = kernel
        self.num_outputs = T = kernel.num_outputs
        self.Z = to_tensor(np.asarray(Z, dtype=np.float64).reshape(len(Z), -1))
        noise_var = np.asarray(noise_var, dtype=np.float64)
        noise_var = np.atleast_1d(noise_var.mean() if shared_noise else np.broadcast_to(noise_var, (T,)))
        self.noise_var = Parameter(noise_var, positive=True, name="noise_var")
        self.mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), (T,)).copy()
        self.jitter = jitter

    def noise_variances(self):
        return tf.broadcast_to(self.noise_var(), [self.num_outputs])

    def _compute_weights(self):
        """Weights ``(w01, grad_log_w2)`` as ``(N, T)`` arrays; unit weights for the standard model."""
        return np.ones_like(self.Y), np.zeros_like(self.Y)

    def fit(self, X, Y):
        """Set the data, compute the weights and the optimal variational posterior."""
        X = np.asarray(X, dtype=np.float64)
        self.X = X[:, None] if X.ndim == 1 else X
        self.Y = np.asarray(Y, dtype=np.float64)
        self.N = self.X.shape[0]
        observed = ~np.isnan(self.Y)
        w01, grad = self._compute_weights()
        vec = lambda A: to_tensor(np.where(observed, A, 0.0).T.reshape(-1))
        self._X = to_tensor(self.X)
        self._y = vec(self.Y)
        self._mask = vec(observed.astype(np.float64))  # 1 for observed, 0 for missing
        self._w2 = vec(w01) ** 2
        self._grad = vec(grad)
        self._mean_vec = tf.repeat(to_tensor(self.mean), self.N)
        self._update_posterior()
        return self

    def _terms(self):
        """Matrices shared by the ELBO and the predictions."""
        A, ls = self.kernel.A(), self.kernel.lengthscale()
        Q, R = tf.shape(self.Z)[0], self.kernel.num_latent
        Kzz = rbf(self.Z, self.Z, ls) + self.jitter * tf.eye(Q, dtype=DTYPE)
        K_FU = kron(A, rbf(self._X, self.Z, ls))  # (T N, R Q)
        K_UU = kron(tf.eye(R, dtype=DTYPE), Kzz)
        noise = tf.repeat(self.noise_variances(), self.N)
        # Sigma^-1 J_W^-1 = w01^2 / sigma^2; zero for missing entries.
        prec = self._mask * self._w2 / noise
        m_w = self._mean_vec + noise * self._grad
        nu = prec * (self._y - m_w)
        P_U = K_UU + tf.matmul(K_FU * prec[:, None], K_FU, transpose_a=True)
        L_P = tf.linalg.cholesky(P_U + self.jitter * tf.eye(tf.shape(P_U)[0], dtype=DTYPE))
        return dict(A=A, Kzz=Kzz, K_FU=K_FU, K_UU=K_UU, noise=noise, prec=prec, nu=nu, L_P=L_P)

    def _update_posterior(self):
        t = self._terms()
        rhs = tf.matmul(t["K_FU"], t["nu"][:, None], transpose_a=True)
        self._alpha = tf.linalg.cholesky_solve(t["L_P"], rhs)  # P_U^-1 K_FU^T nu
        self._L_P = t["L_P"]
        self._L_zz = tf.linalg.cholesky(t["Kzz"])

    def elbo(self):
        """Evidence lower bound of Proposition 3.3 (up to an additive constant)."""
        t = self._terms()
        L_UU = tf.linalg.cholesky(t["K_UU"])
        k_ff = self.kernel.diag(self._X)
        V = tf.linalg.triangular_solve(L_UU, tf.transpose(t["K_FU"]), lower=True)
        trace = tf.reduce_sum(t["prec"] * (k_ff - tf.reduce_sum(tf.square(V), axis=0)))
        resid = self._y - self._mean_vec
        data_fit = tf.reduce_sum(t["prec"] * tf.square(resid))
        log_noise = tf.reduce_sum(self._mask * tf.math.log(t["noise"]))
        quad = tf.reduce_sum(tf.square(tf.linalg.triangular_solve(
            t["L_P"], tf.matmul(t["K_FU"], t["nu"][:, None], transpose_a=True), lower=True)))
        log_det = 2.0 * (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_UU)))
                         - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(t["L_P"]))))
        return -0.5 * (trace + data_fit + log_noise) + 0.5 * quad + 0.5 * log_det

    def optimize(self, method="adam", steps=2000, learning_rate=1e-2, maxiter=None, tol=None):
        """Maximise the ELBO with the weights from the last :meth:`fit` held fixed, then refit.

        Args:
            method: ``"adam"`` (``steps``, ``learning_rate``) or ``"lbfgs"`` (``maxiter``, ``tol``).

        Returns:
            The loss history for Adam or the SciPy result for L-BFGS-B.
        """
        loss = lambda: -self.elbo()
        if method == "adam":
            out = optim.adam(loss, self.trainable_variables(), steps=steps, learning_rate=learning_rate)
        elif method == "lbfgs":
            out = optim.lbfgs(loss, self.trainable_variables(), maxiter=maxiter, tol=tol)
        else:
            raise ValueError(f'Unknown method {method!r}; use "adam" or "lbfgs".')
        self.fit(self.X, self.Y)
        return out

    def predict_f(self, Xs):
        """Posterior mean and marginal variance of the latent functions, each ``(Ns, T)``."""
        Xs = to_tensor(np.asarray(Xs, dtype=np.float64).reshape(len(Xs), -1))
        T, Ns, R = self.num_outputs, Xs.shape[0], self.kernel.num_latent
        Q = self.Z.shape[0]
        K_sU = kron(self.kernel.A(), rbf(Xs, self.Z, self.kernel.lengthscale()))  # (T Ns, R Q)
        mean = tf.matmul(K_sU, self._alpha)[:, 0] + tf.repeat(to_tensor(self.mean), Ns)
        # k_** - k_*U K_UU^-1 k_U* + k_*U P_U^-1 k_U*, using the block structure of K_UU.
        blocks = tf.reshape(tf.transpose(K_sU), [R, Q, T * Ns])
        q_diag = tf.reduce_sum(tf.square(tf.linalg.triangular_solve(self._L_zz, blocks, lower=True)), axis=[0, 1])
        p_diag = tf.reduce_sum(tf.square(tf.linalg.triangular_solve(self._L_P, tf.transpose(K_sU), lower=True)), axis=0)
        var = self.kernel.diag(Xs) - q_diag + p_diag
        to_matrix = lambda a: tf.transpose(tf.reshape(a, [T, Ns])).numpy()
        return to_matrix(mean), np.maximum(to_matrix(var), 0.0)

    def predict_y(self, Xs):
        mean, var = self.predict_f(Xs)
        return mean, var + self.noise_variances().numpy()[None, :]


class RSVMOGP(SVMOGP):
    """Sparse variational R-MOGP (R-SVMOGP).

    Takes the arguments of :class:`SVMOGP` plus ``epsilons`` and ``weight`` as in
    :class:`rmogp.RMOGP`. Weights are computed in :meth:`fit` from the current
    hyperparameters and held fixed while the ELBO is optimised.
    """

    def __init__(self, kernel, Z, noise_var=0.1, mean=0.0, epsilons=0.05, weight="rmogp",
                 shared_noise=False, jitter=1e-6):
        super().__init__(kernel, Z, noise_var=noise_var, mean=mean, shared_noise=shared_noise, jitter=jitter)
        self.epsilons = np.broadcast_to(np.asarray(epsilons, dtype=np.float64), (self.num_outputs,)).copy()
        self.weight = weight

    def _compute_weights(self):
        cov = (self.kernel.B + tf.linalg.diag(self.noise_variances())).numpy()
        self.weights_ = compute_weights(self.Y, self.mean, self.epsilons, cov=cov, weight=self.weight)
        return self.weights_["w01"], self.weights_["grad_log_w2"]

    @property
    def weights(self):
        return self.weights_["w01"]


__all__ = ["SVMOGP", "RSVMOGP"]
