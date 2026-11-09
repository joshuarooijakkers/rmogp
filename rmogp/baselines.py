"""Non-conjugate robust MOGP baselines built with GPflow (needs ``pip install gpflow``).

* :class:`TMOGP`: ICM-MOGP with a Student-t likelihood (Jylänki et al., 2011).
* :class:`LMOGP`: ICM-MOGP with a Laplace likelihood (Kuss, 2006).

Both use a variational Gaussian approximation (``gpflow.models.VGP``), or a sparse
variational GP (``gpflow.models.SVGP``) when inducing inputs ``Z`` are given. GPflow
only supports one likelihood scale shared across outputs, so the scale is shared.
"""


import gpflow
import numpy as np
import tensorflow as tf


class LaplaceLikelihood(gpflow.likelihoods.ScalarLikelihood):
    """Laplace likelihood ``p(y | f) = exp(-|y - f| / b) / (2 b)``."""

    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = gpflow.Parameter(scale, transform=gpflow.utilities.positive())

    def _scalar_log_prob(self, X, F, Y):
        return -tf.math.log(2.0 * self.scale) - tf.abs(Y - F) / self.scale

    def _conditional_mean(self, X, F):
        return F

    def _conditional_variance(self, X, F):
        return tf.ones_like(F) * 2.0 * tf.square(self.scale)

    def _predict_mean_and_var(self, X, Fmu, Fvar):
        return Fmu, Fvar + 2.0 * tf.square(self.scale)


def stack_outputs(X, Y=None, num_outputs=None):
    """Convert ``(N, d)`` inputs and ``(N, T)`` outputs to GPflow's coregionalisation format.

    Returns ``X_aug`` of shape ``(T N, d + 1)`` with the output index as last column and,
    if ``Y`` is given, ``y`` of shape ``(T N, 1)``. Missing outputs (NaN) are dropped.
    """
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    T = Y.shape[1] if Y is not None else num_outputs
    X_aug = np.vstack([np.hstack([X, np.full((len(X), 1), t)]) for t in range(T)])
    if Y is None:
        return X_aug
    y = np.asarray(Y, dtype=np.float64).T.reshape(-1, 1)
    keep = ~np.isnan(y[:, 0])
    return X_aug[keep], y[keep]


class _GPflowMOGP:
    def __init__(self, num_outputs, A=None, rank=None, lengthscale=1.0, kernel_variance=1.0,
                 train_kernel_variance=False, mean=0.0, Z=None, train_Z=False):
        self.num_outputs = T = num_outputs
        rank = rank or (T if A is None else np.shape(A)[1])
        self.A_init = np.asarray(A, dtype=np.float64) if A is not None else 0.5 * np.ones((T, rank))
        self.lengthscale = lengthscale
        self.kernel_variance = kernel_variance
        self.train_kernel_variance = train_kernel_variance
        self.mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), (T,)).copy()
        self.Z = None if Z is None else np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)
        self.train_Z = train_Z

    def _likelihood(self):
        raise NotImplementedError

    def _kernel(self, input_dim):
        base = gpflow.kernels.SquaredExponential(
            variance=self.kernel_variance, lengthscales=self.lengthscale, active_dims=list(range(input_dim)))
        coreg = gpflow.kernels.Coregion(output_dim=self.num_outputs, rank=self.A_init.shape[1], active_dims=[input_dim])
        coreg.W.assign(self.A_init)
        coreg.kappa.assign(np.full(self.num_outputs, 1e-6))
        gpflow.utilities.set_trainable(coreg.kappa, False)
        gpflow.utilities.set_trainable(base.variance, self.train_kernel_variance)
        return base * coreg

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        Y = np.asarray(Y, dtype=np.float64)
        X_aug, y = stack_outputs(X, Y - self.mean[None, :])
        kernel, likelihood = self._kernel(X.shape[1]), self._likelihood()
        if self.Z is None:
            self.model = gpflow.models.VGP((X_aug, y), kernel=kernel, likelihood=likelihood)
        else:
            Z_aug = stack_outputs(self.Z, num_outputs=self.num_outputs)
            self.model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood,
                                            inducing_variable=gpflow.inducing_variables.InducingPoints(Z_aug),
                                            num_data=len(y), whiten=True)
            gpflow.utilities.set_trainable(self.model.inducing_variable.Z, self.train_Z)
            self._data = (tf.constant(X_aug), tf.constant(y))
        return self

    def _loss(self):
        if self.Z is None:
            return self.model.training_loss
        return self.model.training_loss_closure(self._data, compile=False)

    def optimize(self, method="lbfgs", maxiter=None, tol=None, steps=2000, learning_rate=1e-2):
        """Optimise the variational objective.

        ``method="lbfgs"`` uses ``gpflow.optimizers.Scipy`` and returns its result
        (``res.nfev`` is the number of function evaluations). ``method="adam"`` runs
        ``steps`` full-batch Adam steps and returns the loss history.
        """
        loss = self._loss()
        variables = self.model.trainable_variables
        if method == "lbfgs":
            options = {} if maxiter is None else {"maxiter": maxiter}
            return gpflow.optimizers.Scipy().minimize(loss, variables, tol=tol, options=options)
        if method == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            @tf.function
            def step():
                with tf.GradientTape() as tape:
                    value = loss()
                optimizer.apply_gradients(zip(tape.gradient(value, variables), variables))
                return value

            return np.array([float(step().numpy()) for _ in range(steps)])
        raise ValueError(f'Unknown method {method!r}; use "lbfgs" or "adam".')

    def _predict(self, Xs, fn):
        X_aug = stack_outputs(Xs, num_outputs=self.num_outputs)
        mean, var = fn(X_aug)
        to_matrix = lambda a: a.numpy().reshape(self.num_outputs, -1).T
        return to_matrix(mean) + self.mean[None, :], to_matrix(var)

    def predict_f(self, Xs):
        """Latent mean and variance, each ``(Ns, T)``."""
        return self._predict(Xs, self.model.predict_f)

    def predict_y(self, Xs):
        """Predictive mean and variance of new observations, each ``(Ns, T)``."""
        return self._predict(Xs, self.model.predict_y)


class TMOGP(_GPflowMOGP):
    """ICM-MOGP with a Student-t likelihood (``df`` degrees of freedom, shared ``scale``).

    Other arguments: ``num_outputs``, ``A`` (initial ``(T, R)`` coregionalisation
    matrix) or ``rank``, ``lengthscale``, ``kernel_variance``, ``mean`` and optional
    inducing inputs ``Z`` for the sparse version.
    """

    def __init__(self, num_outputs, df=3.0, scale=1.0, **kwargs):
        super().__init__(num_outputs, **kwargs)
        self.df, self.scale = df, scale

    def _likelihood(self):
        return gpflow.likelihoods.StudentT(scale=self.scale, df=self.df)


class LMOGP(_GPflowMOGP):
    """ICM-MOGP with a Laplace likelihood (shared ``scale``). See :class:`TMOGP`."""

    def __init__(self, num_outputs, scale=0.1, **kwargs):
        super().__init__(num_outputs, **kwargs)
        self.scale = scale

    def _likelihood(self):
        return LaplaceLikelihood(scale=self.scale)


__all__ = ["TMOGP", "LMOGP", "LaplaceLikelihood", "stack_outputs"]
