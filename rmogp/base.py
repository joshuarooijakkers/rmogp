"""Small parameter-handling utilities shared by kernels and models."""

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def to_tensor(x):
    return tf.convert_to_tensor(np.asarray(x, dtype=np.float64), dtype=DTYPE)


class Parameter:
    """A model parameter stored as an unconstrained ``tf.Variable``.

    Positive parameters are stored on the log scale. Call the parameter to get
    its constrained value as a tensor, or use :meth:`numpy` for a NumPy copy.
    Set ``trainable = False`` to keep it fixed during optimisation.
    """

    def __init__(self, value, positive=False, trainable=True, name=None):
        value = np.asarray(value, dtype=np.float64)
        if positive and np.any(value <= 0):
            raise ValueError(f"Parameter {name!r} must be positive, got {value}.")
        self.positive = positive
        self.trainable = trainable
        self.name = name
        self.unconstrained = tf.Variable(np.log(value) if positive else value, dtype=DTYPE, name=name)

    def __call__(self):
        return tf.exp(self.unconstrained) if self.positive else tf.identity(self.unconstrained)

    def numpy(self):
        return self().numpy()

    def assign(self, value):
        value = np.asarray(value, dtype=np.float64)
        self.unconstrained.assign(np.log(value) if self.positive else value)

    def __repr__(self):
        return f"Parameter({self.name}={self.numpy()}, trainable={self.trainable})"


class Module:
    """Base class that collects :class:`Parameter` attributes, also from nested modules."""

    def parameters(self, prefix=""):
        params = {}
        for key, value in vars(self).items():
            if isinstance(value, Parameter):
                params[prefix + key] = value
            elif isinstance(value, Module):
                params.update(value.parameters(prefix=f"{prefix}{key}."))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        params.update(item.parameters(prefix=f"{prefix}{key}.{i}."))
        return params

    def trainable_variables(self):
        return [p.unconstrained for p in self.parameters().values() if p.trainable]

    def print_parameters(self):
        for name, p in self.parameters().items():
            flag = "" if p.trainable else "  (fixed)"
            print(f"{name}: {np.array2string(p.numpy(), precision=4)}{flag}")
