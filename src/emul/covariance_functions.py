"""Positive semi-definite covariance functions."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .types import ParametersDict


def exponentiated_quadratic(
    input_1: ArrayLike,
    input_2: ArrayLike,
    parameters: ParametersDict,
) -> Array:
    """
    Exponentiated quadratic covariance function.

    Also known as the squared exponential or radial basis function covariance function.
    """
    return parameters["output_scale"] ** 2 * jnp.exp(
        -((input_1 - input_2) ** 2 / (2 * parameters["input_scales"] ** 2)).sum()
    )


def matern_five_halves(
    input_1: ArrayLike,
    input_2: ArrayLike,
    parameters: ParametersDict,
) -> Array:
    """Matern covariance function with shape parameter 5/2."""
    scaled_input_1 = input_1 / parameters["input_scales"]
    scaled_input_2 = input_2 / parameters["input_scales"]
    scaled_distance = jnp.sqrt(
        jnp.maximum(jnp.sum((scaled_input_1 - scaled_input_2) ** 2), 1e-36)
    )
    return (
        parameters["output_scale"] ** 2
        * (1.0 + (5.0**0.5) * scaled_distance + (5.0 / 3.0) * scaled_distance**2)
        * jnp.exp(-(5.0**0.5) * scaled_distance)
    )


def rational_quadratic(
    input_1: ArrayLike,
    input_2: ArrayLike,
    parameters: ParametersDict,
) -> Array:
    """Rational quadratic covariance function."""
    return parameters["output_scale"] ** 2 * (
        1.0
        + (
            (input_1 - input_2) ** 2
            / (2 * parameters["scale_mixture_rate"] * parameters["input_scales"] ** 2)
        ).sum()
    ) ** (-parameters["scale_mixture_rate"])
