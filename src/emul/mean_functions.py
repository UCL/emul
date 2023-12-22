"""Mean functions for Gaussian processes."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .types import ParametersDict


def zero(_input: ArrayLike, _parameters: ParametersDict) -> Array:
    """Constant zero mean function."""
    return jnp.zeros_like(_input[0])


def centred_and_scaled_quadratic(
    input_: ArrayLike, parameters: ParametersDict
) -> Array:
    """Centred and scaled quadratic mean function with constant offset."""
    return (
        parameters["mean_offset"]
        + (
            parameters["mean_coefficients"] * (input_ - parameters["mean_centres"]) ** 2
        ).sum()
    )


def centred_and_scaled_absolute_polynomial(
    input_: ArrayLike, parameters: ParametersDict
) -> Array:
    """Centred and scaled absolute polynomial mean function with constant offset."""
    return (
        parameters["mean_offset"]
        + (
            parameters["mean_coefficients"]
            * (abs(input_ - parameters["mean_centres"]) ** parameters["mean_exponents"])
        ).sum()
    )


def absolute_polynomial(input_: ArrayLike, parameters: ParametersDict) -> Array:
    """Centred and scaled absolute polynomial mean function with constant offset."""
    return (
        parameters["mean_offset"]
        + (
            abs((input_ - parameters["mean_centres"]) / parameters["mean_scales"])
            ** parameters["mean_exponents"]
        ).sum()
    )
