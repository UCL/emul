"""Type aliases."""

from collections.abc import Callable
from typing import TypedDict

from jax import Array
from jax.typing import ArrayLike

ParametersDict = dict[str, ArrayLike]
"""Type alias for dictionaries of parameters."""

MeanFunction = Callable[[ArrayLike, ParametersDict], Array]
"""Type alias for Gaussian process mean functions."""

CovarianceFunction = Callable[[ArrayLike, ArrayLike, ParametersDict], Array]
"""Type alias for Gaussian process covariance functions."""

NegativeLogMarginalLikelihood = Callable[[ParametersDict], Array]
"""Type alias for negative marginal likelihood function for Gaussian process model."""

PredictiveMeanAndVariance = Callable[[ArrayLike, ParametersDict], tuple[Array, Array]]
"""Type alias for predictive mean and variance function for Gaussian process model."""


class DataDict(TypedDict):
    """Type alias for dictionary of function data to condition Gaussian process on."""

    inputs: ArrayLike
    outputs: ArrayLike
