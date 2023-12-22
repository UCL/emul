"""Gaussian process emulator models."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax.typing import ArrayLike

from .types import (
    CovarianceFunction,
    DataDict,
    MeanFunction,
    NegativeLogMarginalLikelihood,
    ParametersDict,
    PredictiveMeanAndVariance,
)


def _get_vmapped_mean_and_covariance_functions(
    mean_function: MeanFunction,
    covariance_function: CovarianceFunction,
) -> tuple[
    Callable[[ArrayLike, ParametersDict], Array],
    Callable[[ArrayLike, ArrayLike, ParametersDict], Array],
    Callable[[ArrayLike, ArrayLike, ParametersDict], Array],
]:
    vmap_mean_function = jax.vmap(mean_function, in_axes=(0, None))
    vmap_covariance_function = jax.vmap(covariance_function, in_axes=(0, None, None))
    vmap_vmap_covariance_function = jax.vmap(
        vmap_covariance_function,
        in_axes=(None, 0, None),
    )
    return vmap_mean_function, vmap_covariance_function, vmap_vmap_covariance_function


def gaussian_process_with_isotropic_gaussian_observations(
    data: DataDict,
    mean_function: MeanFunction,
    covariance_function: CovarianceFunction,
) -> tuple[NegativeLogMarginalLikelihood, PredictiveMeanAndVariance]:
    """Gaussian process observed with isotropic Gaussian noise."""
    (
        vmap_mean_function,
        vmap_covariance_function,
        vmap_vmap_covariance_function,
    ) = _get_vmapped_mean_and_covariance_functions(mean_function, covariance_function)

    def marginal_covariance_function(parameters: ParametersDict) -> Array:
        gram_matrix = vmap_vmap_covariance_function(
            data["inputs"],
            data["inputs"],
            parameters,
        )
        return gram_matrix.at[  # noqa: PD008
            jnp.diag_indices(gram_matrix.shape[0])
        ].add(
            parameters["observation_noise_std"] ** 2,
        )

    def chol_marginal_covariance_function(parameters: ParametersDict) -> Array:
        marginal_covariance_matrix = marginal_covariance_function(parameters)
        return jnp.linalg.cholesky(marginal_covariance_matrix)

    def neg_log_marginal_likelihood(parameters: ParametersDict) -> Array:
        chol_marginal_covariance_matrix = chol_marginal_covariance_function(parameters)
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        quad_form_vector = jsp.linalg.solve_triangular(
            chol_marginal_covariance_matrix,
            zero_mean_outputs,
            lower=True,
        )
        return (
            quad_form_vector @ quad_form_vector / 2
            + jnp.log(chol_marginal_covariance_matrix.diagonal()).sum()
        )

    def predictive_mean_and_variance(
        new_input: ArrayLike, parameters: ParametersDict
    ) -> tuple[Array, Array]:
        covar_old_new = vmap_covariance_function(data["inputs"], new_input, parameters)
        covar_new_new = covariance_function(new_input, new_input, parameters)
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        chol_marginal_covariance_matrix = chol_marginal_covariance_function(parameters)
        inv_marginal_covariance_covar_new_new = jsp.linalg.cho_solve(
            (chol_marginal_covariance_matrix, True),
            covar_old_new,
        )
        mean = (
            mean_function(new_input, parameters)
            + inv_marginal_covariance_covar_new_new @ zero_mean_outputs
        )
        variance = (
            covar_new_new
            + parameters["observation_noise_std"] ** 2
            - inv_marginal_covariance_covar_new_new @ covar_old_new
        )
        return mean, variance

    return neg_log_marginal_likelihood, predictive_mean_and_variance


def gaussian_process_with_direct_observations_and_reduced_rank(
    data: DataDict,
    mean_function: MeanFunction,
    covariance_function: CovarianceFunction,
    rank: int,
) -> tuple[NegativeLogMarginalLikelihood, PredictiveMeanAndVariance]:
    """Directly observed Gaussian process with reduced-rank covariance."""
    (
        vmap_mean_function,
        vmap_covariance_function,
        vmap_vmap_covariance_function,
    ) = _get_vmapped_mean_and_covariance_functions(mean_function, covariance_function)

    def marginal_covariance_function(parameters: ParametersDict) -> Array:
        return vmap_vmap_covariance_function(data["inputs"], data["inputs"], parameters)

    def eigh_marginal_covariance_function(
        parameters: ParametersDict,
    ) -> tuple[Array, Array]:
        return jnp.linalg.eigh(marginal_covariance_function(parameters))

    def neg_log_marginal_likelihood(parameters: ParametersDict) -> Array:
        eigenvalues, eigenvectors = eigh_marginal_covariance_function(parameters)
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        projected_zero_mean_outputs = eigenvectors[:, -rank:].T @ zero_mean_outputs
        return (
            projected_zero_mean_outputs
            / eigenvalues[-rank:]
            @ projected_zero_mean_outputs
            / 2
            + jnp.log(eigenvalues[-rank:]).sum() / 2
        )

    def predictive_mean_and_variance(
        new_input: ArrayLike, parameters: ParametersDict
    ) -> tuple[Array, Array]:
        kernel_old_new = vmap_covariance_function(data["inputs"], new_input, parameters)
        kernel_new_new = covariance_function(new_input, new_input, parameters)
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        eigenvalues, eigenvectors = eigh_marginal_covariance_function(parameters)
        projected_covar_old_new = kernel_old_new @ eigenvectors[:, -rank:]
        mean = mean_function(new_input, parameters) + (
            projected_covar_old_new
            / eigenvalues[-rank:]
            @ eigenvectors[:, -rank:].T
            @ zero_mean_outputs
        )
        variance = (
            kernel_new_new
            - (projected_covar_old_new / eigenvalues[-rank:]) @ projected_covar_old_new
        )
        return mean, variance

    return neg_log_marginal_likelihood, predictive_mean_and_variance


def gaussian_process_with_direct_observations_and_clipping(
    data: DataDict,
    mean_function: MeanFunction,
    covariance_function: CovarianceFunction,
    eigenvalue_threshold: float = 1e-8,
) -> tuple[NegativeLogMarginalLikelihood, PredictiveMeanAndVariance]:
    """Directly observed Gaussian process with clipping of covariance eigenvalues."""
    (
        vmap_mean_function,
        vmap_covariance_function,
        vmap_vmap_covariance_function,
    ) = _get_vmapped_mean_and_covariance_functions(mean_function, covariance_function)

    def marginal_covariance_function(parameters: ParametersDict) -> Array:
        return vmap_vmap_covariance_function(data["inputs"], data["inputs"], parameters)

    def eigh_marginal_covariance_function(
        parameters: ParametersDict,
    ) -> tuple[Array, Array]:
        return jnp.linalg.eigh(marginal_covariance_function(parameters))

    def neg_log_marginal_likelihood(parameters: ParametersDict) -> Array:
        eigenvalues, eigenvectors = eigh_marginal_covariance_function(parameters)
        clipped_eigenvalues = jnp.clip(eigenvalues, eigenvalue_threshold)
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        projected_zero_mean_outputs = eigenvectors.T @ zero_mean_outputs
        return (
            projected_zero_mean_outputs
            / clipped_eigenvalues
            @ projected_zero_mean_outputs
            / 2
            + jnp.log(clipped_eigenvalues).sum() / 2
        )

    def predictive_mean_and_variance(
        new_input: ArrayLike, parameters: ParametersDict
    ) -> tuple[Array, Array]:
        kernel_old_new = vmap_covariance_function(data["inputs"], new_input, parameters)
        kernel_new_new = covariance_function(new_input, new_input, parameters)
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        eigenvalues, eigenvectors = eigh_marginal_covariance_function(parameters)
        clipped_eigenvalues = jnp.clip(eigenvalues, eigenvalue_threshold)
        projected_covar_old_new = kernel_old_new @ eigenvectors
        mean = mean_function(new_input, parameters) + (
            projected_covar_old_new
            / clipped_eigenvalues
            @ eigenvectors.T
            @ zero_mean_outputs
        )
        variance = (
            kernel_new_new
            - (projected_covar_old_new / clipped_eigenvalues) @ projected_covar_old_new
        )
        return mean, variance

    return neg_log_marginal_likelihood, predictive_mean_and_variance
