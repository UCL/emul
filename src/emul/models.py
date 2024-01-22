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
    PosteriorPredictiveFunctionFactory,
    PosteriorPredictiveLookaheadVarianceReduction,
    PosteriorPredictiveMeanAndVariance,
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
) -> tuple[NegativeLogMarginalLikelihood, PosteriorPredictiveFunctionFactory]:
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

    def get_posterior_predictive_functions(
        parameters: ParametersDict,
    ) -> tuple[
        PosteriorPredictiveMeanAndVariance,
        PosteriorPredictiveLookaheadVarianceReduction,
    ]:
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        chol_marginal_covariance_matrix = chol_marginal_covariance_function(parameters)

        def posterior_covariance_function(
            input_1: ArrayLike, input_2: ArrayLike
        ) -> Array:
            return covariance_function(
                input_1, input_2, parameters
            ) - jsp.linalg.cho_solve(
                (chol_marginal_covariance_matrix, True),
                vmap_covariance_function(data["inputs"], input_1, parameters),
            ) @ vmap_covariance_function(
                data["inputs"], input_2, parameters
            )

        vmap_posterior_covariance_function = jax.vmap(
            posterior_covariance_function, in_axes=(0, None)
        )
        vmap_vmap_posterior_covariance_function = jax.vmap(
            vmap_posterior_covariance_function,
            in_axes=(None, 0),
        )

        def posterior_mean_and_variance_function(input_: ArrayLike) -> Array:
            covariance_data_input = vmap_covariance_function(
                data["inputs"], input_, parameters
            )
            inv_marginal_covariance_covariance_data_input = jsp.linalg.cho_solve(
                (chol_marginal_covariance_matrix, True),
                covariance_data_input,
            )
            mean = (
                mean_function(input_, parameters)
                + inv_marginal_covariance_covariance_data_input @ zero_mean_outputs
            )
            variance = (
                covariance_function(input_, input_, parameters)
                - inv_marginal_covariance_covariance_data_input @ covariance_data_input
            )
            return mean, variance

        def posterior_lookahead_variance_reduction_function(
            new_input: ArrayLike, pending_inputs: ArrayLike
        ) -> Array:
            marginal_covariance_pending = vmap_vmap_posterior_covariance_function(
                pending_inputs, pending_inputs
            )
            marginal_covariance_pending.at[  # noqa: PD008
                jnp.diag_indices(marginal_covariance_pending.shape[0])
            ].add(
                parameters["observation_noise_std"] ** 2,
            )
            chol_covariance_pending = jnp.linalg.cholesky(marginal_covariance_pending)
            covariance_pending_new = vmap_posterior_covariance_function(
                pending_inputs, new_input
            )
            return covariance_pending_new @ jsp.linalg.cho_solve(
                (chol_covariance_pending, True),
                covariance_pending_new,
            )

        return (
            posterior_mean_and_variance_function,
            posterior_lookahead_variance_reduction_function,
        )

    return neg_log_marginal_likelihood, get_posterior_predictive_functions


def gaussian_process_with_direct_observations_and_reduced_rank(
    data: DataDict,
    mean_function: MeanFunction,
    covariance_function: CovarianceFunction,
    rank: int,
) -> tuple[NegativeLogMarginalLikelihood, PosteriorPredictiveFunctionFactory]:
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

    def get_posterior_predictive_functions(
        parameters: ParametersDict,
    ) -> tuple[
        PosteriorPredictiveMeanAndVariance,
        PosteriorPredictiveLookaheadVarianceReduction,
    ]:
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        eigenvectors, eigenvalues = eigh_marginal_covariance_function(parameters)

        def posterior_covariance_function(
            input_1: ArrayLike, input_2: ArrayLike
        ) -> Array:
            return covariance_function(input_1, input_2, parameters) - (
                (
                    vmap_covariance_function(data["inputs"], input_1, parameters)
                    @ eigenvectors[:, -rank:]
                )
                / eigenvalues[-rank:]
            ) @ vmap_covariance_function(data["inputs"], input_2, parameters)

        vmap_posterior_covariance_function = jax.vmap(
            posterior_covariance_function, in_axes=(0, None)
        )
        vmap_vmap_posterior_covariance_function = jax.vmap(
            vmap_posterior_covariance_function,
            in_axes=(None, 0),
        )

        def posterior_mean_and_variance_function(input_: ArrayLike) -> Array:
            covariance_data_input = vmap_covariance_function(
                data["inputs"], input_, parameters
            )
            inv_marginal_covariance_covariance_data_input = (
                (covariance_data_input @ eigenvectors[:, -rank:]) / eigenvalues[-rank:]
            ) @ eigenvectors[:, -rank:].T
            mean = (
                mean_function(input_, parameters)
                + inv_marginal_covariance_covariance_data_input @ zero_mean_outputs
            )
            variance = (
                covariance_function(input_, input_, parameters)
                - inv_marginal_covariance_covariance_data_input @ covariance_data_input
            )
            return mean, variance

        def posterior_lookahead_variance_reduction_function(
            new_input: ArrayLike, pending_inputs: ArrayLike
        ) -> Array:
            marginal_covariance_pending = vmap_vmap_posterior_covariance_function(
                pending_inputs, pending_inputs
            )
            marginal_covariance_pending.at[  # noqa: PD008
                jnp.diag_indices(marginal_covariance_pending.shape[0])
            ].add(
                parameters["observation_noise_std"] ** 2,
            )
            eigenvectors_pending, eigenvalues_pending = jnp.linalg.eigh(
                marginal_covariance_pending
            )
            covariance_pending_new = vmap_posterior_covariance_function(
                pending_inputs, new_input
            )
            return (
                covariance_pending_new
                @ (
                    (covariance_pending_new @ eigenvectors_pending[:, -rank:])
                    / eigenvalues_pending[-rank:]
                )
                @ eigenvectors_pending[:, -rank:].T
            )

        return (
            posterior_mean_and_variance_function,
            posterior_lookahead_variance_reduction_function,
        )

    return neg_log_marginal_likelihood, get_posterior_predictive_functions


def gaussian_process_with_direct_observations_and_clipping(
    data: DataDict,
    mean_function: MeanFunction,
    covariance_function: CovarianceFunction,
    eigenvalue_threshold: float = 1e-8,
) -> tuple[NegativeLogMarginalLikelihood, PosteriorPredictiveFunctionFactory]:
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

    def get_posterior_predictive_functions(
        parameters: ParametersDict,
    ) -> tuple[
        PosteriorPredictiveMeanAndVariance,
        PosteriorPredictiveLookaheadVarianceReduction,
    ]:
        zero_mean_outputs = data["outputs"] - vmap_mean_function(
            data["inputs"],
            parameters,
        )
        eigenvectors, eigenvalues = eigh_marginal_covariance_function(parameters)
        clipped_eigenvalues = jnp.clip(eigenvalues, eigenvalue_threshold)

        def posterior_covariance_function(
            input_1: ArrayLike, input_2: ArrayLike
        ) -> Array:
            return covariance_function(input_1, input_2, parameters) - (
                (
                    vmap_covariance_function(data["inputs"], input_1, parameters)
                    @ eigenvectors
                )
                / clipped_eigenvalues
            ) @ vmap_covariance_function(data["inputs"], input_2, parameters)

        vmap_posterior_covariance_function = jax.vmap(
            posterior_covariance_function, in_axes=(0, None)
        )
        vmap_vmap_posterior_covariance_function = jax.vmap(
            vmap_posterior_covariance_function,
            in_axes=(None, 0),
        )

        def posterior_mean_and_variance_function(input_: ArrayLike) -> Array:
            covariance_data_input = vmap_covariance_function(
                data["inputs"], input_, parameters
            )
            inv_marginal_covariance_covariance_data_input = (
                (covariance_data_input @ eigenvectors) / clipped_eigenvalues
            ) @ eigenvectors.T
            mean = (
                mean_function(input_, parameters)
                + inv_marginal_covariance_covariance_data_input @ zero_mean_outputs
            )
            variance = (
                covariance_function(input_, input_, parameters)
                - inv_marginal_covariance_covariance_data_input @ covariance_data_input
            )
            return mean, variance

        def posterior_lookahead_variance_reduction_function(
            new_input: ArrayLike, pending_inputs: ArrayLike
        ) -> Array:
            marginal_covariance_pending = vmap_vmap_posterior_covariance_function(
                pending_inputs, pending_inputs
            )
            marginal_covariance_pending.at[  # noqa: PD008
                jnp.diag_indices(marginal_covariance_pending.shape[0])
            ].add(
                parameters["observation_noise_std"] ** 2,
            )
            eigenvectors_pending, eigenvalues_pending = jnp.linalg.eigh(
                marginal_covariance_pending
            )
            clipped_eigenvalues_pending = jnp.clip(
                eigenvalues_pending, eigenvalue_threshold
            )
            covariance_pending_new = vmap_posterior_covariance_function(
                pending_inputs, new_input
            )
            return (
                covariance_pending_new
                @ (
                    (covariance_pending_new @ eigenvectors_pending)
                    / clipped_eigenvalues_pending
                )
                @ eigenvectors_pending.T
            )

        return (
            posterior_mean_and_variance_function,
            posterior_lookahead_variance_reduction_function,
        )

    return neg_log_marginal_likelihood, get_posterior_predictive_functions
