from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import subspace_angles

from substrate.capture import LayerAnalysis


@dataclass
class RotationSummary:
    principal_angles: np.ndarray  # all k angles in radians, sorted descending
    mean_angle: float             # mean of principal angles
    max_angle: float              # maximum principal angle
    grassmann_distance: float     # chordal Grassmann distance = sqrt(sum(sin²(angles)))


def compute_subspace_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces.

    Args:
        basis_a: shape [k, hidden_size] — row-oriented PCA components
        basis_b: shape [k, hidden_size] — row-oriented PCA components

    Returns:
        Array of k principal angles in radians, sorted descending.
    """
    # scipy expects column-oriented [hidden_size, k], so transpose
    # scipy.linalg.subspace_angles returns angles in descending order
    return subspace_angles(basis_a.T, basis_b.T)


def compute_rotation_summary(basis_a: np.ndarray, basis_b: np.ndarray) -> RotationSummary:
    """Compute a RotationSummary describing the subspace rotation between two bases.

    Args:
        basis_a: shape [k, hidden_size] — row-oriented PCA components
        basis_b: shape [k, hidden_size] — row-oriented PCA components

    Returns:
        RotationSummary with principal_angles, mean_angle, max_angle, grassmann_distance
    """
    angles = compute_subspace_angles(basis_a, basis_b)
    mean_angle = float(np.mean(angles))
    max_angle = float(np.max(angles))
    grassmann_distance = float(math.sqrt(np.sum(np.sin(angles) ** 2)))
    return RotationSummary(
        principal_angles=angles,
        mean_angle=mean_angle,
        max_angle=max_angle,
        grassmann_distance=grassmann_distance,
    )


def compare_prompts(
    analysis_a: dict[str, LayerAnalysis],
    analysis_b: dict[str, LayerAnalysis],
    pca_k: int = 10,
) -> dict[str, RotationSummary]:
    """Compare PCA subspaces from two prompt analyses across all shared layers.

    Args:
        analysis_a: dict mapping "layer_{idx}" -> LayerAnalysis for prompt A
        analysis_b: dict mapping "layer_{idx}" -> LayerAnalysis for prompt B
        pca_k: which PCA component count to use (must exist in both analyses)

    Returns:
        dict mapping "layer_{idx}" -> RotationSummary for each shared layer
    """
    shared_layers = set(analysis_a.keys()) & set(analysis_b.keys())
    result: dict[str, RotationSummary] = {}
    for layer_key in sorted(shared_layers):
        basis_a = analysis_a[layer_key].pca_results[pca_k].components  # [k, hidden_size]
        basis_b = analysis_b[layer_key].pca_results[pca_k].components  # [k, hidden_size]
        result[layer_key] = compute_rotation_summary(basis_a, basis_b)
    return result
