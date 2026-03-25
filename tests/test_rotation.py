from __future__ import annotations

import math

import numpy as np
import pytest

from substrate.capture import LayerAnalysis, PCAResult
from substrate.rotation import (
    RotationSummary,
    compare_prompts,
    compute_rotation_summary,
    compute_subspace_angles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_orthonormal_basis(k: int, hidden_size: int, seed: int = 0) -> np.ndarray:
    """Return a random orthonormal basis of shape [k, hidden_size]."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((hidden_size, k))
    q, _ = np.linalg.qr(raw)
    return q[:, :k].T.copy()  # [k, hidden_size]


def make_fake_layer_analysis(components: np.ndarray, k: int) -> LayerAnalysis:
    """Build a minimal LayerAnalysis wrapping given components at key k."""
    hidden_size = components.shape[1]
    pca_result = PCAResult(
        components=components,
        explained_variance=np.ones(k, dtype=np.float64),
        explained_variance_ratio=np.ones(k, dtype=np.float64) / k,
        mean=np.zeros(hidden_size, dtype=np.float64),
    )
    return LayerAnalysis(
        raw_activations=np.zeros((10, hidden_size), dtype=np.float64),
        pca_results={k: pca_result},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_identity_angles() -> None:
    """Same basis compared against itself should yield all angles ≈ 0."""
    basis = make_orthonormal_basis(k=5, hidden_size=64)
    angles = compute_subspace_angles(basis, basis)

    assert angles.shape == (5,), f"Expected 5 angles, got shape {angles.shape}"
    np.testing.assert_allclose(
        angles, 0.0, atol=1e-10,
        err_msg="Identical subspaces should have zero principal angles",
    )


def test_orthogonal_subspaces() -> None:
    """Two orthogonal subspaces should have all principal angles ≈ π/2."""
    hidden_size = 20
    # Build a full orthonormal basis of R^hidden_size
    full_q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((hidden_size, hidden_size)))
    # Split into two non-overlapping orthogonal subspaces of size k=5
    k = 5
    basis_a = full_q[:, :k].T.copy()      # [k, hidden_size]
    basis_b = full_q[:, k:2*k].T.copy()   # [k, hidden_size]

    angles = compute_subspace_angles(basis_a, basis_b)

    assert angles.shape == (k,), f"Expected {k} angles, got {angles.shape}"
    np.testing.assert_allclose(
        angles, math.pi / 2, atol=1e-10,
        err_msg="Orthogonal subspaces should have principal angles ≈ π/2",
    )


def test_known_rotation() -> None:
    """Rotate a 2D basis by a known angle θ out-of-plane and verify the recovered angle matches."""
    theta = math.pi / 6  # 30 degrees
    hidden_size = 10

    # basis_a spans e_0 and e_1
    basis_a = np.zeros((2, hidden_size), dtype=np.float64)
    basis_a[0, 0] = 1.0
    basis_a[1, 1] = 1.0

    # basis_b spans e_0 and (cos(theta)*e_1 + sin(theta)*e_2)
    # The e_0 direction is shared; the second vector is rotated by theta out of span(e_1)
    # into e_2, so the two subspaces share one direction and differ by theta on the other.
    basis_b = np.zeros((2, hidden_size), dtype=np.float64)
    basis_b[0, 0] = 1.0
    basis_b[1, 1] = math.cos(theta)
    basis_b[1, 2] = math.sin(theta)

    angles = compute_subspace_angles(basis_a, basis_b)

    # Verify descending sort order (spec requirement)
    assert list(angles) == sorted(angles, reverse=True), "angles must be sorted descending"

    # Descending: [theta, 0]
    np.testing.assert_allclose(angles[0], theta, atol=1e-10,
                               err_msg=f"First (largest) angle should be theta={theta:.4f} rad")
    np.testing.assert_allclose(angles[1], 0.0, atol=1e-10,
                               err_msg="Second (smallest) angle should be 0 (shared direction)")


def test_grassmann_distance_manual() -> None:
    """Manually compute Grassmann distance from known angles and verify it matches RotationSummary."""
    hidden_size = 20
    rng = np.random.default_rng(7)

    # Build two random orthonormal 4-d subspaces
    q, _ = np.linalg.qr(rng.standard_normal((hidden_size, hidden_size)))
    k = 4
    basis_a = q[:, :k].T.copy()
    basis_b = q[:, k:2*k].T.copy()

    summary = compute_rotation_summary(basis_a, basis_b)

    # Manual Grassmann (chordal) distance
    manual_distance = math.sqrt(float(np.sum(np.sin(summary.principal_angles) ** 2)))

    assert isinstance(summary, RotationSummary)
    np.testing.assert_allclose(
        summary.grassmann_distance, manual_distance, atol=1e-12,
        err_msg="grassmann_distance does not match manual computation",
    )
    np.testing.assert_allclose(
        summary.mean_angle, float(np.mean(summary.principal_angles)), atol=1e-12,
        err_msg="mean_angle inconsistent with principal_angles",
    )
    np.testing.assert_allclose(
        summary.max_angle, float(np.max(summary.principal_angles)), atol=1e-12,
        err_msg="max_angle inconsistent with principal_angles",
    )


def test_compare_prompts() -> None:
    """compare_prompts returns a RotationSummary for each shared layer."""
    hidden_size = 32
    k = 5

    rng = np.random.default_rng(42)

    # Build two sets of orthonormal bases per layer
    def random_basis() -> np.ndarray:
        q, _ = np.linalg.qr(rng.standard_normal((hidden_size, hidden_size)))
        return q[:, :k].T.copy()  # [k, hidden_size]

    layer_keys = ["layer_0", "layer_1", "layer_2"]

    analysis_a: dict[str, LayerAnalysis] = {
        key: make_fake_layer_analysis(random_basis(), k) for key in layer_keys
    }
    analysis_b: dict[str, LayerAnalysis] = {
        key: make_fake_layer_analysis(random_basis(), k) for key in layer_keys
    }
    # Add an extra layer only in analysis_a — should be ignored
    analysis_a["layer_99"] = make_fake_layer_analysis(random_basis(), k)

    result = compare_prompts(analysis_a, analysis_b, pca_k=k)

    # Only shared layers should appear
    assert set(result.keys()) == set(layer_keys), (
        f"Expected keys {layer_keys}, got {sorted(result.keys())}"
    )

    for key in layer_keys:
        summary = result[key]
        assert isinstance(summary, RotationSummary), (
            f"{key}: expected RotationSummary, got {type(summary)}"
        )
        assert summary.principal_angles.shape == (k,), (
            f"{key}: principal_angles shape {summary.principal_angles.shape}, expected ({k},)"
        )
        assert 0.0 <= summary.mean_angle <= math.pi / 2 + 1e-10, (
            f"{key}: mean_angle {summary.mean_angle} out of expected range"
        )
        assert summary.grassmann_distance >= 0.0, (
            f"{key}: grassmann_distance {summary.grassmann_distance} is negative"
        )
