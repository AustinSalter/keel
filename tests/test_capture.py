from __future__ import annotations

import numpy as np

from substrate.capture import LayerAnalysis, PCAResult, compute_pca_basis, load_analysis, save_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_random_activations(seq_len: int = 100, hidden_size: int = 2048, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((seq_len, hidden_size)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_pca_basis_shapes() -> None:
    """PCA on [100, 2048] at k=10 produces correctly shaped outputs."""
    activations = make_random_activations(seq_len=100, hidden_size=2048)
    result = compute_pca_basis(activations, n_components=10)

    assert isinstance(result, PCAResult)
    assert result.components.shape == (10, 2048), f"components shape: {result.components.shape}"
    assert result.explained_variance.shape == (10,), f"explained_variance shape: {result.explained_variance.shape}"
    assert result.explained_variance_ratio.shape == (10,), f"explained_variance_ratio shape: {result.explained_variance_ratio.shape}"
    assert result.mean.shape == (2048,), f"mean shape: {result.mean.shape}"


def test_pca_variance_ratio_sums_to_one_or_less() -> None:
    """explained_variance_ratio must sum to <= 1.0 (it is a partial sum for k < rank)."""
    activations = make_random_activations(seq_len=100, hidden_size=2048)
    result = compute_pca_basis(activations, n_components=10)

    total = result.explained_variance_ratio.sum()
    assert total <= 1.0 + 1e-6, f"explained_variance_ratio sums to {total}, expected <= 1.0"
    assert total > 0.0, "explained_variance_ratio sum should be positive"


def test_pca_components_orthonormal() -> None:
    """PCA components form an orthonormal set: components @ components.T ≈ identity."""
    activations = make_random_activations(seq_len=100, hidden_size=2048)
    result = compute_pca_basis(activations, n_components=10)

    gram = result.components @ result.components.T  # [10, 10]
    identity = np.eye(10, dtype=gram.dtype)
    np.testing.assert_allclose(gram, identity, atol=1e-5,
                               err_msg="components are not orthonormal")


def test_low_rank_recovery() -> None:
    """PCA recovers a rank-3 signal embedded in 2048 dims: first 3 PCs capture >99% of variance."""
    rng = np.random.default_rng(0)
    n_samples = 200
    hidden_size = 2048

    # Build a rank-3 matrix: project 3 signal directions through 2048 dims
    signal_directions = rng.standard_normal((3, hidden_size)).astype(np.float32)
    # Orthonormalise via QR so directions are clean
    signal_directions, _ = np.linalg.qr(signal_directions.T)
    signal_directions = signal_directions.T  # [3, hidden_size]

    coefficients = rng.standard_normal((n_samples, 3)).astype(np.float32)
    # Weight the three directions so variance is clearly ordered
    coefficients *= np.array([10.0, 5.0, 2.0], dtype=np.float32)

    activations = coefficients @ signal_directions  # [n_samples, hidden_size]
    # Add tiny noise so the matrix isn't literally rank 3 (avoids sklearn warnings)
    activations += rng.standard_normal(activations.shape).astype(np.float32) * 1e-3

    result = compute_pca_basis(activations, n_components=5)

    variance_in_top3 = result.explained_variance_ratio[:3].sum()
    assert variance_in_top3 > 0.99, (
        f"Expected top-3 PCs to capture >99% variance; got {variance_in_top3:.4f}"
    )


def test_hdf5_roundtrip(tmp_path) -> None:
    """Save a LayerAnalysis to HDF5 and load it back; all arrays must match."""
    rng = np.random.default_rng(7)

    # Build known PCAResult values
    components = rng.standard_normal((5, 64)).astype(np.float32)
    explained_variance = np.array([3.0, 2.0, 1.5, 1.0, 0.5], dtype=np.float32)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    mean = rng.standard_normal(64).astype(np.float32)

    pca_result = PCAResult(
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean,
    )

    raw_activations = rng.standard_normal((20, 64)).astype(np.float32)
    layer_analysis = LayerAnalysis(
        raw_activations=raw_activations,
        pca_results={5: pca_result},
    )

    analysis = {"layer_7": layer_analysis}
    metadata = {"model_id": "test-model", "prompt": "hello world", "version": 1}

    hdf5_path = tmp_path / "test_analysis.h5"
    save_analysis(hdf5_path, analysis, metadata)

    loaded_analysis, loaded_metadata = load_analysis(hdf5_path)

    # Metadata
    assert loaded_metadata == metadata, f"metadata mismatch: {loaded_metadata}"

    # Structure
    assert "layer_7" in loaded_analysis, "layer_7 key missing after load"
    loaded_layer = loaded_analysis["layer_7"]

    # Raw activations
    np.testing.assert_array_equal(
        loaded_layer.raw_activations, raw_activations,
        err_msg="raw_activations mismatch after roundtrip"
    )

    # PCA results
    assert 5 in loaded_layer.pca_results, "pca k=5 missing after load"
    loaded_pca = loaded_layer.pca_results[5]

    np.testing.assert_array_almost_equal(loaded_pca.components, components, decimal=6)
    np.testing.assert_array_almost_equal(loaded_pca.explained_variance, explained_variance, decimal=6)
    np.testing.assert_array_almost_equal(
        loaded_pca.explained_variance_ratio, explained_variance_ratio, decimal=6
    )
    np.testing.assert_array_almost_equal(loaded_pca.mean, mean, decimal=6)
