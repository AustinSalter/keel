from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA

from substrate.hooks import ActivationCollector


@dataclass
class PCAResult:
    components: np.ndarray              # shape [n_components, hidden_size]
    explained_variance: np.ndarray      # shape [n_components]
    explained_variance_ratio: np.ndarray  # shape [n_components]
    mean: np.ndarray                    # shape [hidden_size]


@dataclass
class LayerAnalysis:
    raw_activations: np.ndarray          # [seq_len, hidden_size]
    pca_results: dict[int, PCAResult]    # k -> PCAResult for each k in pca_components


def capture_activations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_indices: list[int],
    max_length: int = 4096,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt, run forward pass with ActivationCollector hooks, return activations dict."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    collector = ActivationCollector()
    collector.register(model, layer_indices=layer_indices)

    with torch.no_grad():
        model(**inputs)

    activations = collector.collect()
    collector.remove_all()

    return activations


def compute_pca_basis(activations: np.ndarray, n_components: int = 20) -> PCAResult:
    """Compute PCA basis from activations matrix.

    Args:
        activations: shape [seq_len, hidden_size] — seq_len is samples, hidden_size is features
        n_components: number of PCA components to compute

    Returns:
        PCAResult with components, explained_variance, explained_variance_ratio, mean
    """
    pca = PCA(n_components=n_components)
    pca.fit(activations)

    return PCAResult(
        components=pca.components_,                          # [n_components, hidden_size]
        explained_variance=pca.explained_variance_,          # [n_components]
        explained_variance_ratio=pca.explained_variance_ratio_,  # [n_components]
        mean=pca.mean_,                                      # [hidden_size]
    )


def capture_and_analyze(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_indices: list[int],
    pca_components: list[int] | None = None,
) -> dict[str, LayerAnalysis]:
    """Orchestrate activation capture and PCA analysis across layers.

    Args:
        model: HuggingFace-style causal LM
        tokenizer: corresponding tokenizer
        prompt: input text
        layer_indices: which decoder layer indices to capture
        pca_components: list of k values for PCA; defaults to [5, 10, 20]

    Returns:
        dict mapping "layer_{idx}" -> LayerAnalysis
    """
    if pca_components is None:
        pca_components = [5, 10, 20]

    raw = capture_activations(model, tokenizer, prompt, layer_indices)

    result: dict[str, LayerAnalysis] = {}
    for idx in layer_indices:
        key = f"layer_{idx}"
        tensor = raw[key]
        activations_np = tensor.numpy()  # [seq_len, hidden_size]

        pca_results: dict[int, PCAResult] = {}
        max_k = min(activations_np.shape[0], activations_np.shape[1])
        for k in pca_components:
            if k > max_k:
                continue  # Skip k values that exceed the rank of the activation matrix
            pca_results[k] = compute_pca_basis(activations_np, n_components=k)

        result[key] = LayerAnalysis(
            raw_activations=activations_np,
            pca_results=pca_results,
        )

    return result


def save_analysis(path: Path, analysis: dict[str, LayerAnalysis], metadata: dict) -> None:
    """Save LayerAnalysis dict to HDF5.

    HDF5 structure:
        /metadata              — JSON string stored as root attribute
        /layer_7/activations   — [seq_len, hidden_size]
        /layer_7/pca_5/components            — [5, hidden_size]
        /layer_7/pca_5/explained_variance    — [5]
        /layer_7/pca_5/explained_variance_ratio — [5]
        /layer_7/pca_5/mean                  — [hidden_size]
        /layer_7/pca_10/...
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = json.dumps(metadata)

        for layer_key, layer_analysis in analysis.items():
            layer_grp = f.create_group(layer_key)
            layer_grp.create_dataset("activations", data=layer_analysis.raw_activations)

            for k, pca_result in layer_analysis.pca_results.items():
                pca_grp = layer_grp.create_group(f"pca_{k}")
                pca_grp.create_dataset("components", data=pca_result.components)
                pca_grp.create_dataset("explained_variance", data=pca_result.explained_variance)
                pca_grp.create_dataset(
                    "explained_variance_ratio", data=pca_result.explained_variance_ratio
                )
                pca_grp.create_dataset("mean", data=pca_result.mean)


def load_analysis(path: Path) -> tuple[dict[str, LayerAnalysis], dict]:
    """Load LayerAnalysis dict from HDF5.

    Returns:
        (analysis_dict, metadata_dict)
    """
    path = Path(path)
    analysis: dict[str, LayerAnalysis] = {}

    with h5py.File(path, "r") as f:
        metadata = json.loads(f.attrs["metadata"])

        for layer_key in f.keys():
            layer_grp = f[layer_key]
            if not isinstance(layer_grp, h5py.Group):
                continue
            raw_activations = layer_grp["activations"][:]

            pca_results: dict[int, PCAResult] = {}
            for grp_name in layer_grp.keys():
                if not grp_name.startswith("pca_"):
                    continue
                k = int(grp_name[len("pca_"):])
                pca_grp = layer_grp[grp_name]
                pca_results[k] = PCAResult(
                    components=pca_grp["components"][:],
                    explained_variance=pca_grp["explained_variance"][:],
                    explained_variance_ratio=pca_grp["explained_variance_ratio"][:],
                    mean=pca_grp["mean"][:],
                )

            analysis[layer_key] = LayerAnalysis(
                raw_activations=raw_activations,
                pca_results=pca_results,
            )

    return analysis, metadata
