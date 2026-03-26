"""Modal app for running substrate geometry inference on remote A100 GPUs.

Provides three Modal functions:
  - verify_pipeline: Sprint 0 acceptance test (capture + PCA on a simple prompt)
  - profile_memory: VRAM usage measurements at various sequence lengths
  - sanity_check: Cross-prompt subspace rotation comparison (code vs philosophy)
"""

from __future__ import annotations

import modal

app = modal.App("keel-substrate")

# Docker image with all dependencies + local substrate package
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "scikit-learn",
        "scipy",
        "numpy",
        "h5py",
    )
    .add_local_python_source("substrate")
)

# Persistent volume for caching HuggingFace model weights (~52 GB for large models)
model_cache = modal.Volume.from_name("keel-model-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helper (runs inside Modal container, not a Modal function itself)
# ---------------------------------------------------------------------------

def load_model(model_id: str, trust_remote_code: bool = True):
    """Load a HuggingFace causal-LM and its tokenizer in bfloat16 with auto device mapping."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer


def get_layer_indices(model) -> list[int]:
    """Auto-detect appropriate layer indices from model architecture.

    Picks 4 evenly-spaced layers (roughly at 25%, 50%, 75%, 100% depth).
    """
    num_layers = len(model.model.layers)
    # 4 evenly spaced layers, 0-indexed
    indices = [
        num_layers // 4 - 1,
        num_layers // 2 - 1,
        3 * num_layers // 4 - 1,
        num_layers - 1,
    ]
    return indices


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
)
def verify_pipeline(model_id: str) -> dict:
    """Sprint 0 acceptance test: capture activations and run PCA on a simple prompt.

    Returns a JSON-serializable dict with model_id, layer keys, and
    explained_variance_ratio at k=20 for each layer.
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    from substrate.capture import capture_and_analyze
    from substrate.config import PCA_COMPONENTS

    model, tokenizer = load_model(model_id)
    layer_indices = get_layer_indices(model)

    prompt = "The quick brown fox jumps over the lazy dog."
    analysis = capture_and_analyze(
        model, tokenizer, prompt, layer_indices=layer_indices, pca_components=PCA_COMPONENTS
    )

    # Build JSON-serializable result
    result: dict = {"model_id": model_id, "layers": {}}
    for layer_key, layer_analysis in analysis.items():
        layer_info: dict = {}
        for k, pca_result in layer_analysis.pca_results.items():
            layer_info[f"explained_variance_ratio_k{k}"] = (
                pca_result.explained_variance_ratio.tolist()
            )
        result["layers"][layer_key] = layer_info

    return result


@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
)
def profile_memory(model_id: str) -> dict:
    """Measure GPU VRAM at various stages: model load, forward pass, hooks, sequence lengths.

    Returns a JSON-serializable dict with all memory measurements in bytes.
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    import torch

    from substrate.hooks import ActivationCollector

    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model(model_id)
    layer_indices = get_layer_indices(model)
    baseline_bytes = torch.cuda.memory_allocated()  # Steady-state footprint after load

    # Forward pass without hooks
    torch.cuda.reset_peak_memory_stats()
    inputs = tokenizer(
        "Hello, world!",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)
    forward_no_hooks_bytes = torch.cuda.max_memory_allocated()

    # Forward pass with hooks at 4 layers
    torch.cuda.reset_peak_memory_stats()
    collector = ActivationCollector()
    collector.register(model, layer_indices=layer_indices)
    with torch.no_grad():
        model(**inputs)
    _ = collector.collect()
    collector.remove_all()
    forward_with_hooks_bytes = torch.cuda.max_memory_allocated()

    # Memory at different sequence lengths
    seq_length_measurements: dict[str, int] = {}
    for seq_len in [512, 1024, 2048, 4096]:
        torch.cuda.reset_peak_memory_stats()
        # Create a dummy input of approximately the target length
        dummy_text = "word " * (seq_len // 2)  # rough approximation
        seq_inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        seq_inputs = {k: v.to(device) for k, v in seq_inputs.items()}
        collector = ActivationCollector()
        collector.register(model, layer_indices=layer_indices)
        with torch.no_grad():
            model(**seq_inputs)
        collector.collect()
        collector.remove_all()
        peak = torch.cuda.max_memory_allocated()
        seq_length_measurements[f"seq_{seq_len}"] = peak

    return {
        "model_id": model_id,
        "baseline_after_load_bytes": baseline_bytes,
        "forward_no_hooks_bytes": forward_no_hooks_bytes,
        "forward_with_hooks_bytes": forward_with_hooks_bytes,
        "sequence_length_measurements": seq_length_measurements,
    }


@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
)
def sanity_check(model_id: str) -> dict:
    """Cross-prompt subspace rotation comparison: code vs philosophy, plus self-comparison control.

    Returns a JSON-serializable dict with rotation summaries for
    code_vs_philosophy and code_vs_code (self-comparison) per layer.
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    from substrate.capture import capture_and_analyze
    from substrate.config import PCA_COMPONENTS
    from substrate.rotation import compare_prompts

    model, tokenizer = load_model(model_id)
    layer_indices = get_layer_indices(model)

    prompt_code = (
        "def quicksort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)"
    )

    prompt_philosophy = (
        "The unexamined life is not worth living, Socrates declared, yet the very act of "
        "examination introduces a paradox: to observe one's own consciousness is to alter it. "
        "Every moment of introspection creates a new state that was not there before the looking."
    )

    analysis_code = capture_and_analyze(
        model, tokenizer, prompt_code, layer_indices=layer_indices, pca_components=PCA_COMPONENTS
    )
    analysis_philosophy = capture_and_analyze(
        model,
        tokenizer,
        prompt_philosophy,
        layer_indices=layer_indices,
        pca_components=PCA_COMPONENTS,
    )

    # Re-run code prompt for self-comparison control
    analysis_code_2 = capture_and_analyze(
        model, tokenizer, prompt_code, layer_indices=layer_indices, pca_components=PCA_COMPONENTS
    )

    # Compare at k=10 (middle PCA component count)
    rotation_code_vs_philosophy = compare_prompts(analysis_code, analysis_philosophy, pca_k=10)
    rotation_code_vs_code = compare_prompts(analysis_code, analysis_code_2, pca_k=10)

    # Convert RotationSummary objects to JSON-serializable dicts
    def _serialize_rotation(rotation_map: dict) -> dict:
        result = {}
        for layer_key, summary in rotation_map.items():
            result[layer_key] = {
                "principal_angles": summary.principal_angles.tolist(),
                "mean_angle": summary.mean_angle,
                "max_angle": summary.max_angle,
                "grassmann_distance": summary.grassmann_distance,
            }
        return result

    return {
        "model_id": model_id,
        "code_vs_philosophy": _serialize_rotation(rotation_code_vs_philosophy),
        "code_vs_code": _serialize_rotation(rotation_code_vs_code),
    }
