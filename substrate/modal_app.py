"""Modal app for running substrate geometry inference on remote A100 GPUs.

Provides Modal functions:
  - verify_pipeline: Sprint 0 acceptance test (capture + PCA on a simple prompt)
  - profile_memory: VRAM usage measurements at various sequence lengths
  - sanity_check: Cross-prompt subspace rotation comparison (code vs philosophy)
  - layer_sweep: Full 32-layer sweep with subtle prompt pairs for layer selection
"""

from __future__ import annotations

import modal

app = modal.App("keel-substrate")

# Docker image with all dependencies + local substrate package
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.57.6",  # Trinity Mini requires 4.57.x (custom afmoe code)
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
# Helpers (run inside Modal container, not Modal functions themselves)
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
    """Sprint 0 acceptance test: capture activations and run PCA on a simple prompt."""
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
    """Measure GPU VRAM at various stages."""
    import os

    os.environ["HF_HOME"] = "/cache"

    import torch

    from substrate.hooks import ActivationCollector

    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model(model_id)
    layer_indices = get_layer_indices(model)
    baseline_bytes = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    inputs = tokenizer("Hello, world!", return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)
    forward_no_hooks_bytes = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    collector = ActivationCollector()
    collector.register(model, layer_indices=layer_indices)
    with torch.no_grad():
        model(**inputs)
    _ = collector.collect()
    collector.remove_all()
    forward_with_hooks_bytes = torch.cuda.max_memory_allocated()

    seq_length_measurements: dict[str, int] = {}
    for seq_len in [512, 1024, 2048, 4096]:
        torch.cuda.reset_peak_memory_stats()
        dummy_text = "word " * (seq_len // 2)
        seq_inputs = tokenizer(
            dummy_text, return_tensors="pt", truncation=True,
            max_length=seq_len, padding="max_length",
        )
        seq_inputs = {k: v.to(device) for k, v in seq_inputs.items()}
        collector = ActivationCollector()
        collector.register(model, layer_indices=layer_indices)
        with torch.no_grad():
            model(**seq_inputs)
        _ = collector.collect()
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
    """Cross-prompt subspace rotation comparison: code vs philosophy."""
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
        model, tokenizer, prompt_philosophy,
        layer_indices=layer_indices, pca_components=PCA_COMPONENTS,
    )
    analysis_code_2 = capture_and_analyze(
        model, tokenizer, prompt_code, layer_indices=layer_indices, pca_components=PCA_COMPONENTS
    )

    rotation_code_vs_philosophy = compare_prompts(analysis_code, analysis_philosophy, pca_k=10)
    rotation_code_vs_code = compare_prompts(analysis_code, analysis_code_2, pca_k=10)

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


@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=1800,
)
def layer_sweep(model_id: str) -> dict:
    """Full layer sweep: capture at ALL layers with subtle prompt pairs.

    Three test conditions:
    1. Same-domain different framing (risk vs benefit analysis)
    2. With/without personal context preamble
    3. Self-comparison control (run condition 1a twice)

    For each layer, returns:
    - Grassmann distance and angles for each comparison
    - Explained variance concentration at k=10
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    import numpy as np

    from substrate.capture import capture_activations, compute_pca_basis
    from substrate.rotation import compute_rotation_summary

    model, tokenizer = load_model(model_id)
    num_layers = len(model.model.layers)
    all_layers = list(range(num_layers))

    # --- Prompt definitions ---

    # Condition 1: Same-domain, different framing
    prompt_risk = (
        "What are the risks of vertical integration for an early-stage marketplace? "
        "Consider the operational, financial, and strategic downsides that could "
        "threaten a startup pursuing this approach."
    )
    prompt_benefit = (
        "What are the benefits of vertical integration for an early-stage marketplace? "
        "Consider the operational, financial, and strategic advantages that could "
        "give a startup pursuing this approach a competitive edge."
    )

    # Condition 2: With/without personal context
    prompt_cold = "What should I prioritize this quarter?"

    context_preamble = (
        "I'm the technical founder of an early-stage startup building preference "
        "infrastructure — tools that help AI systems understand and maintain coherence "
        "with individual user preferences across sessions. I have a background in ML "
        "engineering and I'm currently the sole developer. We have a working prototype "
        "but no revenue yet. Our key hypothesis is that activation geometry in language "
        "models can serve as a structural coherence metric. I'm balancing research "
        "validation, product development, and fundraising.\n\n"
    )
    prompt_warm = context_preamble + "What should I prioritize this quarter?"

    # --- Capture activations at ALL layers ---

    import torch

    def capture_all_layers(prompt: str) -> dict[int, np.ndarray]:
        """Capture residual stream at every layer, return {layer_idx: activations}."""
        from substrate.hooks import ActivationCollector

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        collector = ActivationCollector()
        collector.register(model, layer_indices=all_layers)
        with torch.no_grad():
            model(**inputs)
        raw = collector.collect()
        collector.remove_all()

        return {idx: raw[f"layer_{idx}"].numpy() for idx in all_layers}

    # Run all captures
    acts_risk = capture_all_layers(prompt_risk)
    acts_benefit = capture_all_layers(prompt_benefit)
    acts_risk_2 = capture_all_layers(prompt_risk)  # Self-comparison control
    acts_cold = capture_all_layers(prompt_cold)
    acts_warm = capture_all_layers(prompt_warm)

    # --- Per-layer analysis ---

    k = 10
    results_per_layer: dict[str, dict] = {}

    # Determine effective k for each prompt (min of seq_len and desired k)
    def effective_k(acts: dict[int, np.ndarray], desired_k: int) -> int:
        sample_act = next(iter(acts.values()))
        return min(desired_k, sample_act.shape[0], sample_act.shape[1])

    k_risk = effective_k(acts_risk, k)
    k_cold = effective_k(acts_cold, k)
    k_warm = effective_k(acts_warm, k)
    # Use the minimum across context prompts so subspaces are comparable
    k_context = min(k_cold, k_warm)

    for idx in all_layers:
        layer_key = f"layer_{idx}"

        # PCA at this layer for each prompt
        pca_risk = compute_pca_basis(acts_risk[idx], n_components=k_risk)
        pca_benefit = compute_pca_basis(acts_benefit[idx], n_components=k_risk)
        pca_risk_2 = compute_pca_basis(acts_risk_2[idx], n_components=k_risk)
        pca_cold = compute_pca_basis(acts_cold[idx], n_components=k_context)
        pca_warm = compute_pca_basis(acts_warm[idx], n_components=k_context)

        # Rotation summaries
        rot_framing = compute_rotation_summary(pca_risk.components, pca_benefit.components)
        rot_context = compute_rotation_summary(pca_cold.components, pca_warm.components)
        rot_self = compute_rotation_summary(pca_risk.components, pca_risk_2.components)

        # Explained variance concentration
        evr_risk = pca_risk.explained_variance_ratio
        evr_cold = pca_cold.explained_variance_ratio
        evr_warm = pca_warm.explained_variance_ratio

        results_per_layer[layer_key] = {
            "layer_idx": idx,
            # Condition 1: framing difference
            "framing": {
                "grassmann_distance": rot_framing.grassmann_distance,
                "mean_angle": rot_framing.mean_angle,
                "max_angle": rot_framing.max_angle,
                "principal_angles": rot_framing.principal_angles.tolist(),
            },
            # Condition 2: context injection
            "context": {
                "grassmann_distance": rot_context.grassmann_distance,
                "mean_angle": rot_context.mean_angle,
                "max_angle": rot_context.max_angle,
                "principal_angles": rot_context.principal_angles.tolist(),
            },
            # Control: self-comparison
            "self_comparison": {
                "grassmann_distance": rot_self.grassmann_distance,
                "mean_angle": rot_self.mean_angle,
                "max_angle": rot_self.max_angle,
            },
            # Explained variance concentration
            "explained_variance": {
                "risk_prompt_cumulative_k10": float(evr_risk.sum()),
                "risk_prompt_top5_ratio": float(evr_risk[:5].sum()),
                "cold_prompt_cumulative_k10": float(evr_cold.sum()),
                "warm_prompt_cumulative_k10": float(evr_warm.sum()),
                "risk_prompt_evr": evr_risk.tolist(),
            },
        }

    return {
        "model_id": model_id,
        "num_layers": num_layers,
        "pca_k": k,
        "effective_k_framing": k_risk,
        "effective_k_context": k_context,
        "prompts": {
            "risk": prompt_risk,
            "benefit": prompt_benefit,
            "cold": prompt_cold,
            "warm": prompt_warm,
        },
        "layers": results_per_layer,
    }


@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
)
def pilot_capture(
    model_id: str,
    context: str,
    prompt: str,
    layer_indices: list[int],
) -> dict:
    """Capture activations for a single (context, prompt) pair.

    Injects context as a preamble, runs the prompt, captures PCA at each layer,
    then computes rotation between the context-only geometry and the context+prompt geometry.

    Returns JSON-serializable dict with rotation summaries per layer and EVR data.
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    import numpy as np

    from substrate.capture import capture_activations, compute_pca_basis
    from substrate.rotation import compute_rotation_summary

    model, tokenizer = load_model(model_id)

    k = 10

    # Capture 1: context only (pre-response geometry)
    context_text = context if context else "You are a helpful assistant."
    acts_context = {}
    from substrate.hooks import ActivationCollector
    import torch

    inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=4096)
    device = next(model.parameters()).device
    inputs = {k_: v.to(device) for k_, v in inputs.items()}
    collector = ActivationCollector()
    collector.register(model, layer_indices=layer_indices)
    with torch.no_grad():
        model(**inputs)
    raw_context = collector.collect()
    collector.remove_all()

    # Capture 2: context + prompt (the full input)
    full_text = f"{context}\n\n{prompt}" if context else prompt
    inputs2 = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs2 = {k_: v.to(device) for k_, v in inputs2.items()}
    collector2 = ActivationCollector()
    collector2.register(model, layer_indices=layer_indices)
    with torch.no_grad():
        model(**inputs2)
    raw_full = collector2.collect()
    collector2.remove_all()

    # Per-layer: PCA + rotation
    rotation_results = {}
    evr_results = {}
    for idx in layer_indices:
        layer_key = f"layer_{idx}"

        act_ctx = raw_context[layer_key].numpy()
        act_full = raw_full[layer_key].numpy()

        # Clamp k to min sequence length
        effective_k = min(k, act_ctx.shape[0], act_full.shape[0])
        if effective_k < 2:
            continue

        pca_ctx = compute_pca_basis(act_ctx, n_components=effective_k)
        pca_full = compute_pca_basis(act_full, n_components=effective_k)

        rot = compute_rotation_summary(pca_ctx.components, pca_full.components)
        rotation_results[layer_key] = {
            "grassmann_distance": rot.grassmann_distance,
            "mean_angle": rot.mean_angle,
            "max_angle": rot.max_angle,
            "principal_angles": rot.principal_angles.tolist(),
        }
        evr_results[layer_key] = {
            "context_evr_sum": float(pca_ctx.explained_variance_ratio.sum()),
            "full_evr_sum": float(pca_full.explained_variance_ratio.sum()),
        }

    return {
        "model_id": model_id,
        "rotation": rotation_results,
        "explained_variance": evr_results,
    }
