"""Modal app for running substrate geometry inference on remote A100 GPUs.

Provides Modal functions:
  - verify_pipeline: Sprint 0 acceptance test (capture + PCA on a simple prompt)
  - profile_memory: VRAM usage measurements at various sequence lengths
  - sanity_check: Cross-prompt subspace rotation comparison (code vs philosophy)
  - layer_sweep: Full 32-layer sweep with subtle prompt pairs for layer selection
  - pilot_capture: Single (context, prompt) pair activation capture with rotation
  - deep_analysis: Cross-prompt geometry, context decomposition, EVR, text generation
  - evr_elbow_analysis: EVR elbow analysis + full-dimensional comparison (CKA, cosine)
  - cka_diagnostic: CKA diagnostic comparing context-only vs context+prompt
  - coherence_experiment: SOUL context + prompts x completions coherence correlation
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


@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=1800,
)
def deep_analysis(
    model_id: str,
    layer_indices: list[int],
    prompts: dict[str, str],
    payloads: dict[str, str],
) -> dict:
    """Deep analysis: cross-prompt geometry, context decomposition, EVR, text generation.

    Args:
        prompts: {"ai_ml": "...", "food": "...", "krebs": "..."}
        payloads: {"p_null": "", "p1": "...", "p_phello_food": "...", "p_phello_krebs": "..."}
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    import numpy as np
    import torch

    from substrate.capture import compute_pca_basis
    from substrate.hooks import ActivationCollector
    from substrate.rotation import compute_rotation_summary

    model, tokenizer = load_model(model_id)
    device = next(model.parameters()).device
    k = 10

    def capture(text: str) -> dict[str, np.ndarray]:
        """Run forward pass and capture activations at all target layers."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k_: v.to(device) for k_, v in inputs.items()}
        collector = ActivationCollector()
        collector.register(model, layer_indices=layer_indices)
        with torch.no_grad():
            model(**inputs)
        raw = collector.collect()
        collector.remove_all()
        return {key: val.numpy() for key, val in raw.items()}

    def pca_at_layers(acts: dict[str, np.ndarray], max_k: int = k):
        """Compute PCA at each layer."""
        result = {}
        for layer_key, act in acts.items():
            ek = min(max_k, act.shape[0], act.shape[1])
            if ek >= 2:
                result[layer_key] = compute_pca_basis(act, n_components=ek)
        return result

    def rotation_dict(pca_a, pca_b):
        """Compute rotation between two PCA results across shared layers."""
        result = {}
        for layer_key in pca_a:
            if layer_key in pca_b:
                ek = min(pca_a[layer_key].components.shape[0],
                         pca_b[layer_key].components.shape[0])
                rot = compute_rotation_summary(
                    pca_a[layer_key].components[:ek],
                    pca_b[layer_key].components[:ek],
                )
                result[layer_key] = {
                    "grassmann_distance": rot.grassmann_distance,
                    "mean_angle": rot.mean_angle,
                    "max_angle": rot.max_angle,
                }
        return result

    # ---- Capture all conditions ----

    # P_null + each prompt (for Q1 cross-prompt comparison)
    null_ai = capture(prompts["ai_ml"])
    null_food = capture(prompts["food"])
    null_krebs = capture(prompts["krebs"])

    # Context-only captures (for Q3 decomposition)
    ctx_p1 = capture(payloads["p1"])
    ctx_phello_krebs = capture(payloads["p_phello_krebs"])

    # P1 + krebs (for Q3)
    p1_krebs = capture(f"{payloads['p1']}\n\n{prompts['krebs']}")

    # P_null + krebs and P1 + krebs already captured
    # P_phello + food (for Q5 comparison)
    phello_food = capture(f"{payloads['p_phello_food']}\n\n{prompts['food']}")
    p1_food = capture(f"{payloads['p1']}\n\n{prompts['food']}")

    # P1 + ai (for Q5)
    p1_ai = capture(f"{payloads['p1']}\n\n{prompts['ai_ml']}")

    # ---- Q1: Cross-prompt geometry under P_null ----
    pca_null_ai = pca_at_layers(null_ai)
    pca_null_food = pca_at_layers(null_food)
    pca_null_krebs = pca_at_layers(null_krebs)

    q1 = {
        "ai_vs_food": rotation_dict(pca_null_ai, pca_null_food),
        "ai_vs_krebs": rotation_dict(pca_null_ai, pca_null_krebs),
        "food_vs_krebs": rotation_dict(pca_null_food, pca_null_krebs),
    }

    # ---- Q2: Generate text with and without context for Krebs ----
    gen_kwargs = dict(max_new_tokens=200, temperature=0.1, do_sample=True)

    def generate(text: str) -> str:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k_: v.to(device) for k_, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        # Decode only the new tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    krebs_text = prompts["krebs"]
    q2 = {
        "p_null": generate(krebs_text),
        "p1": generate(f"{payloads['p1']}\n\n{krebs_text}"),
        "p_phello": generate(f"{payloads['p_phello_krebs']}\n\n{krebs_text}"),
    }

    # ---- Q3: Context vs prompt decomposition ----
    # For the Krebs prompt with P1 context:
    # A = context-only geometry, B = prompt-only geometry, C = context+prompt geometry
    # Context share = how much of C is explained by A vs B
    pca_ctx_p1 = pca_at_layers(ctx_p1)
    pca_null_krebs_ = pca_at_layers(null_krebs)  # prompt-only
    pca_p1_krebs = pca_at_layers(p1_krebs)       # context+prompt

    q3 = []
    for layer_key in pca_p1_krebs:
        if layer_key not in pca_ctx_p1 or layer_key not in pca_null_krebs_:
            continue
        ek = min(
            pca_ctx_p1[layer_key].components.shape[0],
            pca_null_krebs_[layer_key].components.shape[0],
            pca_p1_krebs[layer_key].components.shape[0],
        )
        rot_ctx_full = compute_rotation_summary(
            pca_ctx_p1[layer_key].components[:ek],
            pca_p1_krebs[layer_key].components[:ek],
        )
        rot_prm_full = compute_rotation_summary(
            pca_null_krebs_[layer_key].components[:ek],
            pca_p1_krebs[layer_key].components[:ek],
        )
        rot_ctx_prm = compute_rotation_summary(
            pca_ctx_p1[layer_key].components[:ek],
            pca_null_krebs_[layer_key].components[:ek],
        )
        # Context share: how much closer is context to full vs prompt to full
        total = rot_ctx_full.grassmann_distance + rot_prm_full.grassmann_distance
        ctx_share = 1.0 - (rot_ctx_full.grassmann_distance / total) if total > 0 else 0.5

        q3.append({
            "prompt": "krebs",
            "layer": layer_key,
            "context_to_full": rot_ctx_full.grassmann_distance,
            "prompt_to_full": rot_prm_full.grassmann_distance,
            "context_to_prompt": rot_ctx_prm.grassmann_distance,
            "context_share": ctx_share,
        })

    # Also do for AI/ML prompt
    pca_p1_ai = pca_at_layers(p1_ai)
    for layer_key in pca_p1_ai:
        if layer_key not in pca_ctx_p1 or layer_key not in pca_null_ai:
            continue
        pca_nai = pca_at_layers(null_ai)
        ek = min(
            pca_ctx_p1[layer_key].components.shape[0],
            pca_nai[layer_key].components.shape[0],
            pca_p1_ai[layer_key].components.shape[0],
        )
        rot_ctx_full = compute_rotation_summary(
            pca_ctx_p1[layer_key].components[:ek],
            pca_p1_ai[layer_key].components[:ek],
        )
        rot_prm_full = compute_rotation_summary(
            pca_nai[layer_key].components[:ek],
            pca_p1_ai[layer_key].components[:ek],
        )
        rot_ctx_prm = compute_rotation_summary(
            pca_ctx_p1[layer_key].components[:ek],
            pca_nai[layer_key].components[:ek],
        )
        total = rot_ctx_full.grassmann_distance + rot_prm_full.grassmann_distance
        ctx_share = 1.0 - (rot_ctx_full.grassmann_distance / total) if total > 0 else 0.5

        q3.append({
            "prompt": "ai_ml",
            "layer": layer_key,
            "context_to_full": rot_ctx_full.grassmann_distance,
            "prompt_to_full": rot_prm_full.grassmann_distance,
            "context_to_prompt": rot_ctx_prm.grassmann_distance,
            "context_share": ctx_share,
        })

    # ---- Q5: EVR across conditions ----
    conditions = {
        "p_null + ai_ml": pca_null_ai,
        "p_null + food": pca_null_food,
        "p_null + krebs": pca_null_krebs,
        "p1 + ai_ml": pca_p1_ai,
        "p1 + food": pca_at_layers(p1_food),
        "p1 + krebs": pca_p1_krebs,
        "p_phello + food": pca_at_layers(phello_food),
        "p1_context_only": pca_ctx_p1,
    }

    q5 = {}
    for cond_name, pca_data in conditions.items():
        q5[cond_name] = {}
        for layer_key, pca_result in pca_data.items():
            q5[cond_name][layer_key] = float(pca_result.explained_variance_ratio.sum())

    return {
        "q1_cross_prompt": q1,
        "q2_text_comparison": q2,
        "q3_decomposition": q3,
        "q5_evr": q5,
    }


@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=1800,
)
def evr_elbow_analysis(
    model_id: str,
    layer_indices: list[int],
    conditions: dict[str, dict[str, str]],
) -> dict:
    """EVR elbow analysis + full-dimensional comparison (CKA, cosine).

    Args:
        conditions: {"name": {"context": "...", "prompt": "..."}, ...}
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from scipy.linalg import subspace_angles

    from substrate.hooks import ActivationCollector

    model, tokenizer = load_model(model_id)
    device = next(model.parameters()).device

    k_values = [10, 20, 30, 50, 100]

    def capture(text: str) -> dict[str, np.ndarray]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k_: v.to(device) for k_, v in inputs.items()}
        collector = ActivationCollector()
        collector.register(model, layer_indices=layer_indices)
        with torch.no_grad():
            model(**inputs)
        raw = collector.collect()
        collector.remove_all()
        return {key: val.numpy() for key, val in raw.items()}

    # ---- Capture all conditions ----
    activations = {}
    for cond_name, cond in conditions.items():
        ctx = cond["context"]
        prompt = cond["prompt"]
        full_text = f"{ctx}\n\n{prompt}" if ctx else prompt
        activations[cond_name] = capture(full_text)
        print(f"  Captured {cond_name}: seq_len={next(iter(activations[cond_name].values())).shape[0]}")

    # ---- EVR curves at multiple k values ----
    evr_curves = {}
    for cond_name, layer_acts in activations.items():
        evr_curves[cond_name] = {}
        for layer_key, act in layer_acts.items():
            seq_len = act.shape[0]
            hidden = act.shape[1]
            max_possible_k = min(seq_len, hidden)
            # Run PCA at the largest k we can, then slice for smaller k values
            actual_max_k = min(max(k_values), max_possible_k)
            if actual_max_k < 2:
                continue
            pca = PCA(n_components=actual_max_k)
            pca.fit(act)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            evr_curves[cond_name][layer_key] = [
                float(cumulative[min(k, actual_max_k) - 1]) for k in k_values
            ]

    # ---- PCA Grassmann distance at multiple k values ----
    # Compare: null vs p1, null vs p_phello, p1 vs p_phello (for food prompt)
    comparisons = [
        ("null_vs_p1_food", "p_null+food", "p1+food"),
        ("null_vs_phello_food", "p_null+food", "p_phello+food"),
        ("p1_vs_phello_food", "p1+food", "p_phello+food"),
        ("null_vs_p1_krebs", "p_null+krebs", "p1+krebs"),
    ]

    pca_gd_by_k = {}
    for comp_name, cond_a, cond_b in comparisons:
        pca_gd_by_k[comp_name] = {}
        for layer_key in activations[cond_a]:
            if layer_key not in activations[cond_b]:
                continue
            act_a = activations[cond_a][layer_key]
            act_b = activations[cond_b][layer_key]
            max_k = min(act_a.shape[0], act_b.shape[0], act_a.shape[1], max(k_values))

            pca_a = PCA(n_components=max_k)
            pca_a.fit(act_a)
            pca_b = PCA(n_components=max_k)
            pca_b.fit(act_b)

            gd_values = []
            for k in k_values:
                ek = min(k, max_k)
                if ek < 2:
                    gd_values.append(0.0)
                    continue
                angles = subspace_angles(
                    pca_a.components_[:ek].T,
                    pca_b.components_[:ek].T,
                )
                gd = float(np.sqrt(np.sum(np.sin(angles) ** 2)))
                gd_values.append(gd)
            pca_gd_by_k[comp_name][layer_key] = gd_values

    # ---- Full-dimensional comparison ----
    def cosine_sim_mean(act_a: np.ndarray, act_b: np.ndarray) -> float:
        """Mean cosine similarity between centered activation matrices."""
        a_centered = act_a - act_a.mean(axis=0)
        b_centered = act_b - act_b.mean(axis=0)
        # Compare the mean activation vectors
        mean_a = a_centered.mean(axis=0)
        mean_b = b_centered.mean(axis=0)
        dot = np.dot(mean_a, mean_b)
        norm = np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
        return float(dot / norm) if norm > 0 else 0.0

    def linear_cka(act_a: np.ndarray, act_b: np.ndarray) -> float:
        """Feature-space Linear CKA for activation matrices with different sample sizes.

        Compares covariance structure in feature space (d x d), which is
        independent of sequence length. CKA=1 means identical covariance
        structure. CKA=0 means completely unrelated.
        """
        a = act_a - act_a.mean(axis=0)
        b = act_b - act_b.mean(axis=0)

        # Feature covariance matrices [d, d] — normalized by sample count
        feat_a = a.T @ a / a.shape[0]
        feat_b = b.T @ b / b.shape[0]

        numerator = np.linalg.norm(feat_a @ feat_b, "fro") ** 2
        denom = np.linalg.norm(feat_a, "fro") * np.linalg.norm(feat_b, "fro")
        return float(numerator / (denom ** 2)) if denom > 0 else 0.0

    full_dim = {"cosine_similarity": {}, "cka": {}}
    for comp_name, cond_a, cond_b in comparisons:
        full_dim["cosine_similarity"][comp_name] = {}
        full_dim["cka"][comp_name] = {}
        for layer_key in activations[cond_a]:
            if layer_key not in activations[cond_b]:
                continue
            act_a = activations[cond_a][layer_key]
            act_b = activations[cond_b][layer_key]
            full_dim["cosine_similarity"][comp_name][layer_key] = cosine_sim_mean(act_a, act_b)
            full_dim["cka"][comp_name][layer_key] = linear_cka(act_a, act_b)

    return {
        "k_values": k_values,
        "evr_curves": evr_curves,
        "pca_gd_by_k": pca_gd_by_k,
        "full_dimensional": full_dim,
    }


@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=1800,
)
def cka_diagnostic(
    model_id: str,
    layer_indices: list[int],
    prompts: dict[str, str],
    contexts: dict[str, str],
) -> dict:
    """CKA diagnostic: compare context-only vs context+prompt at each layer.

    For each (context, prompt) pair, captures activations for:
    1. Context alone
    2. Context + prompt combined

    Then computes CKA and cosine similarity between them at each layer.
    This measures how much the prompt changes the representation established
    by the context — the over-anchoring / healthy-blend / override signal.
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    import numpy as np
    import torch

    from substrate.hooks import ActivationCollector

    model, tokenizer = load_model(model_id)
    device = next(model.parameters()).device

    def capture(text: str) -> dict[str, np.ndarray]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k_: v.to(device) for k_, v in inputs.items()}
        collector = ActivationCollector()
        collector.register(model, layer_indices=layer_indices)
        with torch.no_grad():
            model(**inputs)
        raw = collector.collect()
        collector.remove_all()
        return {key: val.numpy() for key, val in raw.items()}

    def feature_cka(act_a: np.ndarray, act_b: np.ndarray) -> float:
        """Feature-space CKA for matrices with different sample sizes."""
        a = act_a - act_a.mean(axis=0)
        b = act_b - act_b.mean(axis=0)
        feat_a = a.T @ a / a.shape[0]
        feat_b = b.T @ b / b.shape[0]
        numerator = np.linalg.norm(feat_a @ feat_b, "fro") ** 2
        denom = np.linalg.norm(feat_a, "fro") * np.linalg.norm(feat_b, "fro")
        return float(numerator / (denom ** 2)) if denom > 0 else 0.0

    def cosine_mean(act_a: np.ndarray, act_b: np.ndarray) -> float:
        """Cosine similarity between mean activation vectors."""
        a = act_a - act_a.mean(axis=0)
        b = act_b - act_b.mean(axis=0)
        mean_a = a.mean(axis=0)
        mean_b = b.mean(axis=0)
        dot = np.dot(mean_a, mean_b)
        norm = np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
        return float(dot / norm) if norm > 0 else 0.0

    # Map context names to prompts they should be paired with
    # p_phello variants are query-specific, so pair them with their target prompt
    pairings = []
    for ctx_name, ctx_text in contexts.items():
        for prompt_name, prompt_text in prompts.items():
            # Skip mismatched phello pairings (phello_food with krebs, etc.)
            if ctx_name.startswith("p_phello_"):
                phello_domain = ctx_name.split("p_phello_")[1]
                if phello_domain != prompt_name:
                    continue
            pairings.append((ctx_name, prompt_name, ctx_text, prompt_text))

    cka_results = {}
    cos_results = {}

    for ctx_name, prompt_name, ctx_text, prompt_text in pairings:
        key = f"{ctx_name}|{prompt_name}"
        print(f"  Capturing {key}...")

        # Context-only activations
        acts_ctx = capture(ctx_text)

        # Context + prompt activations
        full_text = f"{ctx_text}\n\n{prompt_text}"
        acts_full = capture(full_text)

        cka_results[key] = {}
        cos_results[key] = {}

        for layer_key in acts_ctx:
            if layer_key in acts_full:
                cka_results[key][layer_key] = feature_cka(acts_ctx[layer_key], acts_full[layer_key])
                cos_results[key][layer_key] = cosine_mean(acts_ctx[layer_key], acts_full[layer_key])

    return {
        "cka_context_vs_full": cka_results,
        "cosine_context_vs_full": cos_results,
    }


@app.function(
    gpu="A100",
    image=image,
    volumes={"/cache": model_cache},
    timeout=3600,
)
def coherence_experiment(
    model_id: str,
    layer_indices: list[int],
    soul_text: str,
    prompts: list[dict],  # [{"id": int, "text": str, "category": str}, ...]
    completions_per_prompt: int = 20,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> list[dict]:
    """Coherence correlation experiment: SOUL context + prompts x completions, computing CKA at each step.

    For each prompt, captures context-only activations once (the SOUL geometry),
    then generates multiple completions and measures how each completion shifts the
    activation geometry via feature-space CKA at each target layer.

    Returns a list of result dicts, one per (prompt, completion) pair.
    """
    import os

    os.environ["HF_HOME"] = "/cache"

    import numpy as np
    import torch

    from substrate.hooks import ActivationCollector

    model, tokenizer = load_model(model_id)
    device = next(model.parameters()).device

    def capture(text: str) -> dict[str, np.ndarray]:
        """Run forward pass and capture activations at target layers."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k_: v.to(device) for k_, v in inputs.items()}
        collector = ActivationCollector()
        collector.register(model, layer_indices=layer_indices)
        with torch.no_grad():
            model(**inputs)
        raw = collector.collect()
        collector.remove_all()
        return {key: val.numpy() for key, val in raw.items()}

    def feature_cka(act_a: np.ndarray, act_b: np.ndarray) -> float:
        """Feature-space CKA for matrices with different sample sizes."""
        a = act_a - act_a.mean(axis=0)
        b = act_b - act_b.mean(axis=0)
        feat_a = a.T @ a / a.shape[0]
        feat_b = b.T @ b / b.shape[0]
        numerator = np.linalg.norm(feat_a @ feat_b, "fro") ** 2
        denom = np.linalg.norm(feat_a, "fro") * np.linalg.norm(feat_b, "fro")
        return float(numerator / (denom ** 2)) if denom > 0 else 0.0

    results: list[dict] = []
    total_completions = len(prompts) * completions_per_prompt
    completed = 0

    for prompt in prompts:
        prompt_id = prompt["id"]
        prompt_text = prompt["text"]
        prompt_category = prompt["category"]
        print(f"  Prompt {prompt_id} ({prompt_category}): capturing context-only activations...")

        # Capture context-only activations ONCE per prompt (SOUL geometry doesn't change)
        acts_context = capture(soul_text)

        for i in range(completions_per_prompt):
            # Generate completion
            full_input = f"{soul_text}\n\n{prompt_text}"
            inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k_: v.to(device) for k_, v in inputs.items()}
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
            completion_text = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Capture post-completion activations
            post_text = f"{soul_text}\n\n{prompt_text}{completion_text}"
            acts_post = capture(post_text)

            # Compute CKA at each layer
            result_entry: dict = {
                "prompt_id": prompt_id,
                "prompt_category": prompt_category,
                "completion_idx": i,
                "completion_text": completion_text,
            }
            for idx in layer_indices:
                layer_key = f"layer_{idx}"
                if layer_key in acts_context and layer_key in acts_post:
                    result_entry[f"cka_{layer_key}"] = feature_cka(
                        acts_context[layer_key], acts_post[layer_key]
                    )
                else:
                    result_entry[f"cka_{layer_key}"] = 0.0

            results.append(result_entry)
            completed += 1

            if completed % 10 == 0:
                print(f"  Progress: {completed}/{total_completions} completions done")

    print(f"  Experiment complete: {len(results)} results collected")
    return results
