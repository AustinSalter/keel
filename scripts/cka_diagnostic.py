"""CKA diagnostic: context-only vs context+prompt at L24 for all pilot conditions.

Tests the over-anchoring / healthy-blend / context-override hypothesis:
- CKA > 0.8: prompt not reaching L24 (over-anchoring)
- CKA 0.3-0.6: healthy blend
- CKA < 0.1: prompt overriding context (context irrelevant)
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from substrate.config import SWEEP_LAYERS, TRINITY_MINI_ID


def main() -> None:
    prompts_data = json.loads(Path("data/prompts.json").read_text())["prompts"]
    payloads_dir = Path("data/payloads")

    # All 3 pilot prompts x all context conditions
    test_prompts = {
        "ai_ml": next(p for p in prompts_data if p["id"] == 1)["text"],
        "food": next(p for p in prompts_data if p["id"] == 14)["text"],
        "krebs": next(p for p in prompts_data if p["id"] == 25)["text"],
    }

    test_contexts = {
        "p1": (payloads_dir / "p1_questionnaire.txt").read_text(),
        "p_soul": (payloads_dir / "p_soul.txt").read_text(),
        "p_phello_ai": (payloads_dir / "phello/prompt_01.txt").read_text(),
        "p_phello_food": (payloads_dir / "phello/prompt_14.txt").read_text(),
        "p_phello_krebs": (payloads_dir / "phello/prompt_25.txt").read_text(),
    }

    diag_fn = modal.Function.from_name("keel-substrate", "cka_diagnostic")

    print("Running CKA diagnostic on Modal...")
    print("Comparing context-only vs context+prompt activations at each layer.\n")

    result = diag_fn.remote(
        model_id=TRINITY_MINI_ID,
        layer_indices=SWEEP_LAYERS,
        prompts=test_prompts,
        contexts=test_contexts,
    )

    # Print results
    print("=" * 80)
    print("CKA: context-only vs context+prompt (how much does the prompt change L24?)")
    print("=" * 80)
    print()
    print("  Interpretation:")
    print("    > 0.8  = over-anchoring (prompt not reaching the layer)")
    print("    0.3-0.6 = healthy blend (context + prompt integrating)")
    print("    < 0.1  = context override (prompt dominates, context irrelevant)")
    print()

    cka_data = result["cka_context_vs_full"]

    print(f"  {'Context':<18} {'Prompt':<12}", end="")
    for idx in SWEEP_LAYERS:
        print(f"  {'L'+str(idx):>8}", end="")
    print(f"  {'L24 Diagnosis':>20}")
    print("  " + "-" * 78)

    for key, layers in cka_data.items():
        ctx_name, prompt_name = key.split("|")
        l24_val = layers.get(f"layer_{SWEEP_LAYERS[2]}", 0)

        if l24_val > 0.8:
            diagnosis = "OVER-ANCHORED"
        elif l24_val > 0.6:
            diagnosis = "strong context"
        elif l24_val > 0.3:
            diagnosis = "healthy blend"
        elif l24_val > 0.1:
            diagnosis = "weak context"
        else:
            diagnosis = "CONTEXT OVERRIDE"

        print(f"  {ctx_name:<18} {prompt_name:<12}", end="")
        for idx in SWEEP_LAYERS:
            val = layers.get(f"layer_{idx}", 0)
            print(f"  {val:>8.4f}", end="")
        print(f"  {diagnosis:>20}")

    # Also print the cosine similarity of mean activations for interpretability
    print()
    print("=" * 80)
    print("COSINE SIMILARITY: mean activation vectors (context-only vs context+prompt)")
    print("=" * 80)
    print()

    cos_data = result["cosine_context_vs_full"]
    print(f"  {'Context':<18} {'Prompt':<12}", end="")
    for idx in SWEEP_LAYERS:
        print(f"  {'L'+str(idx):>8}", end="")
    print()
    print("  " + "-" * 58)

    for key, layers in cos_data.items():
        ctx_name, prompt_name = key.split("|")
        print(f"  {ctx_name:<18} {prompt_name:<12}", end="")
        for idx in SWEEP_LAYERS:
            val = layers.get(f"layer_{idx}", 0)
            print(f"  {val:>8.4f}", end="")
        print()

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
