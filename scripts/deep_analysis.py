"""Deep analysis of pilot results: cross-prompt geometry, context decomposition, EVR, and text comparison.

Answers five questions:
1. Do P_null + AI prompt and P_null + Krebs produce different geometries from each other?
2. Does the model answer Krebs differently with personal context injected?
3. What fraction of geometry is context-derived vs prompt-derived?
4. (answered by reading payloads — see P_phello vs P1 for prompt 14)
5. What does EVR look like at each condition?
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal

from substrate.config import SWEEP_LAYERS, TRINITY_MINI_ID


def main() -> None:
    prompts_data = json.loads(Path("data/prompts.json").read_text())["prompts"]
    pilot_prompts = [
        next(p for p in prompts_data if p["id"] == 1),   # AI/ML
        next(p for p in prompts_data if p["id"] == 14),  # Food
        next(p for p in prompts_data if p["id"] == 25),  # Control
    ]

    payloads_dir = Path("data/payloads")
    p_null = ""
    p1 = (payloads_dir / "p1_questionnaire.txt").read_text()
    p_phello_14 = (payloads_dir / "phello/prompt_14.txt").read_text()
    p_phello_25 = (payloads_dir / "phello/prompt_25.txt").read_text()

    deep_fn = modal.Function.from_name("keel-substrate", "deep_analysis")

    print("Running deep analysis on Modal (5 captures)...")
    print("This will take a few minutes.\n")

    result = deep_fn.remote(
        model_id=TRINITY_MINI_ID,
        layer_indices=SWEEP_LAYERS,
        prompts={
            "ai_ml": pilot_prompts[0]["text"],
            "food": pilot_prompts[1]["text"],
            "krebs": pilot_prompts[2]["text"],
        },
        payloads={
            "p_null": p_null,
            "p1": p1,
            "p_phello_food": p_phello_14,
            "p_phello_krebs": p_phello_25,
        },
    )

    # ---- Q1: Cross-prompt geometry under P_null ----
    print("=" * 70)
    print("Q1: Do different prompts produce different geometries under P_null?")
    print("=" * 70)
    q1 = result["q1_cross_prompt"]
    for comparison, rot in q1.items():
        print(f"\n  {comparison}:")
        for layer_key, data in rot.items():
            print(f"    {layer_key}: GD={data['grassmann_distance']:.4f}  "
                  f"mean={math.degrees(data['mean_angle']):.1f}°")

    # ---- Q2: Text comparison — Krebs with and without context ----
    print("\n" + "=" * 70)
    print("Q2: Does the model answer Krebs differently with personal context?")
    print("=" * 70)
    q2 = result["q2_text_comparison"]
    for condition, text in q2.items():
        print(f"\n  [{condition}] (first 300 chars):")
        print(f"    {text[:300]}...")

    # ---- Q3: Context vs prompt decomposition ----
    print("\n" + "=" * 70)
    print("Q3: What fraction of geometry is context-derived vs prompt-derived?")
    print("=" * 70)
    q3 = result["q3_decomposition"]
    print(f"\n  {'Prompt':<12} {'Layer':<10} {'Ctx→Full GD':>12} {'Prm→Full GD':>12} "
          f"{'Ctx→Prm GD':>12} {'Ctx Share':>10}")
    print("  " + "-" * 68)
    for entry in q3:
        ctx_share = entry["context_share"]
        print(f"  {entry['prompt']:<12} {entry['layer']:<10} "
              f"{entry['context_to_full']:.4f}        "
              f"{entry['prompt_to_full']:.4f}        "
              f"{entry['context_to_prompt']:.4f}        "
              f"{ctx_share:.1%}")

    # ---- Q5: EVR across conditions ----
    print("\n" + "=" * 70)
    print("Q5: Explained variance ratio across conditions")
    print("=" * 70)
    q5 = result["q5_evr"]
    print(f"\n  {'Condition':<30} ", end="")
    for idx in SWEEP_LAYERS:
        print(f"{'L' + str(idx):>8} ", end="")
    print()
    print("  " + "-" * 56)
    for condition, layers in q5.items():
        print(f"  {condition:<30} ", end="")
        for idx in SWEEP_LAYERS:
            evr = layers.get(f"layer_{idx}", 0)
            print(f"{evr:>8.4f} ", end="")
        print()

    print("\nDeep analysis complete.")


if __name__ == "__main__":
    main()
