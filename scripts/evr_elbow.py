"""EVR elbow analysis + full-dimensional comparison (CKA, cosine similarity).

Tests whether PCA is the right decomposition for contextual geometry,
and if so, what k value captures the signal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal
import numpy as np
import matplotlib.pyplot as plt

from substrate.config import SWEEP_LAYERS, TRINITY_MINI_ID


def main() -> None:
    prompts_data = json.loads(Path("data/prompts.json").read_text())["prompts"]
    payloads_dir = Path("data/payloads")

    # Use the food prompt (strongest personal signal) and Krebs (control)
    food_prompt = next(p for p in prompts_data if p["id"] == 14)["text"]
    krebs_prompt = next(p for p in prompts_data if p["id"] == 25)["text"]

    p_null = ""
    p1 = (payloads_dir / "p1_questionnaire.txt").read_text()
    p_phello_food = (payloads_dir / "phello/prompt_14.txt").read_text()

    evr_fn = modal.Function.from_name("keel-substrate", "evr_elbow_analysis")

    print("Running EVR elbow + full-dimensional analysis on Modal...")
    print("This captures activations for 6 conditions and runs PCA at k=10,20,30,50,100.\n")

    result = evr_fn.remote(
        model_id=TRINITY_MINI_ID,
        layer_indices=SWEEP_LAYERS,
        conditions={
            "p_null+food": {"context": p_null, "prompt": food_prompt},
            "p1+food": {"context": p1, "prompt": food_prompt},
            "p_phello+food": {"context": p_phello_food, "prompt": food_prompt},
            "p_null+krebs": {"context": p_null, "prompt": krebs_prompt},
            "p1+krebs": {"context": p1, "prompt": krebs_prompt},
        },
    )

    # ---- Print EVR curves ----
    print("=" * 70)
    print("EVR CURVES: Cumulative explained variance by k")
    print("=" * 70)

    k_values = result["k_values"]
    evr_curves = result["evr_curves"]

    for cond_name, layers in evr_curves.items():
        print(f"\n  {cond_name}:")
        print(f"    {'k':>6}", end="")
        for idx in SWEEP_LAYERS:
            print(f"  {'L'+str(idx):>8}", end="")
        print()
        for i, k in enumerate(k_values):
            print(f"    {k:>6}", end="")
            for idx in SWEEP_LAYERS:
                lk = f"layer_{idx}"
                val = layers.get(lk, [0]*len(k_values))[i]
                print(f"  {val:>8.4f}", end="")
            print()

    # ---- Print full-dimensional comparison ----
    print("\n" + "=" * 70)
    print("FULL-DIMENSIONAL COMPARISON (no PCA reduction)")
    print("=" * 70)

    full_dim = result["full_dimensional"]

    print(f"\n  {'Comparison':<35}", end="")
    for idx in SWEEP_LAYERS:
        print(f"  {'L'+str(idx):>8}", end="")
    print()
    print("  " + "-" * 59)

    for metric_name in ["cosine_similarity", "cka"]:
        print(f"\n  {metric_name}:")
        for comp_name, layers in full_dim[metric_name].items():
            print(f"    {comp_name:<33}", end="")
            for idx in SWEEP_LAYERS:
                lk = f"layer_{idx}"
                val = layers.get(lk, 0)
                print(f"  {val:>8.4f}", end="")
            print()

    # ---- Print PCA-based comparison at various k ----
    print("\n" + "=" * 70)
    print("PCA GRASSMANN DISTANCE BY k")
    print("=" * 70)

    pca_gd = result["pca_gd_by_k"]
    for comp_name, k_data in pca_gd.items():
        print(f"\n  {comp_name}:")
        print(f"    {'k':>6}", end="")
        for idx in SWEEP_LAYERS:
            print(f"  {'L'+str(idx):>8}", end="")
        print()
        for i, k in enumerate(k_values):
            print(f"    {k:>6}", end="")
            for idx in SWEEP_LAYERS:
                lk = f"layer_{idx}"
                vals = k_data.get(lk, [0]*len(k_values))
                print(f"  {vals[i]:>8.4f}", end="")
            print()

    # ---- Plot ----
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: EVR curves for contextual conditions at L7
    ax = axes[0, 0]
    layer_key = f"layer_{SWEEP_LAYERS[0]}"
    for cond_name, layers in evr_curves.items():
        vals = layers.get(layer_key, [])
        if vals:
            ax.plot(k_values[:len(vals)], vals, "o-", label=cond_name, markersize=4)
    ax.set_xlabel("k (PCA components)")
    ax.set_ylabel("Cumulative EVR")
    ax.set_title(f"EVR Curves at {layer_key}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Plot 2: EVR curves at L11
    ax = axes[0, 1]
    layer_key = f"layer_{SWEEP_LAYERS[1]}"
    for cond_name, layers in evr_curves.items():
        vals = layers.get(layer_key, [])
        if vals:
            ax.plot(k_values[:len(vals)], vals, "o-", label=cond_name, markersize=4)
    ax.set_xlabel("k (PCA components)")
    ax.set_ylabel("Cumulative EVR")
    ax.set_title(f"EVR Curves at {layer_key}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Plot 3: Grassmann distance by k for null vs p1 (food prompt)
    ax = axes[1, 0]
    layer_key = f"layer_{SWEEP_LAYERS[0]}"
    for comp_name, k_data in pca_gd.items():
        vals = k_data.get(layer_key, [])
        if vals:
            ax.plot(k_values[:len(vals)], vals, "o-", label=comp_name, markersize=4)
    ax.set_xlabel("k (PCA components)")
    ax.set_ylabel("Grassmann Distance")
    ax.set_title(f"GD Stability by k at {layer_key}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 4: Full-dim comparison bar chart
    ax = axes[1, 1]
    cka = full_dim["cka"]
    comp_names = list(cka.keys())
    layer_key = f"layer_{SWEEP_LAYERS[0]}"
    vals = [cka[c].get(layer_key, 0) for c in comp_names]
    bars = ax.barh(comp_names, vals, color="steelblue")
    ax.set_xlabel("CKA Similarity")
    ax.set_title(f"Centered Kernel Alignment at {layer_key}")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("EVR Elbow + Full-Dimensional Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = figures_dir / "evr_elbow.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")

    # Save raw results
    results_path = Path("results/evr_elbow_result.json")
    results_path.write_text(json.dumps(result, indent=2))
    print(f"Saved raw results: {results_path}")

    print("\nEVR elbow analysis complete.")


if __name__ == "__main__":
    main()
