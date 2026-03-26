"""Layer sweep: find optimal capture layers for substrate geometry.

Runs all 32 layers with subtle prompt pairs to identify where
signal concentrates, noise is lowest, and variance is most structured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import modal

from substrate.config import TRINITY_MINI_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Full layer sweep for layer selection.")
    parser.add_argument(
        "--model-id", default=TRINITY_MINI_ID,
        help=f"HuggingFace model ID (default: {TRINITY_MINI_ID})",
    )
    parser.add_argument(
        "--from-cache", type=str, default=None,
        help="Load results from a cached JSON file instead of running on Modal",
    )
    args = parser.parse_args()

    if args.from_cache:
        print(f"Loading cached results from {args.from_cache}")
        result = json.loads(Path(args.from_cache).read_text())
    else:
        print(f"Running layer_sweep on Modal for model: {args.model_id}")
        print("This captures at ALL layers — may take 5-10 minutes...")
        sweep_fn = modal.Function.from_name("keel-substrate", "layer_sweep")
        result = sweep_fn.remote(args.model_id)

    num_layers = result["num_layers"]
    layers = result["layers"]

    # Save raw results
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "layer_sweep_result.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Raw results saved to: {json_path}")

    # --- Extract per-layer data ---
    idxs = list(range(num_layers))
    framing_gd = []
    context_gd = []
    self_gd = []
    evr_k10 = []
    evr_top5 = []
    framing_snr = []
    context_snr = []

    for idx in idxs:
        layer = layers[f"layer_{idx}"]
        f_gd = layer["framing"]["grassmann_distance"]
        c_gd = layer["context"]["grassmann_distance"]
        s_gd = layer["self_comparison"]["grassmann_distance"]
        framing_gd.append(f_gd)
        context_gd.append(c_gd)
        self_gd.append(s_gd)
        evr_k10.append(layer["explained_variance"]["risk_prompt_cumulative_k10"])
        evr_top5.append(layer["explained_variance"]["risk_prompt_top5_ratio"])
        framing_snr.append(f_gd / max(s_gd, 1e-6))
        context_snr.append(c_gd / max(s_gd, 1e-6))

    idxs_arr = np.array(idxs)

    # --- Print summary table ---
    print()
    print(f"Model: {result['model_id']}  |  {num_layers} layers  |  k={result['pca_k']}")
    print()
    print(f"{'Layer':>7} {'Framing GD':>11} {'Context GD':>11} {'Self GD':>9} "
          f"{'Fr SNR':>8} {'Ctx SNR':>8} {'EVR k10':>8} {'EVR top5':>9}")
    print("-" * 83)

    for idx in idxs:
        layer = layers[f"layer_{idx}"]
        f = layer["framing"]["grassmann_distance"]
        c = layer["context"]["grassmann_distance"]
        s = layer["self_comparison"]["grassmann_distance"]
        e10 = layer["explained_variance"]["risk_prompt_cumulative_k10"]
        e5 = layer["explained_variance"]["risk_prompt_top5_ratio"]
        f_snr = f / max(s, 1e-6)
        c_snr = c / max(s, 1e-6)
        print(f"  {idx:>5} {f:>11.4f} {c:>11.4f} {s:>9.4f} "
              f"{f_snr:>7.1f}x {c_snr:>7.1f}x {e10:>8.4f} {e5:>9.4f}")

    # --- Find optimal layers ---
    print()
    print("=== OPTIMAL LAYER ANALYSIS ===")
    print()

    # Composite score: high framing SNR + high context SNR + high EVR concentration
    framing_snr_arr = np.array(framing_snr)
    context_snr_arr = np.array(context_snr)
    evr_arr = np.array(evr_k10)

    # Normalize each to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    composite = (
        normalize(framing_snr_arr)
        + normalize(context_snr_arr)
        + normalize(evr_arr)
    ) / 3.0

    top_5 = np.argsort(composite)[::-1][:5]
    print("Top 5 layers by composite score (SNR + EVR):")
    for rank, idx in enumerate(top_5):
        layer = layers[f"layer_{idx}"]
        print(f"  #{rank + 1}: layer {idx}  "
              f"(framing SNR={framing_snr[idx]:.1f}x, "
              f"context SNR={context_snr[idx]:.1f}x, "
              f"EVR@k10={evr_k10[idx]:.3f}, "
              f"composite={composite[idx]:.3f})")

    best = top_5[0]
    print()
    print(f"Recommended primary capture layer: {best}")
    print(f"  Framing Grassmann distance: {framing_gd[best]:.4f}")
    print(f"  Context Grassmann distance: {context_gd[best]:.4f}")
    print(f"  Self-comparison noise:      {self_gd[best]:.4f}")
    print(f"  Explained variance @ k=10:  {evr_k10[best]:.4f}")

    # --- Plots ---

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Grassmann distance by layer
    ax = axes[0, 0]
    ax.plot(idxs_arr, framing_gd, "o-", label="Framing (risk vs benefit)", markersize=3)
    ax.plot(idxs_arr, context_gd, "s-", label="Context (cold vs warm)", markersize=3)
    ax.plot(idxs_arr, self_gd, "^-", label="Self-comparison (noise)", markersize=3, alpha=0.7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Grassmann Distance")
    ax.set_title("Signal vs Noise by Layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: SNR by layer
    ax = axes[0, 1]
    ax.plot(idxs_arr, framing_snr, "o-", label="Framing SNR", markersize=3)
    ax.plot(idxs_arr, context_snr, "s-", label="Context SNR", markersize=3)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Signal / Noise Ratio")
    ax.set_title("Signal-to-Noise Ratio by Layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Explained variance concentration
    ax = axes[1, 0]
    ax.plot(idxs_arr, evr_k10, "o-", label="Top-10 cumulative", markersize=3)
    ax.plot(idxs_arr, evr_top5, "s-", label="Top-5 cumulative", markersize=3)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("Variance Concentration by Layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Plot 4: Composite score
    ax = axes[1, 1]
    colors = ["#d62728" if idx in top_5 else "steelblue" for idx in idxs]
    ax.bar(idxs_arr, composite, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Composite Score")
    ax.set_title("Composite Score (SNR + EVR) — Red = Top 5")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Layer Sweep — {result['model_id']}", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = figures_dir / "layer_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved layer sweep plot: {out_path}")

    # --- Explained variance heatmap across layers ---
    fig, ax = plt.subplots(figsize=(14, 5))
    evr_matrix = []
    for idx in idxs:
        evr_matrix.append(layers[f"layer_{idx}"]["explained_variance"]["risk_prompt_evr"])
    evr_matrix = np.array(evr_matrix)  # [num_layers, k]

    im = ax.imshow(evr_matrix.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Principal Component")
    ax.set_title(f"Explained Variance Ratio by Layer and Component — {result['model_id']}")
    plt.colorbar(im, ax=ax, label="Explained Variance Ratio")
    fig.tight_layout()
    heatmap_path = figures_dir / "layer_sweep_evr_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved EVR heatmap: {heatmap_path}")

    print("\nLayer sweep complete.")


if __name__ == "__main__":
    main()
