"""Sprint 0 acceptance test: verify the capture + PCA pipeline works end-to-end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import modal

from substrate.config import TRINITY_MINI_ID


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the capture + PCA pipeline end-to-end via Modal."
    )
    parser.add_argument(
        "--model-id",
        default=TRINITY_MINI_ID,
        help=f"HuggingFace model ID to test (default: {TRINITY_MINI_ID})",
    )
    args = parser.parse_args()

    print(f"Running verify_pipeline on Modal for model: {args.model_id}")
    print("Calling remote function (this may take several minutes on first run)...")

    verify_fn = modal.Function.from_name("keel-substrate", "verify_pipeline")
    result: dict = verify_fn.remote(args.model_id)

    layers: dict = result["layers"]

    # -------------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------------
    print()
    print(f"Model: {result['model_id']}")
    print(f"{'Layer':<20} {'Explained Variance @ k=20':>30}")
    print("-" * 52)

    for layer_key, layer_info in layers.items():
        # Use the highest-k PCA result available
        best_key = None
        for try_k in [20, 10, 5]:
            candidate = f"explained_variance_ratio_k{try_k}"
            if candidate in layer_info:
                best_key = candidate
                break
        if best_key is None:
            print(f"  {layer_key:<18} (no PCA results)")
            continue
        evr = layer_info[best_key]
        k_val = best_key.split("k")[1]
        cumulative = sum(evr)
        print(f"  {layer_key:<18} k={k_val}: cumulative={cumulative:.4f}")

    print()

    # -------------------------------------------------------------------------
    # Save raw result JSON
    # -------------------------------------------------------------------------
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "verify_pipeline_result.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Raw results saved to: {json_path}")

    # -------------------------------------------------------------------------
    # Plot per-layer explained variance bar charts
    # -------------------------------------------------------------------------
    layer_keys = list(layers.keys())

    for layer_key, layer_info in layers.items():
        # Use highest-k available
        evr = None
        for try_k in [20, 10, 5]:
            candidate = f"explained_variance_ratio_k{try_k}"
            if candidate in layer_info:
                evr = layer_info[candidate]
                k_used = try_k
                break
        if evr is None:
            print(f"Skipping plot for {layer_key}: no PCA data.")
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        components = list(range(1, len(evr) + 1))
        ax.bar(components, evr, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(f"Explained Variance @ k={k_used}  |  {layer_key}  |  {result['model_id']}")
        ax.set_xticks(components)
        ax.set_xlim(0.5, len(evr_k20) + 0.5)

        out_path = figures_dir / f"explained_variance_{layer_key}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {out_path}")

    # -------------------------------------------------------------------------
    # Combined plot: all layers on one figure
    # -------------------------------------------------------------------------
    # Use best available k for combined plot
    layers_with_evr = []
    for lk, v in layers.items():
        for try_k in [20, 10, 5]:
            candidate = f"explained_variance_ratio_k{try_k}"
            if candidate in v:
                layers_with_evr.append((lk, v[candidate]))
                break

    if layers_with_evr:
        n = len(layers_with_evr)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, (layer_key, evr) in zip(axes, layers_with_evr):
            components = list(range(1, len(evr) + 1))
            ax.bar(components, evr, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained Variance Ratio")
            ax.set_title(layer_key)
            ax.set_xticks(components[::max(1, len(evr) // 10)])

        fig.suptitle(
            f"Explained Variance  |  {result['model_id']}",
            fontsize=12,
            y=1.02,
        )
        combined_path = figures_dir / "explained_variance_combined.png"
        fig.tight_layout()
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined figure: {combined_path}")

    print()
    print("verify_pipeline complete.")


if __name__ == "__main__":
    main()
