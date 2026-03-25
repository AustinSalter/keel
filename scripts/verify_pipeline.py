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

    verify_fn = modal.Function.lookup("keel-substrate", "verify_pipeline")
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
        evr_k20: list[float] | None = layer_info.get("explained_variance_ratio_k20")
        if evr_k20 is None:
            print(f"  {layer_key:<18} (k=20 not present in results)")
            continue
        cumulative = sum(evr_k20)
        print(f"  {layer_key:<18} cumulative={cumulative:.4f}  (sum of top-20 ratios)")

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
        evr_k20: list[float] | None = layer_info.get("explained_variance_ratio_k20")
        if evr_k20 is None:
            print(f"Skipping plot for {layer_key}: no k=20 data.")
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        components = list(range(1, len(evr_k20) + 1))
        ax.bar(components, evr_k20, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(f"Explained Variance @ k=20  |  {layer_key}  |  {result['model_id']}")
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
    layers_with_k20 = [
        (k, v["explained_variance_ratio_k20"])
        for k, v in layers.items()
        if "explained_variance_ratio_k20" in v
    ]

    if layers_with_k20:
        n = len(layers_with_k20)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, (layer_key, evr_k20) in zip(axes, layers_with_k20):
            components = list(range(1, len(evr_k20) + 1))
            ax.bar(components, evr_k20, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained Variance Ratio")
            ax.set_title(layer_key)
            ax.set_xticks(components[::max(1, len(evr_k20) // 10)])

        fig.suptitle(
            f"Explained Variance @ k=20  |  {result['model_id']}",
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
