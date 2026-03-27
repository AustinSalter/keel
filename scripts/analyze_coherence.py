"""Analyze coherence: Spearman correlation between L24 CKA and human ratings."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from substrate.config import SWEEP_LAYERS


def main():
    parser = argparse.ArgumentParser(description="Correlate CKA with human coherence ratings.")
    parser.add_argument("--generations", default="results/coherence/generations.jsonl")
    parser.add_argument("--ratings", default="results/coherence/ratings.csv")
    parser.add_argument("--output-dir", default="results/coherence")
    args = parser.parse_args()

    # Load generations (CKA values)
    generations = {}
    with open(args.generations) as f:
        for line in f:
            g = json.loads(line)
            key = (g["prompt_id"], g["completion_idx"])
            generations[key] = g

    # Load ratings
    ratings_by_key = {}
    with open(args.ratings) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["prompt_id"]), int(row["completion_idx"]))
            if key not in ratings_by_key:
                ratings_by_key[key] = []
            ratings_by_key[key].append({
                "coherence": int(row["coherence"]),
                "extension": int(row["extension"]),
                "drift": int(row["drift"]),
                "rater_id": row["rater_id"],
            })

    # Match CKA values with averaged human ratings
    l24_key = f"cka_layer_{SWEEP_LAYERS[2]}"
    matched = []
    for key, gen in generations.items():
        if key not in ratings_by_key:
            continue
        human_ratings = ratings_by_key[key]
        avg_coherence = np.mean([r["coherence"] for r in human_ratings])
        avg_extension = np.mean([r["extension"] for r in human_ratings])
        avg_drift = np.mean([r["drift"] for r in human_ratings])
        cka_l24 = gen.get(l24_key, None)
        if cka_l24 is None:
            continue
        matched.append({
            "prompt_id": gen["prompt_id"],
            "prompt_category": gen["prompt_category"],
            "completion_idx": gen["completion_idx"],
            "cka_l24": cka_l24,
            "coherence": avg_coherence,
            "extension": avg_extension,
            "drift": avg_drift,
            "n_raters": len(human_ratings),
        })

    if not matched:
        print("No matched data found. Check that generations and ratings files exist and overlap.")
        return

    print(f"Matched {len(matched)} completions with ratings")
    print()

    # ---- Primary analysis: Spearman correlation ----
    cka_vals = np.array([m["cka_l24"] for m in matched])
    coherence_vals = np.array([m["coherence"] for m in matched])
    extension_vals = np.array([m["extension"] for m in matched])
    drift_vals = np.array([m["drift"] for m in matched])

    rho_coherence, p_coherence = stats.spearmanr(cka_vals, coherence_vals)
    rho_extension, p_extension = stats.spearmanr(cka_vals, extension_vals)
    rho_drift, p_drift = stats.spearmanr(cka_vals, drift_vals)

    print("=" * 60)
    print("SPEARMAN CORRELATIONS: L24 CKA vs Human Ratings")
    print("=" * 60)
    print(f"  CKA vs Coherence:  rho={rho_coherence:+.4f}  p={p_coherence:.4e}")
    print(f"  CKA vs Extension:  rho={rho_extension:+.4f}  p={p_extension:.4e}")
    print(f"  CKA vs Drift:      rho={rho_drift:+.4f}  p={p_drift:.4e}")
    print()

    # ---- Decision ----
    if abs(rho_coherence) > 0.5 and p_coherence < 0.05:
        print("RESULT: S IS VALIDATED (rho > 0.5, p < 0.05)")
        print("  CKA tracks human coherence judgment. CPO reward signal is feasible.")
    elif abs(rho_coherence) > 0.3:
        print("RESULT: PARTIAL SIGNAL (0.3 < rho < 0.5)")
        print("  Signal exists but weak. Investigate per-layer and per-category.")
    else:
        print("RESULT: NO CORRELATION (rho < 0.3)")
        print("  CKA does not track coherence at this scale. Consider pivots.")
    print()

    # ---- Per-category breakdown ----
    print("Per-category breakdown:")
    categories = sorted(set(m["prompt_category"] for m in matched))
    for cat in categories:
        cat_data = [m for m in matched if m["prompt_category"] == cat]
        if len(cat_data) < 5:
            continue
        cat_cka = np.array([m["cka_l24"] for m in cat_data])
        cat_coh = np.array([m["coherence"] for m in cat_data])
        rho, p = stats.spearmanr(cat_cka, cat_coh)
        mean_cka = np.mean(cat_cka)
        mean_coh = np.mean(cat_coh)
        print(f"  {cat:<20} n={len(cat_data):>3}  "
              f"rho={rho:+.3f}  p={p:.3e}  "
              f"mean_CKA={mean_cka:.3f}  mean_coherence={mean_coh:.2f}")
    print()

    # ---- Inter-rater reliability (Krippendorff's alpha approximation) ----
    # Simple pairwise agreement for coherence
    rater_ids = set()
    for ratings_list in ratings_by_key.values():
        for r in ratings_list:
            rater_ids.add(r["rater_id"])
    print(f"Raters: {sorted(rater_ids)}")
    print(f"  (Krippendorff's alpha requires krippendorff package — compute separately if needed)")
    print()

    # ---- Plots ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Money plot: CKA vs Coherence scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Color by category
    cat_colors = {
        "ai_ml": "#e41a1c",
        "strategy": "#377eb8",
        "food_hospitality": "#4daf4a",
        "outdoor": "#ff7f00",
        "generic_control": "#984ea3",
    }

    # Plot 1: CKA vs Coherence
    ax = axes[0]
    for cat in categories:
        cat_data = [m for m in matched if m["prompt_category"] == cat]
        x = [m["cka_l24"] for m in cat_data]
        y = [m["coherence"] for m in cat_data]
        ax.scatter(x, y, alpha=0.6, label=cat, color=cat_colors.get(cat, "gray"), s=30)
    ax.set_xlabel("L24 CKA (context-only vs post-completion)")
    ax.set_ylabel("Human Coherence Rating (1-5)")
    ax.set_title(f"CKA vs Coherence (rho={rho_coherence:+.3f})")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Plot 2: CKA vs Extension
    ax = axes[1]
    for cat in categories:
        cat_data = [m for m in matched if m["prompt_category"] == cat]
        x = [m["cka_l24"] for m in cat_data]
        y = [m["extension"] for m in cat_data]
        ax.scatter(x, y, alpha=0.6, label=cat, color=cat_colors.get(cat, "gray"), s=30)
    ax.set_xlabel("L24 CKA")
    ax.set_ylabel("Substrate Extension (1-3)")
    ax.set_title(f"CKA vs Extension (rho={rho_extension:+.3f})")
    ax.grid(True, alpha=0.3)

    # Plot 3: CKA distribution by category
    ax = axes[2]
    cat_cka_lists = []
    cat_labels = []
    for cat in categories:
        cat_data = [m["cka_l24"] for m in matched if m["prompt_category"] == cat]
        if cat_data:
            cat_cka_lists.append(cat_data)
            cat_labels.append(cat)
    ax.boxplot(cat_cka_lists, labels=cat_labels, vert=True)
    ax.set_ylabel("L24 CKA")
    ax.set_title("CKA Distribution by Category")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Sprint 2 Track A: CKA vs Human Coherence", fontsize=13, y=1.02)
    fig.tight_layout()
    plot_path = output_dir / "coherence_correlation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation plot: {plot_path}")

    # Save matched data as JSON for further analysis
    matched_path = output_dir / "matched_data.json"
    matched_path.write_text(json.dumps(matched, indent=2))
    print(f"Saved matched data: {matched_path}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
