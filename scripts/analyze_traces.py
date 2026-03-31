#!/usr/bin/env python3
"""Sprint 2.5 Analysis: Trajectory comparison and visualization.

Reads all 16 trace results + pre-flight base rate.
Produces trajectory plots, layer comparison, statistical tests, and dual-outcome assessment.
"""

import json
from pathlib import Path

import numpy as np

KEEL_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = KEEL_ROOT / "results" / "traces"
PREFLIGHT_DIR = KEEL_ROOT / "results" / "preflight"
FIGURES_DIR = KEEL_ROOT / "results" / "figures"

SESSIONS = ["thesis_geometry", "harness_thesis", "agentic_commerce", "kinnected"]
VARIANTS = ["coherent", "d1_shuffled", "d2_plausible", "d3_dilutory"]
VARIANT_COLORS = {
    "coherent": "#2ecc71",
    "d1_shuffled": "#e74c3c",
    "d2_plausible": "#f39c12",
    "d3_dilutory": "#9b59b6",
}
VARIANT_LABELS = {
    "coherent": "Coherent",
    "d1_shuffled": "D1: Shuffled",
    "d2_plausible": "D2: Plausible",
    "d3_dilutory": "D3: Dilutory",
}
LAYER_INDICES = [6, 13, 20, 27]


def load_result(session: str, variant: str) -> dict | None:
    path = RESULTS_DIR / f"{session}_{variant}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_results() -> dict[str, dict[str, dict]]:
    """Load all 16 results into {session: {variant: result}}."""
    results = {}
    for session in SESSIONS:
        results[session] = {}
        for variant in VARIANTS:
            r = load_result(session, variant)
            if r:
                results[session][variant] = r
    return results


def trajectory_variance(cka_traj: list[float]) -> float:
    """Variance of a CKA trajectory — our primary separation metric."""
    if len(cka_traj) < 2:
        return 0.0
    return float(np.var(cka_traj))


METRICS = ["cka_trajectory", "cosine_trajectory", "grassmann_trajectory",
           "evr_trajectory", "curvature_trajectory"]
METRIC_LABELS = {
    "cka_trajectory": "CKA",
    "cosine_trajectory": "Cosine",
    "grassmann_trajectory": "Grassmann",
    "evr_trajectory": "EVR",
    "curvature_trajectory": "Curvature",
}


def compute_separation_stats(results: dict) -> dict:
    """Compute per-layer, per-session separation statistics for all metrics."""
    stats = {}
    for session in SESSIONS:
        stats[session] = {}
        for idx in LAYER_INDICES:
            layer_key = f"layer_{idx}"
            layer_stats = {}
            for variant in VARIANTS:
                if variant not in results[session]:
                    continue
                layer_data = results[session][variant]["layers"].get(layer_key, {})
                variant_stats = {"n_points": 0}
                for metric in METRICS:
                    traj = layer_data.get(metric, [])
                    if traj:
                        variant_stats[f"{metric}_mean"] = float(np.mean(traj))
                        variant_stats[f"{metric}_std"] = float(np.std(traj))
                        variant_stats[f"{metric}_variance"] = trajectory_variance(traj)
                        variant_stats["n_points"] = max(variant_stats["n_points"], len(traj))
                layer_stats[variant] = variant_stats
            stats[session][layer_key] = layer_stats
    return stats


def plot_metric_trajectories(results: dict, metric: str, ylabel: str, ylim=None):
    """Plot a metric's trajectories per session, colored by variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = METRIC_LABELS.get(metric, metric)

    for session in SESSIONS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{label} Trajectories — {session}", fontsize=14, fontweight="bold")

        for ax_idx, layer_idx in enumerate(LAYER_INDICES):
            ax = axes[ax_idx // 2][ax_idx % 2]
            layer_key = f"layer_{layer_idx}"

            for variant in VARIANTS:
                if variant in results[session]:
                    traj = results[session][variant]["layers"].get(layer_key, {}).get(metric, [])
                    if traj:
                        ax.plot(traj, color=VARIANT_COLORS[variant],
                                label=VARIANT_LABELS[variant], linewidth=2, alpha=0.8)

            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlabel("Turn transition")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)

        plt.tight_layout()
        path = FIGURES_DIR / f"{metric}_{session}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def plot_layer_comparison(stats: dict):
    """Heatmap of separation strength by layer × session × metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compare_metrics = ["cka_trajectory", "cosine_trajectory",
                       "grassmann_trajectory", "curvature_trajectory"]

    fig, axes = plt.subplots(1, len(compare_metrics), figsize=(5 * len(compare_metrics), 5))
    fig.suptitle("Separation Strength: Degraded/Coherent Variance Ratio", fontsize=14)

    for ax_idx, metric in enumerate(compare_metrics):
        ax = axes[ax_idx] if len(compare_metrics) > 1 else axes
        var_key = f"{metric}_variance"

        data = np.zeros((len(SESSIONS), len(LAYER_INDICES)))
        for i, session in enumerate(SESSIONS):
            for j, idx in enumerate(LAYER_INDICES):
                layer_key = f"layer_{idx}"
                layer_stats = stats[session].get(layer_key, {})
                coh_var = layer_stats.get("coherent", {}).get(var_key, 1e-10)
                deg_vars = []
                for v in ["d1_shuffled", "d2_plausible", "d3_dilutory"]:
                    dv = layer_stats.get(v, {}).get(var_key, 0)
                    if dv:
                        deg_vars.append(dv)
                avg_deg = np.mean(deg_vars) if deg_vars else 0
                data[i, j] = avg_deg / coh_var if coh_var > 0 else 0

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(LAYER_INDICES)))
        ax.set_xticklabels([f"L{idx}" for idx in LAYER_INDICES])
        ax.set_yticks(range(len(SESSIONS)))
        ax.set_yticklabels(SESSIONS if ax_idx == 0 else [])
        for i in range(len(SESSIONS)):
            for j in range(len(LAYER_INDICES)):
                ax.text(j, i, f"{data[i, j]:.1f}x", ha="center", va="center",
                        color="white" if data[i, j] > 2 else "black", fontsize=9)
        ax.set_title(METRIC_LABELS.get(metric, metric))

    plt.tight_layout()
    path = FIGURES_DIR / "layer_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_evr_evolution(results: dict):
    """EVR trajectory curves by variant across sessions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use L20 (deep processing layer) for EVR
    layer_key = "layer_20"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"EVR Evolution at {layer_key}", fontsize=14, fontweight="bold")

    for ax_idx, session in enumerate(SESSIONS):
        ax = axes[ax_idx // 2][ax_idx % 2]

        for variant in VARIANTS:
            if variant in results[session]:
                evr = results[session][variant]["layers"][layer_key]["evr_trajectory"]
                ax.plot(evr, color=VARIANT_COLORS[variant],
                        label=VARIANT_LABELS[variant], linewidth=2, alpha=0.8)

        ax.set_title(session)
        ax.set_xlabel("Accumulation point")
        ax.set_ylabel("EVR (top-k)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "evr_evolution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def write_analysis(results: dict, stats: dict):
    """Write the full analysis markdown."""
    lines = ["# Sprint 2.5 Trace Geometry Analysis\n"]

    # Per-metric separation summary
    compare_metrics = ["cka_trajectory", "cosine_trajectory",
                       "grassmann_trajectory", "curvature_trajectory"]

    lines.append("## Metric Comparison — Which Best Separates Coherent from Degraded?\n")

    best_metric_overall = {}
    best_layer_per_session = {}

    for metric in compare_metrics:
        var_key = f"{metric}_variance"
        label = METRIC_LABELS.get(metric, metric)
        lines.append(f"### {label}\n")
        lines.append(f"| Session | Layer | Coherent | D1 | D2 | D3 | Ratio |")
        lines.append(f"|---------|-------|----------|----|----|-------|-------|")

        for session in SESSIONS:
            for idx in LAYER_INDICES:
                layer_key = f"layer_{idx}"
                s = stats[session].get(layer_key, {})
                coh = s.get("coherent", {}).get(var_key, 0)
                d1 = s.get("d1_shuffled", {}).get(var_key, 0)
                d2 = s.get("d2_plausible", {}).get(var_key, 0)
                d3 = s.get("d3_dilutory", {}).get(var_key, 0)
                avg_deg = np.mean([d1, d2, d3]) if any([d1, d2, d3]) else 0
                ratio = avg_deg / coh if coh > 0 else 0

                lines.append(f"| {session} | L{idx} | {coh:.6f} | {d1:.6f} | "
                             f"{d2:.6f} | {d3:.6f} | {ratio:.1f}x |")

                key = (session, f"L{idx}")
                if key not in best_metric_overall or ratio > best_metric_overall[key][1]:
                    best_metric_overall[key] = (label, ratio)

        lines.append("")

    # Best metric per session/layer
    lines.append("### Winner by Session/Layer\n")
    lines.append("| Session | Layer | Best Metric | Ratio |")
    lines.append("|---------|-------|-------------|-------|")
    for session in SESSIONS:
        best_ratio = 0
        best_layer = ""
        for idx in LAYER_INDICES:
            key = (session, f"L{idx}")
            if key in best_metric_overall:
                metric_name, ratio = best_metric_overall[key]
                lines.append(f"| {session} | L{idx} | {metric_name} | {ratio:.1f}x |")
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_layer = f"L{idx}"
        best_layer_per_session[session] = (best_layer, best_ratio)
    lines.append("")

    # Dual-outcome assessment
    lines.append("\n## Dual-Outcome Assessment\n")

    # Check Path A: coherent trajectories smoother than degraded?
    path_a_evidence = []
    for session in SESSIONS:
        for idx in LAYER_INDICES:
            layer_key = f"layer_{idx}"
            s = stats[session].get(layer_key, {})
            coh_var = s.get("coherent", {}).get("cka_variance", 0)
            for dv in ["d1_shuffled", "d2_plausible", "d3_dilutory"]:
                deg_var = s.get(dv, {}).get("cka_variance", 0)
                if deg_var > coh_var * 1.5:
                    path_a_evidence.append(f"{session}/{layer_key}/{dv}")

    if len(path_a_evidence) > len(SESSIONS) * len(LAYER_INDICES):
        lines.append("### Path A: Thermometer Works\n")
        lines.append(f"**Evidence count**: {len(path_a_evidence)} layer/session/variant "
                     f"combinations show degraded variance > 1.5x coherent variance.\n")
        lines.append("Coherent traces produce measurably smoother geometric trajectories "
                     "than degraded traces. The thermometer principle is validated.\n")
    else:
        lines.append("### Path A: Thermometer Works\n")
        lines.append(f"**Evidence count**: {len(path_a_evidence)} — "
                     f"{'sufficient' if len(path_a_evidence) > 8 else 'insufficient'} "
                     f"for Path A conclusion.\n")

    # Best layer
    lines.append("\n## Best Measurement Layer\n")
    for session, (layer, ratio) in best_layer_per_session.items():
        lines.append(f"- **{session}**: {layer} ({ratio:.1f}x separation)")

    # Domain comparison
    lines.append("\n## Domain Comparison\n")
    layers_across_sessions = [best_layer_per_session[s][0] for s in SESSIONS]
    if len(set(layers_across_sessions)) == 1:
        lines.append(f"All sessions show best separation at **{layers_across_sessions[0]}** — "
                     f"coherence signal is domain-invariant at this layer.\n")
    else:
        lines.append(f"Optimal layer varies by session: {dict(zip(SESSIONS, layers_across_sessions))}. "
                     f"Coherence measurement may need to be layer-adaptive.\n")

    output_path = RESULTS_DIR / "analysis.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {output_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_all_results()

    # Check completeness
    total = sum(len(v) for v in results.values())
    print(f"Loaded {total}/16 trace results")
    if total < 16:
        missing = []
        for s in SESSIONS:
            for v in VARIANTS:
                if v not in results[s]:
                    missing.append(f"{s}/{v}")
        print(f"Missing: {missing}")

    print("\nComputing separation statistics...")
    stats = compute_separation_stats(results)

    print("\nGenerating plots...")
    plot_metric_trajectories(results, "cka_trajectory", "CKA", ylim=(-0.1, 1.1))
    plot_metric_trajectories(results, "cosine_trajectory", "Cosine Similarity", ylim=(-0.1, 1.1))
    plot_metric_trajectories(results, "grassmann_trajectory", "Grassmann Distance")
    plot_metric_trajectories(results, "curvature_trajectory", "Displacement Cosine (Curvature)", ylim=(-1.1, 1.1))
    plot_layer_comparison(stats)
    plot_evr_evolution(results)

    print("\nWriting analysis...")
    write_analysis(results, stats)

    print("\nDone.")


if __name__ == "__main__":
    main()
