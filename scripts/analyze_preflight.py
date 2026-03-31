#!/usr/bin/env python3
"""Sprint 2.5 Pre-flight Analysis: Go/No-Go assessment.

Reads sanity check, base rate, and scaling ladder results.
Produces analysis report and visualizations.
"""

import json
from pathlib import Path

import numpy as np

KEEL_ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT_DIR = KEEL_ROOT / "results" / "preflight"
FIGURES_DIR = KEEL_ROOT / "results" / "figures"


def analyze_sanity(data: dict) -> tuple[bool, list[str]]:
    """Analyze sanity check results. Returns (go, findings)."""
    findings = []
    go = True

    for layer_key in sorted(data["code_vs_philosophy"].keys()):
        signal = data["code_vs_philosophy"][layer_key]["grassmann_distance"]
        noise = data["code_vs_code"][layer_key]["grassmann_distance"]
        snr = signal / noise if noise > 0 else float("inf")

        layer_idx = int(layer_key.split("_")[1])
        findings.append(f"- {layer_key}: SNR={snr:.1f}x (signal={signal:.4f}, noise={noise:.4f})")

        if layer_idx in (13, 20) and snr < 20:
            go = False
            findings.append(f"  **GATE FAILURE**: L{layer_idx} SNR={snr:.1f}x < 20x threshold")

    return go, findings


def analyze_base_rate(data: dict) -> list[str]:
    """Analyze base rate results."""
    findings = []
    for layer_key, stats in data.get("per_layer", {}).items():
        s1_mean = stats.get("session_1_cka_mean")
        s2_mean = stats.get("session_2_cka_mean")
        if s1_mean is not None and s2_mean is not None:
            avg = (s1_mean + s2_mean) / 2
            findings.append(f"- {layer_key}: S1 CKA μ={s1_mean:.4f}, S2 CKA μ={s2_mean:.4f}, "
                          f"avg={avg:.4f}")
            if avg > 0.7:
                findings.append(f"  → High baseline: CKA likely detects coherence structure")
            elif avg < 0.4:
                findings.append(f"  → Low baseline: focus on trajectory smoothness, not absolute level")
    return findings


def plot_scaling_curve(data: dict):
    """Plot SNR vs model size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = []
    size_labels = []
    snr_by_depth = {25: [], 50: [], 75: [], 100: []}

    for size_label in ["0.5B", "1.5B", "3B", "7B", "14B"]:
        if size_label not in data:
            continue

        result = data[size_label]
        # Parse numeric size for x-axis
        num = float(size_label.replace("B", ""))
        sizes.append(num)
        size_labels.append(size_label)

        for layer_key, layer_data in result["layers"].items():
            depth = layer_data["layer_depth_pct"]
            snr = layer_data["snr"]
            # Bucket by nearest depth quartile
            if depth <= 30:
                snr_by_depth[25].append((num, snr))
            elif depth <= 55:
                snr_by_depth[50].append((num, snr))
            elif depth <= 80:
                snr_by_depth[75].append((num, snr))
            else:
                snr_by_depth[100].append((num, snr))

    fig, ax = plt.subplots(figsize=(10, 6))
    depth_colors = {25: "#3498db", 50: "#2ecc71", 75: "#f39c12", 100: "#e74c3c"}
    depth_labels = {25: "25% depth", 50: "50% depth", 75: "75% depth", 100: "100% depth"}

    for depth, points in snr_by_depth.items():
        if points:
            x, y = zip(*sorted(points))
            ax.plot(x, y, 'o-', color=depth_colors[depth], label=depth_labels[depth],
                   linewidth=2, markersize=8)

    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="Go/No-Go threshold (20x)")
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Signal-to-Noise Ratio")
    ax.set_title("Qwen Scaling Ladder: Geometric Signal vs Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = FIGURES_DIR / "scaling_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["# Sprint 2.5 Pre-flight Analysis\n"]

    # Sanity check
    sanity_path = PREFLIGHT_DIR / "qwen_sanity.json"
    if sanity_path.exists():
        with open(sanity_path) as f:
            sanity_data = json.load(f)
        go, findings = analyze_sanity(sanity_data)
        lines.append("## A. Qwen 7B Sanity Check\n")
        lines.append(f"**Go/No-Go: {'GO' if go else 'NO-GO'}**\n")
        lines.extend(findings)
        lines.append("")
    else:
        lines.append("## A. Qwen 7B Sanity Check\n*Not yet run.*\n")

    # Base rate
    base_rate_path = PREFLIGHT_DIR / "base_rate.json"
    if base_rate_path.exists():
        with open(base_rate_path) as f:
            base_rate_data = json.load(f)
        findings = analyze_base_rate(base_rate_data)
        lines.append("## B. CKA Base Rate\n")
        lines.extend(findings)
        lines.append("")
    else:
        lines.append("## B. CKA Base Rate\n*Not yet run.*\n")

    # Scaling ladder
    ladder_path = PREFLIGHT_DIR / "scaling_ladder.json"
    if ladder_path.exists():
        with open(ladder_path) as f:
            ladder_data = json.load(f)
        plot_scaling_curve(ladder_data)
        lines.append("## C. Scaling Ladder\n")
        lines.append("![Scaling Curve](../figures/scaling_curve.png)\n")
        for size_label in ["0.5B", "1.5B", "3B", "7B", "14B"]:
            if size_label in ladder_data:
                result = ladder_data[size_label]
                snrs = [v["snr"] for v in result["layers"].values()]
                max_snr = max(snrs) if snrs else 0
                lines.append(f"- **{size_label}**: max SNR = {max_snr:.1f}x")
        lines.append("")
    else:
        lines.append("## C. Scaling Ladder\n*Not yet run.*\n")

    # Overall go/no-go
    lines.append("## Overall Assessment\n")
    sanity_go = sanity_path.exists()  # placeholder
    lines.append("*Assessment will be written after all pre-flight experiments complete.*\n")

    output_path = PREFLIGHT_DIR / "analysis.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
