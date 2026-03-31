#!/usr/bin/env python3
"""Sprint 2.5 Cone Violation Analysis.

For each turn N, compute the expected displacement direction as the mean of
displacement vectors from turns N-3 through N-1. Then compute the cosine angle
between the actual displacement at turn N and this expected direction.

A "cone violation" is when the actual displacement deviates significantly from
the expected direction — the argument changed course.

Usage:
    # Step 1: Rerun thesis_geometry coherent + d2 with displacement vectors
    python scripts/cone_violation.py --rerun

    # Step 2: Analyze (uses saved results with displacement vectors)
    python scripts/cone_violation.py --analyze

    # Step 3: Just plot from existing data
    python scripts/cone_violation.py --plot
"""

import json
import sys
from pathlib import Path

import numpy as np

KEEL_ROOT = Path(__file__).resolve().parent.parent
RESULTS = KEEL_ROOT / "results" / "traces"
FIGURES = KEEL_ROOT / "results" / "figures"
TRACES = KEEL_ROOT / "traces"

LAYER_INDICES = [6, 13, 20, 27]


def load_d2_target(session: str) -> dict:
    path = TRACES / session / "d2_target.json"
    with open(path) as f:
        return json.load(f)


def compute_cone_violations(displacement_vectors: list, lookback: int = 3) -> list:
    """Compute per-turn cone violation angles.

    For each turn N, the expected direction is the mean of displacement vectors
    from turns max(0, N-lookback) through N-1. The cone violation is the cosine
    angle between the actual displacement and this expected direction.

    Returns list of (turn_index, cosine_angle) — high cosine = consistent,
    low cosine = violation.
    """
    violations = []

    for i in range(1, len(displacement_vectors)):
        actual = displacement_vectors[i]
        if actual is None:
            violations.append((i, float("nan")))
            continue

        actual = np.array(actual)

        # Collect lookback vectors
        lookback_vecs = []
        for j in range(max(0, i - lookback), i):
            if displacement_vectors[j] is not None:
                lookback_vecs.append(np.array(displacement_vectors[j]))

        if not lookback_vecs:
            violations.append((i, float("nan")))
            continue

        # Expected direction = mean of lookback vectors
        # Pad to same length
        max_k = max(len(v) for v in lookback_vecs + [actual])
        padded = []
        for v in lookback_vecs:
            if len(v) < max_k:
                v = np.pad(v, (0, max_k - len(v)))
            padded.append(v)

        expected = np.mean(padded, axis=0)

        # Pad actual if needed
        if len(actual) < max_k:
            actual = np.pad(actual, (0, max_k - len(actual)))

        # Cosine angle
        dot = np.dot(actual[:max_k], expected[:max_k])
        norm = np.linalg.norm(actual[:max_k]) * np.linalg.norm(expected[:max_k])
        cosine = float(dot / norm) if norm > 0 else 0.0

        violations.append((i, cosine))

    return violations


def rerun_with_displacement_vectors():
    """Rerun thesis_geometry coherent + d2 with displacement vector storage."""
    import modal

    TraceGeometry = modal.Cls.from_name("keel-substrate", "TraceGeometry")
    geo = TraceGeometry()

    for variant in ["coherent", "d2_plausible"]:
        path = TRACES / "thesis_geometry" / (
            "formatted.json" if variant == "coherent" else "d2_plausible.json"
        )
        with open(path) as f:
            data = json.load(f)
        sorted_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        points = [data[k] for k in sorted_keys]

        save_name = f"thesis_geometry_{variant}"
        print(f"Processing {save_name}: {len(points)} points (saving activations)...")

        result = geo.process_trace.remote(
            accumulation_points=points,
            layer_indices=LAYER_INDICES,
            save_activations=save_name,
        )

        result["session"] = "thesis_geometry"
        result["variant"] = variant

        # Save results (now includes displacement_vectors)
        output = RESULTS / f"thesis_geometry_{variant}.json"
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {output}")
        print(f"  Activations saved to Modal volume: {result.get('activations_saved')}")

        # Download activations locally
        act_data = geo.download_activations.remote(save_name)
        local_path = RESULTS / f"thesis_geometry_{variant}_activations.npz"
        with open(local_path, "wb") as f:
            f.write(act_data)
        print(f"  Downloaded activations: {local_path} ({len(act_data) / 1e6:.1f} MB)")


def analyze():
    """Compute cone violations from saved displacement vectors."""
    sessions_to_analyze = [
        ("thesis_geometry", ["coherent", "d2_plausible"]),
    ]

    for session, variants in sessions_to_analyze:
        d2_target = load_d2_target(session)
        sub_turn = d2_target["turn"]
        sub_iter = d2_target["iteration"]

        print(f"\n{'='*60}")
        print(f"{session} — D2 substitution at turn {sub_turn} (iteration {sub_iter})")
        print(f"{'='*60}")

        for layer_idx in LAYER_INDICES:
            layer_key = f"layer_{layer_idx}"
            print(f"\n  {layer_key}:")

            variant_violations = {}
            for variant in variants:
                path = RESULTS / f"{session}_{variant}.json"
                with open(path) as f:
                    data = json.load(f)

                dvs = data["layers"][layer_key].get("displacement_vectors")
                if not dvs:
                    print(f"    {variant}: no displacement vectors (needs rerun)")
                    continue

                violations = compute_cone_violations(dvs)
                variant_violations[variant] = violations

                print(f"    {variant}:")
                for turn_idx, cosine in violations:
                    marker = ""
                    if variant == "d2_plausible":
                        # Find which measurement point corresponds to the substituted turn
                        # The D2 trace has the same accumulation points as coherent
                        # except one turn is replaced
                        if abs(cosine) < 0.5:
                            marker = " ← LOW"
                    print(f"      turn {turn_idx:>2}: cosine={cosine:>7.4f}{marker}")

            if len(variant_violations) == 2:
                coh = [c for _, c in variant_violations["coherent"] if not np.isnan(c)]
                d2 = [c for _, c in variant_violations["d2_plausible"] if not np.isnan(c)]
                print(f"\n    Summary:")
                print(f"      Coherent mean cosine: {np.mean(coh):.4f}")
                print(f"      D2 mean cosine:       {np.mean(d2):.4f}")

                # Find max deviation point
                diffs = []
                for (t1, c1), (t2, c2) in zip(
                    variant_violations["coherent"], variant_violations["d2_plausible"]
                ):
                    if not np.isnan(c1) and not np.isnan(c2):
                        diffs.append((t1, c1 - c2))
                if diffs:
                    max_diff_turn, max_diff = max(diffs, key=lambda x: abs(x[1]))
                    print(f"      Max divergence at turn {max_diff_turn}: "
                          f"Δcosine = {max_diff:+.4f}")


def plot():
    """Plot coherent vs D2 cone violations at L20 and L27."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)

    session = "thesis_geometry"
    d2_target = load_d2_target(session)
    sub_turn = d2_target["turn"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cone Violation Analysis — {session}", fontsize=14, fontweight="bold")

    for ax_idx, layer_idx in enumerate([20, 27]):
        ax = axes[ax_idx]
        layer_key = f"layer_{layer_idx}"

        for variant, color, label in [
            ("coherent", "#2ecc71", "Coherent"),
            ("d2_plausible", "#f39c12", "D2: Plausible sub"),
        ]:
            path = RESULTS / f"{session}_{variant}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)

            dvs = data["layers"][layer_key].get("displacement_vectors")
            if not dvs:
                continue

            violations = compute_cone_violations(dvs)
            turns = [t for t, _ in violations]
            cosines = [c for _, c in violations]

            ax.plot(turns, cosines, 'o-', color=color, label=label,
                    linewidth=2, markersize=5, alpha=0.8)

        # Mark substitution point
        # Find which accumulation point index corresponds to the substituted turn
        ax.axvline(x=2, color="red", linestyle="--", alpha=0.4,
                   label=f"D2 sub (iter {d2_target['iteration']} critique)")

        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Measurement point")
        ax.set_ylabel("Cosine with expected direction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    path = FIGURES / "cone_violation_thesis_geometry.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def main():
    if "--rerun" in sys.argv:
        rerun_with_displacement_vectors()
    if "--analyze" in sys.argv or "--rerun" in sys.argv:
        analyze()
    if "--plot" in sys.argv or "--rerun" in sys.argv:
        plot()
    if not any(f in sys.argv for f in ["--rerun", "--analyze", "--plot"]):
        print("Usage:")
        print("  python scripts/cone_violation.py --rerun    # rerun traces + analyze + plot")
        print("  python scripts/cone_violation.py --analyze  # analyze from saved data")
        print("  python scripts/cone_violation.py --plot     # plot only")


if __name__ == "__main__":
    main()
