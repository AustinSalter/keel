#!/usr/bin/env python3
"""Sprint 2.5 Pre-flight B: CKA base rate experiment.

Uses TraceGeometry class — model loads once, processes all points.
"""

import json
from pathlib import Path

import modal
import numpy as np

KEEL_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = KEEL_ROOT / "traces"
RESULTS_DIR = KEEL_ROOT / "results" / "preflight"

MODEL_ID = "Qwen/Qwen2.5-7B"
LAYER_INDICES = [6, 13, 20, 27]

BASELINE_SESSIONS = ["thesis_geometry", "harness_thesis"]


def load_accumulation_points(session: str) -> list[str]:
    path = TRACES_DIR / session / "formatted.json"
    with open(path) as f:
        data = json.load(f)
    sorted_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    return [data[k] for k in sorted_keys]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    TraceGeometry = modal.Cls.from_name("keel-substrate", "TraceGeometry")
    geo = TraceGeometry()

    session_results = {}

    for session in BASELINE_SESSIONS:
        points = load_accumulation_points(session)
        print(f"\n{session}: {len(points)} accumulation points")

        result = geo.process_trace.remote(
            accumulation_points=points,
            layer_indices=LAYER_INDICES,
        )
        session_results[session] = result

        # Save individual session result
        session_path = RESULTS_DIR / f"base_rate_{session}.json"
        with open(session_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {session_path}")

    # Summary
    summary = {
        "model_id": MODEL_ID,
        "sessions": BASELINE_SESSIONS,
        "per_layer": {},
    }
    for idx in LAYER_INDICES:
        layer_key = f"layer_{idx}"
        s1 = session_results[BASELINE_SESSIONS[0]]["layers"][layer_key]
        s2 = session_results[BASELINE_SESSIONS[1]]["layers"][layer_key]
        cka1 = np.array(s1["cka_trajectory"])
        cka2 = np.array(s2["cka_trajectory"])
        summary["per_layer"][layer_key] = {
            "session_1_cka_mean": float(cka1.mean()) if len(cka1) else 0,
            "session_1_cka_std": float(cka1.std()) if len(cka1) else 0,
            "session_2_cka_mean": float(cka2.mean()) if len(cka2) else 0,
            "session_2_cka_std": float(cka2.std()) if len(cka2) else 0,
            "session_1_evr_mean": float(np.mean(s1["evr_trajectory"])),
            "session_2_evr_mean": float(np.mean(s2["evr_trajectory"])),
        }

    output_path = RESULTS_DIR / "base_rate.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {output_path}")

    # Print summary
    print(f"\n{'Layer':<12} {'S1 CKA μ':>10} {'S1 CKA σ':>10} {'S2 CKA μ':>10} {'S2 CKA σ':>10}")
    print("-" * 55)
    for layer_key, stats in summary["per_layer"].items():
        print(f"{layer_key:<12} "
              f"{stats['session_1_cka_mean']:>10.4f} {stats['session_1_cka_std']:>10.4f} "
              f"{stats['session_2_cka_mean']:>10.4f} {stats['session_2_cka_std']:>10.4f}")


if __name__ == "__main__":
    main()
