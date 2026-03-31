#!/usr/bin/env python3
"""Sprint 2.5 Pre-flight A: Qwen 7B sanity check.

Replicates Sprint 0's code-vs-philosophy test on Qwen 2.5 7B.
Go/no-go gate: SNR >= 20x at L13 and L20.
"""

import json
from pathlib import Path

import modal

KEEL_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = KEEL_ROOT / "results" / "preflight"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sanity_fn = modal.Function.from_name("keel-substrate", "sanity_check")

    model_id = "Qwen/Qwen2.5-7B"
    print(f"Running sanity check on {model_id}...")
    result = sanity_fn.remote(model_id)

    # Save raw result
    output_path = RESULTS_DIR / "qwen_sanity.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_path}")

    # Print summary
    print(f"\n{'Layer':<12} {'Signal GD':>10} {'Noise GD':>10} {'SNR':>8} {'Status':>8}")
    print("-" * 55)

    go = True
    for layer_key in sorted(result["code_vs_philosophy"].keys()):
        signal = result["code_vs_philosophy"][layer_key]["grassmann_distance"]
        noise = result["code_vs_code"][layer_key]["grassmann_distance"]
        snr = signal / noise if noise > 0 else float("inf")

        # Go/no-go layers
        layer_idx = int(layer_key.split("_")[1])
        is_gate = layer_idx in (13, 20)
        status = ""
        if is_gate:
            if snr >= 20:
                status = "PASS"
            else:
                status = "FAIL"
                go = False

        print(f"{layer_key:<12} {signal:>10.4f} {noise:>10.4f} {snr:>8.1f}x {status:>8}")

    print(f"\nGo/No-Go: {'GO ✓' if go else 'NO-GO ✗ — investigate before proceeding'}")


if __name__ == "__main__":
    main()
