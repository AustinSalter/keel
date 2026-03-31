#!/usr/bin/env python3
"""Sprint 2.5 Pre-flight C: Qwen scaling ladder.

Run code-vs-philosophy sanity check on Qwen 2.5 at multiple sizes:
0.5B, 1.5B, 3B, 7B, 14B.

Produces a scaling curve showing at what model size the geometric signal emerges.
"""

import json
from pathlib import Path

import modal

KEEL_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = KEEL_ROOT / "results" / "preflight"

QWEN_MODELS = [
    {"size": "0.5B", "id": "Qwen/Qwen2.5-0.5B", "num_layers": 24},
    {"size": "1.5B", "id": "Qwen/Qwen2.5-1.5B", "num_layers": 28},
    {"size": "3B",   "id": "Qwen/Qwen2.5-3B",   "num_layers": 36},
    {"size": "7B",   "id": "Qwen/Qwen2.5-7B",   "num_layers": 28},
    {"size": "14B",  "id": "Qwen/Qwen2.5-14B",  "num_layers": 48},
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ladder_fn = modal.Function.from_name("keel-substrate", "scaling_ladder")

    all_results = {}

    for model in QWEN_MODELS:
        print(f"\n{'='*50}")
        print(f"Running {model['size']} ({model['id']})...")
        print(f"{'='*50}")

        result = ladder_fn.remote(
            model_id=model["id"],
            num_layers=model["num_layers"],
        )
        all_results[model["size"]] = result

    # Save combined results
    output_path = RESULTS_DIR / "scaling_ladder.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print scaling curve summary
    print(f"\n{'Size':<8} {'Layer':>12} {'Depth%':>8} {'Signal':>8} {'Noise':>8} {'SNR':>8} {'EVR':>6}")
    print("-" * 65)

    for model in QWEN_MODELS:
        size = model["size"]
        if size not in all_results:
            continue
        result = all_results[size]
        for layer_key in sorted(result["layers"].keys(),
                                key=lambda k: result["layers"][k]["layer_idx"]):
            layer = result["layers"][layer_key]
            print(f"{size:<8} {layer_key:>12} {layer['layer_depth_pct']:>7.1f}% "
                  f"{layer['grassmann_signal']:>8.4f} {layer['grassmann_noise']:>8.4f} "
                  f"{layer['snr']:>7.1f}x {layer['evr_code']:>6.3f}")
        print()


if __name__ == "__main__":
    main()
