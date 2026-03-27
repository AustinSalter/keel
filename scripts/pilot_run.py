"""Pilot validation: 3 payloads x 3 prompts x 1 completion on Modal with sweep layers.

Tests that payloads inject correctly, activation capture works with new layer
indices, and geometric differences are visible between null and contextual payloads.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal

from substrate.config import SWEEP_LAYERS, TRINITY_MINI_ID


def main() -> None:
    # Select pilot prompts: one from category 1 (AI/ML), 3 (food), 5 (control)
    prompts_data = json.loads(Path("data/prompts.json").read_text())["prompts"]
    pilot_prompts = [
        next(p for p in prompts_data if p["id"] == 1),   # AI/ML: Phello standalone case
        next(p for p in prompts_data if p["id"] == 14),   # Food: dinner party menu
        next(p for p in prompts_data if p["id"] == 25),   # Control: Krebs cycle
    ]

    # Select pilot payloads: null, P1, P_phello
    payloads_dir = Path("data/payloads")
    pilot_payloads = {
        "p_null": (payloads_dir / "p_null.txt").read_text(),
        "p1_questionnaire": (payloads_dir / "p1_questionnaire.txt").read_text(),
    }

    print(f"Pilot run: 3 payloads x 3 prompts on {TRINITY_MINI_ID}")
    print(f"Capture layers: {SWEEP_LAYERS}")
    print()

    # Deploy the pilot function
    pilot_fn = modal.Function.from_name("keel-substrate", "pilot_capture")

    results = {}
    for prompt in pilot_prompts:
        pid = f"prompt_{prompt['id']:02d}"
        # Load P_phello for this specific prompt
        phello_path = payloads_dir / "phello" / f"{pid}.txt"
        phello_text = phello_path.read_text() if phello_path.exists() else ""

        payloads_for_prompt = {
            **pilot_payloads,
            "p_phello": phello_text,
        }

        print(f"--- Prompt {prompt['id']} ({prompt['category']}, type {prompt['type']}) ---")
        print(f"  {prompt['text'][:80]}...")

        for payload_name, payload_text in payloads_for_prompt.items():
            result = pilot_fn.remote(
                model_id=TRINITY_MINI_ID,
                context=payload_text,
                prompt=prompt["text"],
                layer_indices=SWEEP_LAYERS,
            )
            results[(pid, payload_name)] = result
            gd_by_layer = {k: v["grassmann_distance"] for k, v in result["rotation"].items()}
            print(f"  {payload_name:<20} GD: {gd_by_layer}")

        print()

    # Summary table
    print("=" * 80)
    print("PILOT RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Prompt':<12} {'Payload':<20} ", end="")
    for layer_idx in SWEEP_LAYERS:
        print(f"{'L' + str(layer_idx) + ' GD':>10} ", end="")
    print(f"{'L' + str(SWEEP_LAYERS[0]) + ' Angle':>12}")
    print("-" * 76)

    for prompt in pilot_prompts:
        pid = f"prompt_{prompt['id']:02d}"
        for payload_name in ["p_null", "p1_questionnaire", "p_phello"]:
            key = (pid, payload_name)
            if key not in results:
                continue
            rot = results[key]["rotation"]
            print(f"  {pid:<10} {payload_name:<20} ", end="")
            for layer_idx in SWEEP_LAYERS:
                layer_key = f"layer_{layer_idx}"
                gd = rot.get(layer_key, {}).get("grassmann_distance", 0)
                print(f"{gd:>10.4f} ", end="")
            layer_key = f"layer_{SWEEP_LAYERS[0]}"
            mean_angle = rot.get(layer_key, {}).get("mean_angle", 0)
            print(f"{math.degrees(mean_angle):>11.2f}°")
        print()

    # Assessment
    print("ASSESSMENT:")
    for prompt in pilot_prompts:
        pid = f"prompt_{prompt['id']:02d}"
        null_key = (pid, "p_null")
        p1_key = (pid, "p1_questionnaire")
        if null_key in results and p1_key in results:
            null_gd = results[null_key]["rotation"].get(f"layer_{SWEEP_LAYERS[0]}", {}).get("grassmann_distance", 0)
            p1_gd = results[p1_key]["rotation"].get(f"layer_{SWEEP_LAYERS[0]}", {}).get("grassmann_distance", 0)
            diff = abs(p1_gd - null_gd)
            cat = prompt["category"]
            if cat == "generic_control":
                status = "PASS (control)" if diff < 0.5 else "INVESTIGATE (control shows signal)"
            else:
                status = "PASS (signal)" if diff > 0.1 else "INVESTIGATE (weak signal)"
            print(f"  {pid} ({cat}): null_GD={null_gd:.4f} p1_GD={p1_gd:.4f} diff={diff:.4f} → {status}")

    print("\nPilot complete.")


if __name__ == "__main__":
    main()
