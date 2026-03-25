"""Sanity check: verify code and philosophy prompts produce different activation geometries."""

from __future__ import annotations

import argparse
import math

import modal

from substrate.config import TRINITY_MINI_ID

# PASS criteria (Grassmann distance in radians)
_CROSS_PROMPT_MIN = 0.1   # code_vs_philosophy mean distance must exceed this
_SELF_COMPARE_MAX = 0.01  # code_vs_code distance must be below this


def _print_rotation_table(label: str, rotation_map: dict) -> None:
    """Print a formatted table of rotation summary values for all layers."""
    print(f"  {label}")
    col_layer = 18
    col_val = 16

    header = (
        f"  | {'Layer':<{col_layer}}"
        f" | {'Grassmann Dist':>{col_val}}"
        f" | {'Mean Angle (°)':>{col_val}}"
        f" | {'Max Angle (°)':>{col_val}} |"
    )
    sep = f"  | {'-' * col_layer} | {'-' * col_val} | {'-' * col_val} | {'-' * col_val} |"

    print(header)
    print(sep)

    for layer_key, summary in rotation_map.items():
        gd = summary["grassmann_distance"]
        mean_a = math.degrees(summary["mean_angle"])
        max_a = math.degrees(summary["max_angle"])
        print(
            f"  | {layer_key:<{col_layer}}"
            f" | {gd:>{col_val}.6f}"
            f" | {mean_a:>{col_val}.2f}"
            f" | {max_a:>{col_val}.2f} |"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check: verify code and philosophy prompts produce "
            "different activation geometries via Modal."
        )
    )
    parser.add_argument(
        "--model-id",
        default=TRINITY_MINI_ID,
        help=f"HuggingFace model ID to test (default: {TRINITY_MINI_ID})",
    )
    args = parser.parse_args()

    print(f"Running sanity_check on Modal for model: {args.model_id}")
    print("Calling remote function (this may take several minutes on first run)...")

    sanity_fn = modal.Function.lookup("keel-substrate", "sanity_check")
    result: dict = sanity_fn.remote(args.model_id)

    code_vs_philosophy: dict = result["code_vs_philosophy"]
    code_vs_code: dict = result["code_vs_code"]

    print()
    print(f"Model: {result['model_id']}")
    print()

    # -------------------------------------------------------------------------
    # Cross-prompt comparison: code vs philosophy
    # -------------------------------------------------------------------------
    print("Cross-prompt comparison (code vs philosophy):")
    _print_rotation_table("Should show large Grassmann distances", code_vs_philosophy)

    # -------------------------------------------------------------------------
    # Self-comparison control: code vs code (re-run)
    # -------------------------------------------------------------------------
    print("Self-comparison control (code vs code re-run):")
    _print_rotation_table("Should be ~0 across all layers", code_vs_code)

    # -------------------------------------------------------------------------
    # PASS / FAIL assessment
    # -------------------------------------------------------------------------
    cross_distances = [v["grassmann_distance"] for v in code_vs_philosophy.values()]
    self_distances = [v["grassmann_distance"] for v in code_vs_code.values()]

    mean_cross = sum(cross_distances) / len(cross_distances) if cross_distances else 0.0
    mean_self = sum(self_distances) / len(self_distances) if self_distances else 0.0

    layers_pass_cross = all(d > _CROSS_PROMPT_MIN for d in cross_distances)
    layers_pass_self = all(d < _SELF_COMPARE_MAX for d in self_distances)

    print("Assessment")
    print(f"  code_vs_philosophy  mean Grassmann distance: {mean_cross:.6f}"
          f"  (threshold > {_CROSS_PROMPT_MIN})")
    print(f"  code_vs_code        mean Grassmann distance: {mean_self:.6f}"
          f"  (threshold < {_SELF_COMPARE_MAX})")
    print()

    per_layer_pass = all(
        code_vs_philosophy[k]["grassmann_distance"] > code_vs_code[k]["grassmann_distance"]
        for k in code_vs_philosophy
        if k in code_vs_code
    )

    overall_pass = layers_pass_cross and layers_pass_self and per_layer_pass

    if overall_pass:
        print("RESULT: PASS")
        print(
            "  code_vs_philosophy Grassmann distance >> code_vs_code distance "
            "across all layers."
        )
    else:
        print("RESULT: FAIL")
        if not layers_pass_cross:
            print(
                f"  FAIL reason: code_vs_philosophy mean distance {mean_cross:.6f} "
                f"does not exceed {_CROSS_PROMPT_MIN}."
            )
        if not layers_pass_self:
            print(
                f"  FAIL reason: code_vs_code mean distance {mean_self:.6f} "
                f"exceeds {_SELF_COMPARE_MAX} (model is non-deterministic or hooks are leaking)."
            )
        if not per_layer_pass:
            print(
                "  FAIL reason: not all layers show cross_distance > self_distance."
            )

    print()
    print("sanity_check complete.")


if __name__ == "__main__":
    main()
