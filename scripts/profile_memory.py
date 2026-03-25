"""VRAM profiling: measure GPU memory usage at various configurations."""

from __future__ import annotations

import argparse

import modal

from substrate.config import TRINITY_MINI_ID

# Thresholds (in bytes) for the comfort recommendation.
# A100-80GB has 85,899,345,920 bytes; a comfortable fit is <80% utilisation.
_A100_80GB = 80 * 1024 ** 3
_COMFORTABLE_THRESHOLD = 0.80 * _A100_80GB
_TIGHT_THRESHOLD = 0.95 * _A100_80GB


def _bytes_to_gb(n: int | float) -> float:
    return n / (1024 ** 3)


def _recommendation(peak_bytes: float) -> str:
    if peak_bytes <= _COMFORTABLE_THRESHOLD:
        return "Fits comfortably"
    if peak_bytes <= _TIGHT_THRESHOLD:
        return "Tight — monitor closely"
    return "Consider fallback (Qwen 2.5-7B or smaller batch size)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile GPU VRAM usage at various sequence lengths via Modal."
    )
    parser.add_argument(
        "--model-id",
        default=TRINITY_MINI_ID,
        help=f"HuggingFace model ID to profile (default: {TRINITY_MINI_ID})",
    )
    args = parser.parse_args()

    print(f"Running profile_memory on Modal for model: {args.model_id}")
    print("Calling remote function (this may take several minutes on first run)...")

    profile_fn = modal.Function.lookup("keel-substrate", "profile_memory")
    result: dict = profile_fn.remote(args.model_id)

    # -------------------------------------------------------------------------
    # Build rows for the formatted table
    # -------------------------------------------------------------------------
    seq_measurements: dict = result.get("sequence_length_measurements", {})

    rows: list[tuple[str, float]] = [
        ("Model load (baseline)", _bytes_to_gb(result["baseline_after_load_bytes"])),
        ("Forward (no hooks)", _bytes_to_gb(result["forward_no_hooks_bytes"])),
        ("Forward (4 hooks)", _bytes_to_gb(result["forward_with_hooks_bytes"])),
    ]

    for seq_len in [512, 1024, 2048, 4096]:
        key = f"seq_{seq_len}"
        if key in seq_measurements:
            rows.append((f"Seq {seq_len} + hooks", _bytes_to_gb(seq_measurements[key])))

    # -------------------------------------------------------------------------
    # Print table
    # -------------------------------------------------------------------------
    print()
    print(f"Model: {result['model_id']}")
    print()

    col_stage = 30
    col_vram = 12
    header_stage = "Stage"
    header_vram = "VRAM (GB)"
    separator = f"| {'-' * col_stage} | {'-' * col_vram} |"

    print(f"| {header_stage:<{col_stage}} | {header_vram:>{col_vram}} |")
    print(separator)
    for stage, gb in rows:
        print(f"| {stage:<{col_stage}} | {gb:>{col_vram}.2f} |")

    # -------------------------------------------------------------------------
    # Print recommendation based on worst-case (seq 4096)
    # -------------------------------------------------------------------------
    worst_key = "seq_4096"
    worst_bytes = seq_measurements.get(worst_key, result["forward_with_hooks_bytes"])
    rec = _recommendation(worst_bytes)

    print()
    print(f"Recommendation: {rec}")
    print(
        f"  (Based on peak VRAM of {_bytes_to_gb(worst_bytes):.2f} GB at worst-case configuration)"
    )
    print()
    print("profile_memory complete.")


if __name__ == "__main__":
    main()
