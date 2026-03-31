#!/usr/bin/env python3
"""Sprint 2.5 Main Experiment: Run all 16 traces through Qwen 7B.

Uses TraceGeometry class — model loads once per trace, processes all points.
Container stays warm between traces (container_idle_timeout=120s).
"""

import json
from pathlib import Path

import modal

KEEL_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = KEEL_ROOT / "traces"
RESULTS_DIR = KEEL_ROOT / "results" / "traces"

LAYER_INDICES = [6, 13, 20, 27]

SESSIONS = ["thesis_geometry", "harness_thesis", "agentic_commerce", "kinnected"]

VARIANTS = {
    "coherent": "formatted.json",
    "d1_shuffled": "d1_shuffled.json",
    "d2_plausible": "d2_plausible.json",
    "d3_dilutory": "d3_dilutory.json",
}


def load_points(session: str, variant_file: str) -> list[str]:
    """Load accumulation points from a trace file, sorted by turn order."""
    path = TRACES_DIR / session / variant_file
    with open(path) as f:
        data = json.load(f)
    sorted_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    return [data[k] for k in sorted_keys]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    TraceGeometry = modal.Cls.from_name("keel-substrate", "TraceGeometry")
    geo = TraceGeometry()

    total = len(SESSIONS) * len(VARIANTS)
    done = 0

    for session in SESSIONS:
        print(f"\n{'='*50}")
        print(f"Session: {session}")
        print(f"{'='*50}")

        for variant_name, variant_file in VARIANTS.items():
            output_path = RESULTS_DIR / f"{session}_{variant_name}.json"

            if output_path.exists():
                print(f"  SKIP {variant_name} (already exists)")
                done += 1
                continue

            points = load_points(session, variant_file)
            print(f"  {variant_name}: {len(points)} points...")

            result = geo.process_trace.remote(
                accumulation_points=points,
                layer_indices=LAYER_INDICES,
            )

            result["session"] = session
            result["variant"] = variant_name

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved {output_path}")
            done += 1

    print(f"\nDone: {done}/{total} traces processed")


if __name__ == "__main__":
    main()
