"""Human evaluation interface for coherence experiment completions."""

import argparse
import csv
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rate completions for coherence.")
    parser.add_argument("--rater-id", required=True, help="Your rater identifier (e.g., 'austin')")
    parser.add_argument("--generations", default="results/coherence/generations.jsonl")
    parser.add_argument("--output", default="results/coherence/ratings.csv")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for blind ordering")
    args = parser.parse_args()

    # Load generations
    generations = []
    with open(args.generations) as f:
        for line in f:
            generations.append(json.loads(line))

    # Load prompts for display
    prompts_by_id = {
        p["id"]: p for p in json.loads(Path("data/prompts.json").read_text())["prompts"]
    }

    # Load SOUL for display
    soul_text = Path("data/payloads/p_soul.txt").read_text()

    # Load existing ratings to support resume
    rated_keys = set()
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["rater_id"] == args.rater_id:
                    rated_keys.add((int(row["prompt_id"]), int(row["completion_idx"])))

    # Shuffle for blind ordering
    rng = random.Random(args.seed)
    order = list(range(len(generations)))
    rng.shuffle(order)

    # Filter out already-rated
    remaining = [
        i for i in order
        if (generations[i]["prompt_id"], generations[i]["completion_idx"]) not in rated_keys
    ]

    total = len(generations)
    done = total - len(remaining)
    print(f"Rater: {args.rater_id}")
    print(f"Total completions: {total}, already rated: {done}, remaining: {remaining}")
    print(f"Context: SOUL document ({len(soul_text)} chars)")
    print()
    print("Rating scale:")
    print("  Coherence (1-5): 1=incoherent, 5=perfectly germane to the context frame")
    print("  Extension (1-3): 1=trivial restate, 2=stays in frame, 3=adds new dimensions")
    print("  Drift (y/n): does the response depart from the structural frame?")
    print()
    input("Press Enter to begin...")

    # Open CSV for appending
    write_header = not output_path.exists()
    with open(output_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "prompt_id", "completion_idx", "coherence", "extension", "drift", "rater_id"
        ])
        if write_header:
            writer.writeheader()

        for count, idx in enumerate(remaining, start=done + 1):
            gen = generations[idx]
            prompt = prompts_by_id.get(gen["prompt_id"], {})

            print("\n" + "=" * 70)
            print(f"  Rating {count}/{total}")
            print("=" * 70)
            print()
            print("CONTEXT (SOUL):")
            print(f"  {soul_text[:200]}...")
            print()
            print(f"PROMPT: {prompt.get('text', 'Unknown')}")
            print()
            print("COMPLETION:")
            print("-" * 40)
            print(gen["completion_text"])
            print("-" * 40)
            print()

            # Collect ratings
            while True:
                try:
                    coherence = int(input("  Coherence (1-5): "))
                    if 1 <= coherence <= 5:
                        break
                    print("  Must be 1-5")
                except (ValueError, EOFError):
                    print("  Must be 1-5")

            while True:
                try:
                    extension = int(input("  Extension (1-3): "))
                    if 1 <= extension <= 3:
                        break
                    print("  Must be 1-3")
                except (ValueError, EOFError):
                    print("  Must be 1-3")

            while True:
                drift_input = input("  Drift? (y/n): ").strip().lower()
                if drift_input in ("y", "n", "yes", "no"):
                    drift = 1 if drift_input.startswith("y") else 0
                    break
                print("  Must be y or n")

            writer.writerow({
                "prompt_id": gen["prompt_id"],
                "completion_idx": gen["completion_idx"],
                "coherence": coherence,
                "extension": extension,
                "drift": drift,
                "rater_id": args.rater_id,
            })
            csvfile.flush()  # Flush after each rating so progress isn't lost

            print(f"  Saved. ({count}/{total})")

    print(f"\nAll ratings complete! Saved to {output_path}")


if __name__ == "__main__":
    main()
