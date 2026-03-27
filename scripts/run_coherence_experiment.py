"""Run the coherence experiment: SOUL × 10 prompts × 20 completions."""

import argparse
import json
from pathlib import Path

import modal

from substrate.config import SWEEP_LAYERS, TRINITY_MINI_ID


def main():
    parser = argparse.ArgumentParser(description="Run coherence correlation experiment on Modal.")
    parser.add_argument("--model-id", default=TRINITY_MINI_ID)
    parser.add_argument("--completions", type=int, default=20)
    parser.add_argument("--output-dir", default="results/coherence")
    args = parser.parse_args()

    # Load SOUL payload
    soul_text = Path("data/payloads/p_soul.txt").read_text()

    # Select 10 prompts: 8 personal (type A) + 2 control
    all_prompts = json.loads(Path("data/prompts.json").read_text())["prompts"]
    selected_ids = [1, 2, 7, 8, 13, 14, 19, 20, 25, 27]
    prompts = [
        {"id": p["id"], "text": p["text"], "category": p["category"]}
        for p in all_prompts if p["id"] in selected_ids
    ]
    assert len(prompts) == 10, f"Expected 10 prompts, got {len(prompts)}"

    print(f"Coherence experiment: {len(prompts)} prompts × {args.completions} completions")
    print(f"Model: {args.model_id}")
    print(f"Layers: {SWEEP_LAYERS}")
    print(f"SOUL: {len(soul_text)} chars")
    print()

    # Call Modal function per-prompt to avoid timeout
    # (200 generations in one call exceeded 3600s)
    fn = modal.Function.from_name("keel-substrate", "coherence_experiment")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "generations.jsonl"

    # Load existing results to support resume
    existing_prompt_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                existing_prompt_ids.add(r["prompt_id"])
        print(f"Resuming: {len(existing_prompt_ids)} prompts already done")

    remaining = [p for p in prompts if p["id"] not in existing_prompt_ids]
    all_results = []

    for i, prompt in enumerate(remaining):
        print(f"\nPrompt {prompt['id']} ({prompt['category']}) [{i+1}/{len(remaining)}]")
        print(f"  {prompt['text'][:60]}...")

        batch_results = fn.remote(
            model_id=args.model_id,
            layer_indices=SWEEP_LAYERS,
            soul_text=soul_text,
            prompts=[prompt],
            completions_per_prompt=args.completions,
        )

        # Append to JSONL immediately
        with open(jsonl_path, "a") as f:
            for r in batch_results:
                f.write(json.dumps(r) + "\n")
        all_results.extend(batch_results)
        print(f"  {len(batch_results)} completions saved")

    # Reload all results for summary
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    print(f"\nTotal: {len(results)} results in {jsonl_path}")

    # Print summary
    print(f"\n{'Prompt':>8} {'Category':<20} {'Mean CKA L24':>14} {'Std':>8} {'N':>4}")
    print("-" * 58)

    from collections import defaultdict
    by_prompt = defaultdict(list)
    for r in results:
        by_prompt[r["prompt_id"]].append(r)

    for pid in selected_ids:
        if pid not in by_prompt:
            continue
        entries = by_prompt[pid]
        layer_key = f"cka_layer_{SWEEP_LAYERS[2]}"  # L24
        cka_vals = [e[layer_key] for e in entries if layer_key in e]
        if cka_vals:
            import statistics
            mean_cka = statistics.mean(cka_vals)
            std_cka = statistics.stdev(cka_vals) if len(cka_vals) > 1 else 0
            cat = entries[0]["prompt_category"]
            print(f"  {pid:>6} {cat:<20} {mean_cka:>14.4f} {std_cka:>8.4f} {len(cka_vals):>4}")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
