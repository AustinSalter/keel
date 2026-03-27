"""Recompute CKA using context+prompt as reference instead of context-only.

Reads existing completions from generations.jsonl, sends them to Modal
for CKA recomputation, then correlates with human ratings.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import modal
from scipy import stats

from substrate.config import SWEEP_LAYERS, TRINITY_MINI_ID


def main():
    soul_text = Path("data/payloads/p_soul.txt").read_text()
    prompts_by_id = {
        p["id"]: p for p in json.loads(Path("data/prompts.json").read_text())["prompts"]
    }

    # Load existing completions
    completions = []
    with open("results/coherence/generations.jsonl") as f:
        for line in f:
            g = json.loads(line)
            completions.append({
                "prompt_id": g["prompt_id"],
                "prompt_text": prompts_by_id[g["prompt_id"]]["text"],
                "completion_idx": g["completion_idx"],
                "completion_text": g["completion_text"],
            })

    print(f"Recomputing CKA for {len(completions)} completions with context+prompt reference")
    print(f"Layers: {SWEEP_LAYERS}")

    # Split into batches of 20 to avoid timeout
    fn = modal.Function.from_name("keel-substrate", "recompute_cka_from_prompt_ref")
    batch_size = 20
    all_results = []

    for start in range(0, len(completions), batch_size):
        batch = completions[start:start + batch_size]
        print(f"\nBatch {start // batch_size + 1}/{(len(completions) + batch_size - 1) // batch_size} "
              f"({len(batch)} completions)...")
        batch_results = fn.remote(
            model_id=TRINITY_MINI_ID,
            layer_indices=SWEEP_LAYERS,
            soul_text=soul_text,
            completions=batch,
        )
        all_results.extend(batch_results)
        print(f"  Got {len(batch_results)} results")

    # Merge with existing data
    generations = []
    with open("results/coherence/generations.jsonl") as f:
        for line in f:
            generations.append(json.loads(line))

    # Index recomputed CKA by (prompt_id, completion_idx)
    recomputed = {}
    for r in all_results:
        key = (r["prompt_id"], r["completion_idx"])
        recomputed[key] = r

    # Load ratings
    import csv
    ratings_by_key = {}
    with open("results/coherence/ratings.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["prompt_id"]), int(row["completion_idx"]))
            ratings_by_key[key] = {
                "coherence": int(row["coherence"]),
                "extension": int(row["extension"]),
                "drift": int(row["drift"]),
            }

    # Build matched dataset
    matched = []
    for g in generations:
        key = (g["prompt_id"], g["completion_idx"])
        if key in recomputed and key in ratings_by_key:
            entry = {
                "prompt_id": g["prompt_id"],
                "prompt_category": g["prompt_category"],
                "completion_idx": g["completion_idx"],
                "coherence": ratings_by_key[key]["coherence"],
                "extension": ratings_by_key[key]["extension"],
                "drift": ratings_by_key[key]["drift"],
                "cka_context_ref_l24": g.get("cka_layer_24", 0),  # Old metric
            }
            for idx in SWEEP_LAYERS:
                lk = f"layer_{idx}"
                entry[f"cka_prompt_ref_{lk}"] = recomputed[key].get(f"cka_prompt_ref_{lk}", 0)
            matched.append(entry)

    print(f"\n{'=' * 70}")
    print(f"CORRELATION: Context+Prompt Reference CKA vs Human Coherence")
    print(f"{'=' * 70}")
    print(f"Matched: {len(matched)} completions\n")

    for idx in SWEEP_LAYERS:
        lk = f"layer_{idx}"
        cka_vals = [m[f"cka_prompt_ref_{lk}"] for m in matched]
        coh_vals = [m["coherence"] for m in matched]
        rho, p = stats.spearmanr(cka_vals, coh_vals)
        print(f"  L{idx} CKA(ctx+prompt → ctx+prompt+completion) vs Coherence:")
        print(f"    rho={rho:+.4f}  p={p:.4e}")

    # Compare old vs new metric
    print(f"\n  Old metric (context-only ref, L24):")
    old_cka = [m["cka_context_ref_l24"] for m in matched]
    old_coh = [m["coherence"] for m in matched]
    rho_old, p_old = stats.spearmanr(old_cka, old_coh)
    print(f"    rho={rho_old:+.4f}  p={p_old:.4e}")

    # Per-category for best layer
    print(f"\n  Per-category breakdown (prompt-ref CKA):")
    by_cat = defaultdict(list)
    for m in matched:
        by_cat[m["prompt_category"]].append(m)

    for layer_idx in [11, 24]:
        lk = f"layer_{layer_idx}"
        print(f"\n  Layer {layer_idx}:")
        for cat in sorted(by_cat):
            entries = by_cat[cat]
            cka = [e[f"cka_prompt_ref_{lk}"] for e in entries]
            coh = [e["coherence"] for e in entries]
            if len(entries) >= 5:
                rho, p = stats.spearmanr(cka, coh)
                print(f"    {cat:<20} rho={rho:+.3f}  p={p:.3e}  "
                      f"mean_CKA={statistics.mean(cka):.3f}  mean_coh={statistics.mean(coh):.2f}  n={len(entries)}")

    # Save
    out_path = Path("results/coherence/prompt_ref_cka.json")
    out_path.write_text(json.dumps(matched, indent=2))
    print(f"\nSaved matched data: {out_path}")


if __name__ == "__main__":
    main()
