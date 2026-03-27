"""Generate all 8 payload variants and save to data/payloads/."""

import argparse
import json
from pathlib import Path

from substrate.config import PHELLO_USER_ID, SUPABASE_URL
from substrate.payloads import (
    build_combined,
    build_p1,
    build_p2,
    fetch_phello_synthesis,
    fetch_questionnaire_data,
    fetch_soul,
    load_claude_memory,
    save_all_payloads,
)


def main():
    parser = argparse.ArgumentParser(description="Generate all Sprint 1 payloads.")
    parser.add_argument("--service-key", required=True, help="Supabase service role key")
    parser.add_argument("--spotify-payload", type=str, default=None,
                        help="Path to pre-generated Spotify payload (P3)")
    parser.add_argument("--claude-memory", type=str,
                        default="Austin's Claude Profile/memories.json")
    parser.add_argument("--output-dir", type=str, default="data/payloads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 1. Fetch Phello data
    print("Fetching questionnaire data from Supabase...")
    q_data = fetch_questionnaire_data(args.service_key)
    print(f"  {len(q_data['responses'])} Q&A pairs, {len(q_data['fragments'])} fragments, {len(q_data['signals'])} signals")

    print("Fetching SOUL document...")
    soul = fetch_soul(args.service_key)
    print(f"  SOUL: {len(soul)} chars")

    # 2. Build static payloads
    print("Building P1 (questionnaire)...")
    p1 = build_p1(q_data)

    print("Building P2 (Claude memory)...")
    claude_raw = load_claude_memory(Path(args.claude_memory))
    p2 = build_p2(claude_raw)

    # 3. P3 (Spotify) — loaded from pre-generated file if available
    p3 = None
    if args.spotify_payload and Path(args.spotify_payload).exists():
        p3 = Path(args.spotify_payload).read_text().strip()
        print(f"Loaded P3 from {args.spotify_payload}")

    # 4. Save static payloads
    payloads = {
        "p_null": "",
        "p1_questionnaire": p1,
        "p2_claude_memory": p2,
        "p_soul": soul,
        "p1_p2_combined": build_combined(p1, p2),
    }
    if p3:
        payloads["p3_spotify"] = p3
        payloads["p1_p2_p3_combined"] = build_combined(p1, p2, p3)

    save_all_payloads(output_dir, payloads)

    # 5. Generate P_phello (per-prompt steering documents)
    prompts = json.loads(Path("data/prompts.json").read_text())["prompts"]
    phello_dir = output_dir / "phello"
    phello_dir.mkdir(exist_ok=True)

    print(f"Generating {len(prompts)} P_phello steering documents...")
    import time
    failed = []
    for prompt in prompts:
        pid = f"prompt_{prompt['id']:02d}"
        out_path = phello_dir / f"{pid}.txt"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  {pid}: cached ({out_path.stat().st_size} chars)")
            continue
        for attempt in range(3):
            try:
                steering = fetch_phello_synthesis(args.service_key, prompt["text"])
                out_path.write_text(steering)
                print(f"  {pid}: {len(steering)} chars")
                break
            except Exception as e:
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"  {pid}: error ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  {pid}: FAILED after 3 attempts ({e})")
                    failed.append(pid)
    if failed:
        print(f"\n  WARNING: {len(failed)} P_phello payloads failed: {failed}")

    # 6. Print token count summary
    print("\n=== Token Count Summary ===")
    for name, text in payloads.items():
        est_tokens = len(text) // 4  # Rough estimate
        print(f"  {name}: ~{est_tokens} tokens ({len(text)} chars)")

    print("\nPayload generation complete.")


if __name__ == "__main__":
    main()
