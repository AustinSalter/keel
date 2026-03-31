#!/usr/bin/env python3
"""
Sprint 2.5 trace preparation.

Takes the 4 coherent dialectic traces and produces:
1. Truncated versions (5 iterations max)
2. Qwen chat-template formatted versions with accumulation points
3. D1 (shuffled turns) degraded variants
4. D2 (plausible substitution) targets + Opus-generated substitutions
5. D3 (dilutory substitution) Opus-generated substitutions

Usage:
    python scripts/prepare_traces.py                    # Steps 1a-1d (no API calls)
    python scripts/prepare_traces.py --generate-d2-d3   # Steps 1e-1f (Opus API calls)
"""

import json
import random
import re
import sys
from pathlib import Path

KEEL_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = KEEL_ROOT / "traces"

SESSIONS = ["thesis_geometry", "harness_thesis", "agentic_commerce", "kinnected"]
MAX_ITERATIONS = 5
SHUFFLE_SEED = 42

QWEN_CHAT_SYSTEM = "You are reading a multi-turn dialectic reasoning session."

ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}


# ---------------------------------------------------------------------------
# 1a. Truncate to 5 iterations
# ---------------------------------------------------------------------------

def truncate_to_iterations(session: str, max_iter: int = MAX_ITERATIONS) -> list[dict]:
    """Load coherent.jsonl and filter to turns from iterations 1-max_iter."""
    input_path = TRACES_DIR / session / "coherent.jsonl"
    entries = []
    with open(input_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["iteration"] <= max_iter:
                entries.append(entry)
    return entries


def save_truncated(session: str, entries: list[dict]):
    """Write truncated entries to coherent_5iter.jsonl."""
    output_path = TRACES_DIR / session / "coherent_5iter.jsonl"
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return output_path


# ---------------------------------------------------------------------------
# 1b. Format for Qwen chat template
# ---------------------------------------------------------------------------

def format_turn_qwen(role: str, content: str) -> str:
    """Format a single turn in Qwen's chat template."""
    mapped_role = ROLE_MAP.get(role, "user")
    return f"<|im_start|>{mapped_role}\n{content}<|im_end|>\n"


def build_accumulation_points(entries: list[dict]) -> dict[str, str]:
    """Build accumulation points at phase boundaries only.

    Measurement points are created at:
    - The first user turn (thesis/context)
    - Every expansion, compression, and critique phase turn

    Non-phase turns (system messages, tool results, inter-iteration messages)
    accumulate in the running text but don't get their own measurement point.

    Returns dict like {"turn_0": "...", "turn_5": "...", ...} where keys are
    the turn numbers of measurement points and values are the full chat-formatted
    string up to and including that turn.
    """
    system_header = f"<|im_start|>system\n{QWEN_CHAT_SYSTEM}<|im_end|>\n"

    accumulation = {}
    running = system_header
    first_user_seen = False

    for entry in entries:
        turn_str = format_turn_qwen(entry["role"], entry["content"])
        running += turn_str

        # Create measurement point at first user turn or any phase turn
        is_first_user = entry["role"] == "user" and not first_user_seen
        is_phase_turn = entry["phase"] is not None

        if is_first_user:
            first_user_seen = True
            accumulation[f"turn_{entry['turn']}"] = running
        elif is_phase_turn:
            accumulation[f"turn_{entry['turn']}"] = running

    return accumulation


def save_formatted(session: str, accumulation: dict[str, str]):
    """Write formatted accumulation points to formatted.json."""
    output_path = TRACES_DIR / session / "formatted.json"
    with open(output_path, "w") as f:
        json.dump(accumulation, f, ensure_ascii=False, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# 1c. Generate D1 (shuffled turns)
# ---------------------------------------------------------------------------

def build_d1_shuffled(entries: list[dict]) -> dict[str, str]:
    """Shuffle all turns except turn 0 (context), then build accumulation points."""
    if not entries:
        return {}

    # Separate turn 0 from the rest
    turn_0 = [e for e in entries if e["turn"] == entries[0]["turn"]]
    rest = [e for e in entries if e["turn"] != entries[0]["turn"]]

    # Shuffle the rest with fixed seed
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(rest)

    # Reassemble with sequential turn numbers
    shuffled = []
    for i, entry in enumerate(turn_0 + rest):
        shuffled.append({**entry, "turn": i})

    return build_accumulation_points(shuffled)


def save_d1(session: str, accumulation: dict[str, str]):
    output_path = TRACES_DIR / session / "d1_shuffled.json"
    with open(output_path, "w") as f:
        json.dump(accumulation, f, ensure_ascii=False, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# 1d. Identify D2 substitution targets
# ---------------------------------------------------------------------------

def has_thesis_modification(content: str) -> bool:
    """Check if a turn contains thesis modification markers."""
    content_lower = content.lower()

    # Explicit markers
    if any(marker in content for marker in [
        "THESIS MODIFIED", "Thesis Modified", "MODIFIED →",
        "**Updated Thesis**", "**Revised Thesis**", "**Modified Thesis**",
    ]):
        return True

    # Thesis update language in critique/compression
    if re.search(r'(?:current|updated|revised|new|modified)\s+thesis\s*:', content_lower):
        return True
    if any(w in content_lower for w in [
        "thesis should be", "thesis needs to", "refine the thesis",
        "modify the thesis", "update the thesis", "amend the thesis",
        "thesis has evolved", "thesis now"
    ]):
        return True

    return False


def find_d2_target(entries: list[dict]) -> dict | None:
    """Find the critique turn from the first iteration with a thesis modification."""
    # Find iterations with modifications
    mod_iterations = set()
    for entry in entries:
        if entry["phase"] in ("critique", "compression") and has_thesis_modification(entry["content"]):
            mod_iterations.add(entry["iteration"])

    if not mod_iterations:
        # Fallback: use the critique from iteration 3 (middle)
        for entry in entries:
            if entry["phase"] == "critique" and entry["iteration"] == 3:
                return {
                    "iteration": entry["iteration"],
                    "turn": entry["turn"],
                    "phase": entry["phase"],
                    "original_content_preview": entry["content"][:200],
                    "fallback": True,
                }
        return None

    first_mod_iter = min(mod_iterations)

    # Find the critique turn from that iteration
    for entry in entries:
        if entry["phase"] == "critique" and entry["iteration"] == first_mod_iter:
            return {
                "iteration": entry["iteration"],
                "turn": entry["turn"],
                "phase": entry["phase"],
                "original_content_preview": entry["content"][:200],
                "original_content_length": len(entry["content"]),
                "fallback": False,
            }

    return None


def save_d2_target(session: str, target: dict):
    output_path = TRACES_DIR / session / "d2_target.json"
    with open(output_path, "w") as f:
        json.dump(target, f, ensure_ascii=False, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# 1e-f. Generate D2/D3 via Opus API
# ---------------------------------------------------------------------------

D2_SYSTEM_PROMPT = """You are generating a plausible but structurally drifting response for a dialectic reasoning experiment.

Given the conversation context up to this point, generate a CRITIQUE PASS that sounds thoughtful and on-topic but does NOT actually follow from the evidence presented in the preceding expansion and compression passes.

It should drift — introduce new framings that weren't set up, reference considerations not established in prior turns, or reach conclusions that don't connect to the compression pass's synthesis. The reader should feel "this sounds smart but doesn't quite follow from what came before."

The response should be approximately the same length as the original critique. Maintain the same structural format (headers, markers, confidence scores if present)."""

D3_SYSTEM_PROMPT = """You are generating a dilutory, non-committal response for a dialectic reasoning experiment.

Given the conversation context, generate a CRITIQUE PASS that stays on-topic but adds nothing structural. It should:
- Restate what was already said in different words
- Hedge every position ("on the other hand", "it depends", "more research needed")
- Ask rhetorical questions without answering them
- Avoid taking any clear position or advancing the argument
- Reference the same evidence but draw no new conclusions

The response should be approximately the same length as the original critique but carry no argumentative weight. The reader should feel "this says a lot of words but doesn't actually move anything forward."

Maintain the same structural format (headers, markers, confidence scores if present)."""


def build_context_for_substitution(entries: list[dict], target_turn: int) -> str:
    """Build the conversation context up to (but not including) the target turn."""
    context_parts = []
    for entry in entries:
        if entry["turn"] >= target_turn:
            break
        context_parts.append(f"[Turn {entry['turn']} — {entry['role']}, "
                           f"phase={entry['phase']}, iteration={entry['iteration']}]\n"
                           f"{entry['content']}")
    return "\n\n---\n\n".join(context_parts)


def get_original_turn(entries: list[dict], target_turn: int) -> dict | None:
    """Get the original entry for a specific turn number."""
    for entry in entries:
        if entry["turn"] == target_turn:
            return entry
    return None


def generate_substitution(entries: list[dict], target_turn: int,
                          system_prompt: str, variant: str) -> str:
    """Call Opus API to generate a substitution for the target turn."""
    import anthropic

    client = anthropic.Anthropic()
    context = build_context_for_substitution(entries, target_turn)
    original = get_original_turn(entries, target_turn)

    user_prompt = (
        f"Here is the dialectic conversation up to the turn you need to replace:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"The original turn was a {original['phase']} pass from iteration "
        f"{original['iteration']}, approximately {len(original['content'])} characters long.\n\n"
        f"Generate the {variant} replacement. Match the approximate length."
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        temperature=1.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text


def build_substituted_trace(entries: list[dict], target_turn: int,
                            substitution_text: str) -> dict[str, str]:
    """Build accumulation points with one turn replaced."""
    modified = []
    for entry in entries:
        if entry["turn"] == target_turn:
            modified.append({**entry, "content": substitution_text})
        else:
            modified.append(entry)
    return build_accumulation_points(modified)


def save_substitution(session: str, text: str, variant: str):
    filename = f"d2_substitution.md" if variant == "d2" else "d3_substitution.md"
    output_path = TRACES_DIR / session / filename
    with open(output_path, "w") as f:
        f.write(text)
    return output_path


def save_d2_trace(session: str, accumulation: dict[str, str]):
    output_path = TRACES_DIR / session / "d2_plausible.json"
    with open(output_path, "w") as f:
        json.dump(accumulation, f, ensure_ascii=False, indent=2)
    return output_path


def save_d3_trace(session: str, accumulation: dict[str, str]):
    output_path = TRACES_DIR / session / "d3_dilutory.json"
    with open(output_path, "w") as f:
        json.dump(accumulation, f, ensure_ascii=False, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_local_steps():
    """Steps 1a-1d: no API calls needed."""
    print("=" * 60)
    print("Sprint 2.5 Trace Preparation (local steps)")
    print("=" * 60)

    for session in SESSIONS:
        print(f"\n--- {session} ---")

        # 1a. Truncate
        entries = truncate_to_iterations(session)
        save_truncated(session, entries)
        print(f"  Truncated: {len(entries)} turns (iterations 1-{MAX_ITERATIONS})")

        # 1b. Format
        accumulation = build_accumulation_points(entries)
        save_formatted(session, accumulation)
        n_turns = len(accumulation)
        last_key = list(accumulation.keys())[-1]
        last_len = len(accumulation[last_key])
        print(f"  Formatted: {n_turns} accumulation points, final={last_len:,} chars")

        # 1c. D1 shuffled
        d1 = build_d1_shuffled(entries)
        save_d1(session, d1)
        print(f"  D1 shuffled: {len(d1)} accumulation points")

        # 1d. D2 target
        target = find_d2_target(entries)
        if target:
            save_d2_target(session, target)
            print(f"  D2 target: iteration {target['iteration']}, "
                  f"turn {target['turn']}, "
                  f"{'(fallback)' if target.get('fallback') else ''}")
            print(f"    Preview: {target['original_content_preview'][:80]}...")
        else:
            print("  D2 target: NONE FOUND — no thesis modifications detected")


def run_api_steps():
    """Steps 1e-1f: Opus API calls for D2/D3 generation."""
    print("\n" + "=" * 60)
    print("Sprint 2.5 Trace Preparation (API steps)")
    print("=" * 60)

    for session in SESSIONS:
        print(f"\n--- {session} ---")

        # Load entries and target
        entries = truncate_to_iterations(session)
        target_path = TRACES_DIR / session / "d2_target.json"
        if not target_path.exists():
            print("  SKIP: no d2_target.json — run local steps first")
            continue

        with open(target_path) as f:
            target = json.load(f)

        target_turn = target["turn"]

        # 1e. Generate D2
        print(f"  Generating D2 (plausible substitution) for turn {target_turn}...")
        d2_text = generate_substitution(entries, target_turn, D2_SYSTEM_PROMPT, "plausible-but-drifting")
        save_substitution(session, d2_text, "d2")
        d2_trace = build_substituted_trace(entries, target_turn, d2_text)
        save_d2_trace(session, d2_trace)
        print(f"  D2: {len(d2_text):,} chars substitution, {len(d2_trace)} accumulation points")

        # 1f. Generate D3
        print(f"  Generating D3 (dilutory substitution) for turn {target_turn}...")
        d3_text = generate_substitution(entries, target_turn, D3_SYSTEM_PROMPT, "dilutory-non-committal")
        save_substitution(session, d3_text, "d3")
        d3_trace = build_substituted_trace(entries, target_turn, d3_text)
        save_d3_trace(session, d3_trace)
        print(f"  D3: {len(d3_text):,} chars substitution, {len(d3_trace)} accumulation points")


def main():
    generate = "--generate-d2-d3" in sys.argv

    run_local_steps()

    if generate:
        run_api_steps()
    else:
        print("\n\nTo generate D2/D3 substitutions via Opus API:")
        print("  python scripts/prepare_traces.py --generate-d2-d3")


if __name__ == "__main__":
    main()
