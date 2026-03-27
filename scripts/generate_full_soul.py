"""Generate a full SOUL document from questionnaire + Claude memory + Spotify data."""

import argparse
import json
from pathlib import Path

import anthropic

from substrate.config import PHELLO_USER_ID, SUPABASE_URL
from substrate.payloads import fetch_questionnaire_data, fetch_soul, load_claude_memory


SOUL_SYSTEM_PROMPT = """You are synthesizing a person's identity document (~300 tokens) from three data sources:
1. Questionnaire responses: explicit stated preferences about lifestyle, food, travel, etc.
2. Claude conversation memory: revealed behavioral patterns from months of AI interaction — work context, intellectual style, reasoning patterns, project themes.
3. Spotify listening data: implicit taste — what they actually consume, aesthetic preferences.

Produce a single flowing prose paragraph (no headers, no bullets, no sections) that captures WHO this person is — their identity, not just their preferences. Write in third person. The document should read like a portrait by someone who deeply understands this person.

Key principles:
- Density over length. Every sentence must carry signal.
- Integrate across sources — don't list facts from each source separately.
- Capture tensions and dualities (e.g., gritty outdoor person who also appreciates fine dining).
- Include the intellectual/professional dimension alongside the lifestyle dimension.
- Target: ~300 tokens of flowing prose."""


def main():
    parser = argparse.ArgumentParser(description="Generate full SOUL from all 3 data sources.")
    parser.add_argument("--service-key", required=True, help="Supabase service role key")
    parser.add_argument("--claude-memory", default="Austin's Claude Profile/memories.json")
    parser.add_argument("--spotify-payload", default="data/payloads/p3_spotify.txt")
    parser.add_argument("--output", default="data/payloads/p_soul_full.txt")
    parser.add_argument("--anthropic-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env)")
    args = parser.parse_args()

    # Gather all three data sources
    print("Gathering data sources...")

    # Source 1: Questionnaire
    q_data = fetch_questionnaire_data(args.service_key)
    q_text = ""
    responses = q_data.get("responses", [])
    if responses and isinstance(responses[0], list):
        responses = responses[0]
    for qa in responses:
        if isinstance(qa, dict):
            q_text += f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}\n\n"
    print(f"  Questionnaire: {len(q_text)} chars")

    # Source 2: Claude memory
    claude_raw = load_claude_memory(Path(args.claude_memory))
    print(f"  Claude memory: {len(claude_raw)} chars")

    # Source 3: Spotify
    spotify_text = Path(args.spotify_payload).read_text() if Path(args.spotify_payload).exists() else ""
    print(f"  Spotify: {len(spotify_text)} chars")

    # Also get current SOUL for comparison
    current_soul = fetch_soul(args.service_key)
    print(f"  Current SOUL: {len(current_soul)} chars")

    # Build the synthesis input
    user_message = f"""Here are the three data sources for this person:

## Source 1: Questionnaire Responses (explicit preferences)
{q_text}

## Source 2: Claude Conversation Memory (behavioral patterns)
{claude_raw}

## Source 3: Spotify Listening Profile (implicit taste)
{spotify_text}

Synthesize a ~300 token identity document from all three sources."""

    # Call Claude API
    print("\nSynthesizing full SOUL via Claude...")
    client = anthropic.Anthropic(api_key=args.anthropic_key) if args.anthropic_key else anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=SOUL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    full_soul = response.content[0].text
    print(f"Full SOUL generated: {len(full_soul)} chars (~{len(full_soul)//4} tokens)")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(full_soul)
    print(f"Saved to {args.output}")

    # Print both for comparison
    print("\n" + "=" * 60)
    print("CURRENT SOUL (questionnaire only):")
    print("=" * 60)
    print(current_soul[:500] + "..." if len(current_soul) > 500 else current_soul)

    print("\n" + "=" * 60)
    print("FULL SOUL (questionnaire + Claude memory + Spotify):")
    print("=" * 60)
    print(full_soul)


if __name__ == "__main__":
    main()
