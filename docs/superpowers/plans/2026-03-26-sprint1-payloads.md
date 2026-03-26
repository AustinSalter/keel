# Sprint 1: Context Payloads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce 8 context payloads and 30 prompts as static files, ready for Sprint 2's activation capture pipeline.

**Architecture:** Data flows from three sources (Supabase/Phello, Claude memory JSON, Spotify API) through a payload assembly module into static text files. All payloads are pre-generated — Sprint 2 reads files, not APIs. A `data/` directory holds prompts and payloads; `substrate/` holds the code that generates them.

**Tech Stack:** Python 3.11, spotipy (Spotify OAuth), psycopg2-binary (Supabase), existing substrate modules.

---

## File Structure

```
data/                           # NEW directory (tracked in git except secrets)
  prompts.json                  # 30 prompts with category/type metadata
  payloads/
    p_null.txt                  # Empty file
    p1_questionnaire.txt        # Synthesized questionnaire prose
    p2_claude_memory.txt        # Trimmed Claude conversation memory
    p3_spotify.txt              # Synthesized Spotify taste profile
    p_soul.txt                  # Phello SOUL document verbatim
    p1_p2_combined.txt          # P1 + P2 with transition
    p1_p2_p3_combined.txt       # P1 + P2 + P3 with transitions
    phello/                     # Per-prompt steering documents
      prompt_01.txt
      prompt_02.txt
      ... (30 files)
  .secrets/                     # Gitignored
    spotify_refresh_token

substrate/
  config.py                     # MODIFY: add SWEEP_LAYERS, SUPABASE_* constants
  spotify_context.py            # NEW: Spotify OAuth + taste synthesis
  payloads.py                   # NEW: fetch data, assemble and save all payloads

scripts/
  generate_payloads.py          # NEW: CLI entry point to run the full payload generation
```

---

### Task 1: Update config and project structure

**Files:**
- Modify: `substrate/config.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add sweep-derived layer config and Supabase constants to config.py**

```python
# After existing QWEN_FALLBACK_LAYERS line, add:

# Layer indices derived from Sprint 0 layer sweep
# Layers 7, 11: optimal zone (EVR 0.87+, SNR 12-18x)
# Layer 24: post-noise-wall recovery zone
SWEEP_LAYERS = [7, 11, 24]

# Supabase config (Phello)
SUPABASE_URL = "https://qfyufzqfjxfiqehnveck.supabase.co"
PHELLO_USER_ID = "d09855bd-2290-4d52-bb04-13517eef5b81"
```

- [ ] **Step 2: Update CaptureConfig default to use SWEEP_LAYERS**

Change the default `layer_indices` in `CaptureConfig` from `TRINITY_LAYERS` to `SWEEP_LAYERS`.

- [ ] **Step 3: Add secrets directory to .gitignore**

Append to `.gitignore`:
```
# Secrets
data/.secrets/
```

- [ ] **Step 4: Create data directory structure**

```bash
mkdir -p data/payloads/phello data/.secrets
touch data/payloads/p_null.txt  # Empty file = null payload
```

- [ ] **Step 5: Commit**

```bash
git add substrate/config.py .gitignore data/payloads/p_null.txt
git commit -m "Update config with sweep layers, add data directory structure"
```

---

### Task 2: Write prompts.json

**Files:**
- Create: `data/prompts.json`

- [ ] **Step 1: Write the prompts file**

Create `data/prompts.json` with all 30 prompts. Structure:

```json
{
  "version": 1,
  "prompts": [
    {
      "id": 1,
      "text": "What's the strongest version of the case against building Phello as a standalone company vs integrating into an existing AI platform?",
      "category": "ai_ml",
      "type": "A",
      "description": "Personalized reasoning about Phello strategy"
    }
  ]
}
```

Full prompt list is in the spec at `docs/superpowers/specs/2026-03-26-sprint1-context-payloads-design.md` lines 72-120. Copy all 30 prompts exactly. Categories: `ai_ml`, `strategy`, `food_hospitality`, `outdoor`, `generic_control`. Types: `A` (deeply personal), `B` (domain with personal lens), `C` (generic control).

- [ ] **Step 2: Validate JSON parses**

```bash
uv run python -c "import json; d=json.load(open('data/prompts.json')); print(f'{len(d[\"prompts\"])} prompts loaded'); assert len(d['prompts'])==30"
```

- [ ] **Step 3: Commit**

```bash
git add data/prompts.json
git commit -m "Add 30 experimental prompts across 5 categories"
```

---

### Task 3: Build Spotify context module

**Files:**
- Create: `substrate/spotify_context.py`
- Create: `tests/test_spotify_context.py`

- [ ] **Step 1: Add spotipy dependency**

```bash
uv add spotipy
```

- [ ] **Step 2: Write test for synthesis function**

The synthesis function takes structured Spotify data (dicts of artists, tracks, genres) and produces a prose taste profile. This is testable without API access.

```python
# tests/test_spotify_context.py
def test_synthesize_taste_profile():
    """Synthesize produces prose from structured Spotify data."""
    from substrate.spotify_context import synthesize_taste_profile

    data = {
        "top_artists": [
            {"name": "Khruangbin", "genres": ["psychedelic soul", "funk"]},
            {"name": "Tame Impala", "genres": ["psychedelic rock", "indie"]},
            {"name": "Mac DeMarco", "genres": ["indie rock", "slacker rock"]},
        ],
        "top_tracks": [
            {"name": "Time (You and I)", "artist": "Khruangbin"},
            {"name": "Let It Happen", "artist": "Tame Impala"},
        ],
        "top_genres": ["psychedelic soul", "indie rock", "funk", "psychedelic rock"],
    }

    result = synthesize_taste_profile(data)
    assert isinstance(result, str)
    assert len(result) > 100  # Non-trivial output
    assert "Khruangbin" in result or "psychedelic" in result  # References source data
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_spotify_context.py -v
```

- [ ] **Step 4: Implement spotify_context.py**

```python
# substrate/spotify_context.py
"""Spotify OAuth and taste profile synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyOAuth


def create_oauth(
    client_id: str,
    client_secret: str,
    redirect_uri: str = "http://localhost:8888/callback",
    cache_path: str | None = None,
) -> SpotifyOAuth:
    """Create a SpotifyOAuth manager."""
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope="user-top-read user-read-recently-played",
        cache_path=cache_path,
    )


def fetch_spotify_data(sp: spotipy.Spotify) -> dict:
    """Fetch top artists, tracks, and derive genre distribution."""
    top_artists_raw = sp.current_user_top_artists(limit=20, time_range="medium_term")
    top_tracks_raw = sp.current_user_top_tracks(limit=20, time_range="medium_term")

    top_artists = [
        {"name": a["name"], "genres": a["genres"]}
        for a in top_artists_raw["items"]
    ]

    top_tracks = [
        {"name": t["name"], "artist": t["artists"][0]["name"]}
        for t in top_tracks_raw["items"]
    ]

    # Aggregate genres across all artists
    genre_counts: dict[str, int] = {}
    for artist in top_artists:
        for genre in artist["genres"]:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:10]

    return {
        "top_artists": top_artists,
        "top_tracks": top_tracks,
        "top_genres": top_genres,
    }


def synthesize_taste_profile(data: dict) -> str:
    """Synthesize structured Spotify data into a natural prose taste profile.

    Target: 300-500 tokens of flowing prose about listening patterns and taste.
    """
    artists = data["top_artists"]
    tracks = data["top_tracks"]
    genres = data["top_genres"]

    artist_names = [a["name"] for a in artists[:10]]
    track_lines = [f'"{t["name"]}" by {t["artist"]}' for t in tracks[:8]]
    genre_str = ", ".join(genres[:6])

    # Build prose sections
    sections = []

    sections.append(
        f"Music taste centers on {genre_str}. "
        f"Top artists include {', '.join(artist_names[:5])}"
        + (f", along with {', '.join(artist_names[5:8])}" if len(artist_names) > 5 else "")
        + "."
    )

    if tracks:
        sections.append(
            f"Frequently played tracks include {', '.join(track_lines[:4])}."
        )

    # Infer aesthetic from genres
    aesthetic_signals = []
    genre_set = set(g.lower() for g in genres)
    if genre_set & {"indie rock", "indie", "indie pop", "indie folk"}:
        aesthetic_signals.append("indie sensibility that values authenticity over mainstream polish")
    if genre_set & {"psychedelic rock", "psychedelic soul", "neo-psychedelia"}:
        aesthetic_signals.append("affinity for atmospheric, textured soundscapes")
    if genre_set & {"funk", "soul", "r&b"}:
        aesthetic_signals.append("groove-oriented, rhythmically driven listening")
    if genre_set & {"hip hop", "rap", "conscious hip hop"}:
        aesthetic_signals.append("engagement with lyrical storytelling and cultural commentary")
    if genre_set & {"electronic", "ambient", "downtempo"}:
        aesthetic_signals.append("comfort with ambient and electronic textures, possibly as focus or work music")
    if genre_set & {"country", "americana", "folk"}:
        aesthetic_signals.append("connection to roots music and storytelling traditions")
    if genre_set & {"classical", "jazz", "instrumental"}:
        aesthetic_signals.append("appreciation for instrumental complexity and compositional depth")

    if aesthetic_signals:
        sections.append(
            "These patterns suggest " + "; ".join(aesthetic_signals[:3]) + "."
        )

    return " ".join(sections)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_spotify_context.py -v
```

- [ ] **Step 6: Commit**

```bash
git add substrate/spotify_context.py tests/test_spotify_context.py
git commit -m "Add Spotify context module with taste synthesis"
```

---

### Task 4: Build payload assembly module

**Files:**
- Create: `substrate/payloads.py`
- Create: `tests/test_payloads.py`

This is the core module. It fetches data from all sources and assembles the 8 payload variants.

- [ ] **Step 1: Write tests for payload assembly**

Test the assembly logic using mock data (no live API calls). Key tests:

```python
# tests/test_payloads.py

def test_build_p1_from_questionnaire_data():
    """P1 synthesis produces prose from questionnaire data."""
    ...

def test_build_p2_from_claude_memory():
    """P2 extraction produces trimmed behavioral profile."""
    ...

def test_build_combined_p1_p2():
    """P1+P2 concatenation includes transition text."""
    ...

def test_build_combined_p1_p2_p3():
    """Full stack concatenation includes all three payloads."""
    ...

def test_payload_token_counts():
    """All payloads fall within target token ranges."""
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_payloads.py -v
```

- [ ] **Step 3: Implement payloads.py**

Key functions:

```python
# substrate/payloads.py

def fetch_questionnaire_data(service_key: str) -> dict:
    """Query Supabase for questionnaire responses, fragments, and signals."""
    ...

def fetch_soul(service_key: str) -> str:
    """Query Supabase for the SOUL document."""
    ...

def fetch_phello_synthesis(service_key: str, prompt: str) -> str:
    """Call the Supabase synthesize edge function for a single prompt."""
    ...

def load_claude_memory(path: Path) -> str:
    """Load and extract conversations_memory from memories.json."""
    ...

def build_p1(questionnaire_data: dict) -> str:
    """Synthesize questionnaire data into natural prose. Target: 500-800 tokens."""
    ...

def build_p2(claude_memory_raw: str) -> str:
    """Trim Claude memory to behavioral essentials. Target: 500-800 tokens."""
    ...

def build_combined(p1: str, p2: str, p3: str | None = None) -> str:
    """Concatenate payloads with natural transitions."""
    ...

def save_all_payloads(output_dir: Path, payloads: dict[str, str]) -> None:
    """Save all payload variants to text files."""
    ...
```

The `build_p1` function is the most creative — it must synthesize 10 Q&A answers, preference signals, and fragments into natural flowing prose. NOT Q&A format. NOT bullet points. A narrative profile document.

The `build_p2` function extracts `conversations_memory` from the JSON, then trims to 500-800 tokens by:
- Keeping: Work context, Personal context, Brief history (recent months)
- Removing: "Top of mind" (ephemeral), "Earlier context" (dated), specific names/dates

The `fetch_phello_synthesis` function calls:
```
POST https://qfyufzqfjxfiqehnveck.supabase.co/functions/v1/synthesize
Headers: Authorization: Bearer {service_key}
Body: {"user_id": "{PHELLO_USER_ID}", "query": "{prompt_text}"}
```
Check the edge function's expected request format in `/Users/austinsalter/Documents/Side-Projects/phello-mobile/supabase/functions/synthesize/index.ts` before implementing.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_payloads.py -v
```

- [ ] **Step 5: Commit**

```bash
git add substrate/payloads.py tests/test_payloads.py
git commit -m "Add payload assembly module with P1-P8 builders"
```

---

### Task 5: Build payload generation script

**Files:**
- Create: `scripts/generate_payloads.py`

- [ ] **Step 1: Implement the generation script**

This script orchestrates the full payload generation pipeline:

```python
# scripts/generate_payloads.py
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
    parser = argparse.ArgumentParser()
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

    print("Fetching SOUL document...")
    soul = fetch_soul(args.service_key)

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
    for prompt in prompts:
        pid = f"prompt_{prompt['id']:02d}"
        steering = fetch_phello_synthesis(args.service_key, prompt["text"])
        (phello_dir / f"{pid}.txt").write_text(steering)
        print(f"  {pid}: {len(steering)} chars")

    # 6. Print token count summary
    print("\n=== Token Count Summary ===")
    for name, text in payloads.items():
        # Rough token estimate: ~4 chars per token
        est_tokens = len(text) // 4
        print(f"  {name}: ~{est_tokens} tokens ({len(text)} chars)")

    print("\nPayload generation complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/generate_payloads.py
git commit -m "Add payload generation script"
```

---

### Task 6: Spotify OAuth flow and P3 generation

**Files:**
- Create: `scripts/spotify_auth.py`

This is a one-time interactive script to get the Spotify refresh token.

- [ ] **Step 1: Create Spotify app at developer.spotify.com**

The user must create a Spotify app at https://developer.spotify.com/dashboard and set the redirect URI to `http://localhost:8888/callback`. Note the Client ID and Client Secret.

- [ ] **Step 2: Write the auth + fetch script**

```python
# scripts/spotify_auth.py
"""One-time Spotify auth + P3 payload generation."""

import argparse
from pathlib import Path

import spotipy

from substrate.spotify_context import create_oauth, fetch_spotify_data, synthesize_taste_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True)
    parser.add_argument("--output", default="data/payloads/p3_spotify.txt")
    args = parser.parse_args()

    auth = create_oauth(
        client_id=args.client_id,
        client_secret=args.client_secret,
        cache_path="data/.secrets/spotify_token_cache",
    )
    sp = spotipy.Spotify(auth_manager=auth)

    print("Fetching Spotify data...")
    data = fetch_spotify_data(sp)
    print(f"  {len(data['top_artists'])} artists, {len(data['top_tracks'])} tracks, {len(data['top_genres'])} genres")

    print("Synthesizing taste profile...")
    profile = synthesize_taste_profile(data)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(profile)
    print(f"Saved P3 to {args.output}")
    print(f"  ~{len(profile) // 4} tokens ({len(profile)} chars)")
    print(f"\n{profile}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the auth flow interactively**

The user runs:
```bash
uv run python scripts/spotify_auth.py --client-id YOUR_ID --client-secret YOUR_SECRET
```
This opens a browser for OAuth consent, then saves the token and generates P3.

- [ ] **Step 4: Commit (excluding secrets)**

```bash
git add scripts/spotify_auth.py
git commit -m "Add Spotify auth script for one-time P3 generation"
```

---

### Task 7: Run full payload generation

This task requires all previous tasks to be complete. It's the integration run.

**Files:**
- All `data/payloads/*.txt` files (generated, tracked in git)
- All `data/payloads/phello/*.txt` files (generated, tracked in git)

- [ ] **Step 1: Generate P3 (Spotify) if not already done**

```bash
uv run python scripts/spotify_auth.py --client-id ID --client-secret SECRET
```

- [ ] **Step 2: Run the full payload generation**

```bash
PYTHONPATH=. uv run python scripts/generate_payloads.py \
  --service-key "eyJ..." \
  --spotify-payload data/payloads/p3_spotify.txt \
  --claude-memory "Austin's Claude Profile/memories.json"
```

- [ ] **Step 3: Verify token counts fall within targets**

| Payload | Target tokens | Acceptable range |
|---------|--------------|------------------|
| p_null | 0 | 0 |
| p1_questionnaire | 500-800 | 400-1000 |
| p2_claude_memory | 500-800 | 400-1000 |
| p3_spotify | 300-500 | 200-600 |
| p_soul | ~300 | 200-500 |
| p1_p2_combined | 1000-1600 | 800-2000 |
| p1_p2_p3_combined | 1300-2100 | 1000-2500 |
| phello/prompt_XX | 400-600 each | 200-800 |

- [ ] **Step 4: Spot-check payload quality**

Read P1, P2, P_soul. Verify:
- P1 reads as natural prose (not Q&A, not bullets)
- P2 focuses on behavioral patterns (not ephemeral details)
- P_soul matches the Supabase SOUL document

- [ ] **Step 5: Commit all payloads**

```bash
git add data/
git commit -m "Generate all Sprint 1 payloads (8 variants, 30 P_phello steering docs)"
```

---

### Task 8: Pilot validation run

Run a small subset through the Sprint 0 infrastructure to verify payloads work end-to-end.

**Files:**
- Create: `scripts/pilot_run.py`

- [ ] **Step 1: Write pilot script**

The pilot runs 3 payloads (P_null, P1, P_phello) x 3 prompts (one from categories 1, 3, 5) x 1 completion on Modal. Uses the existing `capture_and_analyze` pipeline with the new `SWEEP_LAYERS`.

This verifies:
- Payloads inject correctly as context
- Activation capture works with new layer indices
- PCA runs without errors across payload types
- Geometric differences are visible between P_null and P1/P_phello

- [ ] **Step 2: Deploy updated Modal app (if needed for SWEEP_LAYERS)**

```bash
uv run modal deploy substrate/modal_app.py
```

- [ ] **Step 3: Run the pilot**

```bash
PYTHONPATH=. uv run python scripts/pilot_run.py --service-key "eyJ..."
```

Expected output: a table showing Grassmann distances between P_null and P1/P_phello for each of the 3 prompts, at each of the 3 sweep layers.

- [ ] **Step 4: Verify signal exists**

P_null vs P1 and P_null vs P_phello should show meaningful Grassmann distances (> 0.5) for category 1 and 3 prompts, and minimal distances for category 5 (control).

- [ ] **Step 5: Commit pilot script**

```bash
git add scripts/pilot_run.py
git commit -m "Add Sprint 1 pilot validation script"
```

---

## Acceptance Criteria

- [ ] `data/prompts.json` has 30 prompts with correct category/type metadata
- [ ] `data/payloads/` has 7 static payload files + 30 phello/ steering documents
- [ ] All payload token counts within acceptable ranges
- [ ] `substrate/config.py` uses `SWEEP_LAYERS = [7, 11, 24]` as default
- [ ] Pilot run shows geometric signal for personalized prompts, minimal signal for control prompts
- [ ] All tests pass: `uv run pytest tests/ -v`
