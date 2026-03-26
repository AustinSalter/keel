"""Tests for substrate.payloads — builder functions only, no live API calls."""

from __future__ import annotations

import json
import re
from pathlib import Path

from substrate.payloads import (
    build_combined,
    build_p1,
    build_p2,
    load_claude_memory,
    save_all_payloads,
)


# ---------------------------------------------------------------------------
# Fixtures / mock data
# ---------------------------------------------------------------------------

MOCK_QUESTIONNAIRE_DATA: dict = {
    "responses": [
        {
            "travel_dream": "I'd love to spend a month in Spain — Barcelona for the architecture and San Sebastián for the food.",
            "favorite_meal": "Wagyu ribeye, medium rare, with a simple side of roasted vegetables. Omakase if I'm feeling adventurous.",
            "food_aversions": "Shellfish allergy, and I can't stand truffle oil — it ruins everything it touches.",
            "dining_vibe": "Buzzy and energetic when I'm out with friends, but for a date I want somewhere intimate with great lighting.",
            "decompression": "Climbing is the main thing. I also run, always without headphones — I need the quiet. Afterwards, mezcal on the patio.",
            "entertainment": "NYT crosswords most mornings, indie movies when I can find them, and I've gotten into pottery recently.",
            "lifestyle_shift": "I used to be a total night owl but forced myself into mornings. It stuck. The Bear is probably my favorite show right now.",
            "reading": "Just finished Unreasonable Hospitality and it genuinely changed how I think about service. TBPN podcast is my go-to.",
            "trip_highlight": "A road trip along the Central Coast — Big Sur into SLO. The combination of landscape and food scene was perfect.",
        }
    ],
    "fragments": [
        {
            "domain": "food",
            "subdomain": "dining_style",
            "prose_body": "He gravitates toward chef-driven restaurants where the kitchen takes creative risks. Buzzy atmospheres with friends, intimate settings for dates. Allergic to shellfish, viscerally opposed to truffle oil.",
        },
        {
            "domain": "travel",
            "subdomain": "destinations",
            "prose_body": "The Central Coast of California holds deep appeal — SLO and Big Sur in particular. Spain is the aspirational destination, specifically for the intersection of architecture and food culture.",
        },
        {
            "domain": "lifestyle",
            "subdomain": "routines",
            "prose_body": "A reformed night owl who now protects his mornings. Decompresses through climbing, silent running, and mezcal on the patio. Values craft in entertainment — pottery, indie film, NYT crosswords.",
        },
    ],
    "signals": [
        {"domain": "food", "signal_type": "allergy", "value": "shellfish", "confidence": 1.0},
        {"domain": "food", "signal_type": "aversion", "value": "truffle oil", "confidence": 0.95},
        {"domain": "food", "signal_type": "preference", "value": "wagyu ribeye", "confidence": 0.9},
        {"domain": "food", "signal_type": "preference", "value": "omakase", "confidence": 0.85},
        {"domain": "travel", "signal_type": "aspiration", "value": "Spain", "confidence": 0.9},
        {"domain": "outdoor", "signal_type": "activity", "value": "rock climbing", "confidence": 1.0},
    ],
}

MOCK_CLAUDE_MEMORY = """\
**Work context**

Austin is an independent contractor based in Austin, Texas, operating at the intersection of \
AI tooling, data science, and marketing strategy. He recently transitioned from a retail media \
analytics background (Ad Measurement Lead at The Bluebird Group) into full-time contract work, \
with a pipeline of engagements spanning AI harness development, GTM strategy, full-stack feature \
delivery, and fractional AI integration. He is actively building his consulting practice while \
pursuing longer-term opportunities in VC, founding engineer, or technical PM roles.

**Personal context**

Austin has a classical education background with national-level debate experience, which deeply \
informs his intellectual approach — particularly his interest in dialectical reasoning, rhetorical \
structure, and philosophical frameworks applied to AI systems. He is an advanced rock climber \
(5.13a/V8) training toward V10+/14a. He writes a Substack series exploring the intersection \
of classical philosophy and AI methodology.

**Top of mind**

Austin is preparing for a live call with Dan, founder of OpenProse, and has built a full GTM \
deck and prep materials for the engagement. He is also actively building a March Madness bracket \
prediction system using a sophisticated ML pipeline.

**Brief history**

*Recent months*

Austin has been deeply engaged with OpenProse, conducting technical research and developing \
GTM strategy materials. He has been finalizing his Substack series on dialectical AI reasoning. \
He ran a blind memo evaluation comparing multi-agent, multi-pass, and dialectic plugin outputs.

Austin has been building his contractor business infrastructure: LLC formation, MSA/SOW templates, \
IP protection for pre-existing work.

*Earlier context*

Austin was job searching after leaving The Bluebird Group, targeting VC associate, founding \
engineer, technical PM, and strategy roles in AI, hard tech, energy, or defense. He developed \
a detailed Stanford GSB application strategy and GMAT prep plan.

*Long-term background*

Austin spent several years as Ad Measurement Lead at The Bluebird Group building production ML \
systems: incrementality measurement engines, campaign automation platforms, attribution models, \
and a sophisticated ASIN budget allocation system. His core intellectual thesis: capability does \
not equal coherence, prioritization over prediction, and augmentation over automation."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_p1_produces_prose() -> None:
    """P1 output is flowing prose — no Q&A formatting, no bullets, right length."""
    result = build_p1(MOCK_QUESTIONNAIRE_DATA)

    # Must be a non-empty string
    assert isinstance(result, str)
    assert len(result) > 0

    # Length in range: 500-800 tokens ≈ 2000-3200 characters.
    # Allow some slack (1500–4000) since char-to-token ratios vary.
    assert 1500 <= len(result) <= 4000, (
        f"P1 length {len(result)} chars is outside the 1500-4000 char target range"
    )

    # No Q&A formatting
    assert "Q:" not in result
    assert "A:" not in result

    # No bullet points
    assert not re.search(r"^\s*[-•*]\s", result, re.MULTILINE), (
        "P1 should not contain bullet points"
    )

    # Should be multi-paragraph prose
    paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 2, "P1 should have at least 2 paragraphs"

    # Should contain substance from the questionnaire domains
    result_lower = result.lower()
    assert any(
        word in result_lower
        for word in ["food", "dining", "restaurant", "wagyu", "omakase", "chef"]
    ), "P1 should reference food preferences"


def test_build_p2_trims_memory() -> None:
    """P2 trims Claude memory, excludes ephemeral content, stays in length range."""
    result = build_p2(MOCK_CLAUDE_MEMORY)

    assert isinstance(result, str)
    assert len(result) > 0

    # Should be shorter than the input
    assert len(result) < len(MOCK_CLAUDE_MEMORY), (
        "P2 should be shorter than the raw Claude memory"
    )

    # "Top of mind" section should be removed
    assert "Top of mind" not in result
    assert "March Madness" not in result

    # Length target: 500-800 tokens ≈ 2000-3200 characters. Allow slack.
    assert 1000 <= len(result) <= 4000, (
        f"P2 length {len(result)} chars is outside the 1000-4000 char target range"
    )

    # Should retain important context
    result_lower = result.lower()
    assert any(
        w in result_lower for w in ["ai", "data science", "contractor", "independent"]
    ), "P2 should retain work context"
    assert any(
        w in result_lower for w in ["climbing", "climber", "rock"]
    ), "P2 should retain personal context about climbing"
    assert any(
        w in result_lower for w in ["dialectic", "reasoning", "classical"]
    ), "P2 should retain reasoning/intellectual patterns"


def test_build_combined_p1_p2() -> None:
    """Two-payload combination includes the P1→P2 transition."""
    p1 = "This is the first payload about stated preferences."
    p2 = "This is the second payload about behavioral patterns."

    result = build_combined(p1, p2)

    assert p1 in result
    assert p2 in result
    assert "Beyond stated preferences" in result
    assert "listening habits" not in result  # No P3 transition


def test_build_combined_p1_p2_p3() -> None:
    """Three-payload combination includes both transition passages."""
    p1 = "This is the first payload about stated preferences."
    p2 = "This is the second payload about behavioral patterns."
    p3 = "This is the third payload about listening habits."

    result = build_combined(p1, p2, p3)

    assert p1 in result
    assert p2 in result
    assert p3 in result
    assert "Beyond stated preferences" in result
    assert "listening habits add an implicit layer" in result


def test_save_all_payloads(tmp_path: Path) -> None:
    """Payloads are written as .txt files to the output directory."""
    payloads = {
        "p1_questionnaire": "Payload one content.",
        "p2_claude_memory": "Payload two content.",
        "p1_p2_combined": "Combined payload content.",
    }

    save_all_payloads(tmp_path, payloads)

    for name, content in payloads.items():
        path = tmp_path / f"{name}.txt"
        assert path.exists(), f"Expected file {path} to exist"
        assert path.read_text() == content


def test_load_claude_memory(tmp_path: Path) -> None:
    """load_claude_memory extracts conversations_memory from the first element."""
    expected = "This is the conversations memory content."
    data = [{"conversations_memory": expected, "project_memories": {}}]
    mem_file = tmp_path / "memories.json"
    mem_file.write_text(json.dumps(data))

    result = load_claude_memory(mem_file)
    assert result == expected
