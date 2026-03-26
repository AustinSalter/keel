"""Payload assembly module for the keel substrate geometry experiment.

Fetches data from Phello (Supabase), Claude profile exports, and assembles
the 8 payload variants (P1, P2, P3, P1+P2, P1+P3, P2+P3, P1+P2+P3, P_null)
used as context injections during activation capture.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import httpx

from substrate.config import PHELLO_USER_ID, SUPABASE_URL


# ---------------------------------------------------------------------------
# Data-fetching functions (Supabase REST API via httpx)
# ---------------------------------------------------------------------------

def _supabase_headers(service_key: str) -> dict[str, str]:
    """Standard headers for Supabase REST API calls."""
    return {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
    }


def fetch_questionnaire_data(service_key: str) -> dict[str, Any]:
    """Fetch questionnaire responses, preference fragments, and signals.

    Returns:
        ``{"responses": [...], "fragments": [...], "signals": [...]}``
    """
    headers = _supabase_headers(service_key)
    base = SUPABASE_URL

    with httpx.Client(timeout=30) as client:
        # Latest questionnaire responses
        r_responses = client.get(
            f"{base}/rest/v1/questionnaire_responses",
            params={
                "select": "responses",
                "user_id": f"eq.{PHELLO_USER_ID}",
                "order": "created_at.desc",
                "limit": "1",
            },
            headers=headers,
        )
        r_responses.raise_for_status()

        # All preference fragments
        r_fragments = client.get(
            f"{base}/rest/v1/preference_fragments",
            params={
                "select": "domain,subdomain,prose_body",
                "user_id": f"eq.{PHELLO_USER_ID}",
                "order": "created_at.desc",
            },
            headers=headers,
        )
        r_fragments.raise_for_status()

        # Questionnaire-sourced preference signals
        r_signals = client.get(
            f"{base}/rest/v1/preference_signals",
            params={
                "select": "domain,signal_type,value,confidence",
                "user_id": f"eq.{PHELLO_USER_ID}",
                "source": "eq.questionnaire",
            },
            headers=headers,
        )
        r_signals.raise_for_status()

    # Supabase returns an array; extract the responses object from the first row.
    responses_rows = r_responses.json()
    responses = responses_rows[0]["responses"] if responses_rows else {}

    return {
        "responses": [responses] if responses else [],
        "fragments": r_fragments.json(),
        "signals": r_signals.json(),
    }


def fetch_soul(service_key: str) -> str:
    """Fetch the latest SOUL prose_body for the Phello user.

    Returns:
        The prose_body string from the most recent soul document.
    """
    headers = _supabase_headers(service_key)

    with httpx.Client(timeout=30) as client:
        r = client.get(
            f"{SUPABASE_URL}/rest/v1/souls",
            params={
                "select": "prose_body",
                "user_id": f"eq.{PHELLO_USER_ID}",
                "order": "created_at.desc",
                "limit": "1",
            },
            headers=headers,
        )
        r.raise_for_status()

    rows = r.json()
    if not rows:
        raise ValueError("No SOUL document found for user")
    return rows[0]["prose_body"]


def fetch_phello_synthesis(service_key: str, prompt: str) -> str:
    """Call the Phello synthesize edge function.

    The edge function expects a JSON body with ``user_id`` and ``query``,
    and returns a JSON object whose ``synthesized_prose`` field contains
    the steering document text.

    Returns:
        The synthesized steering document prose.
    """
    headers = {
        **_supabase_headers(service_key),
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=60) as client:
        r = client.post(
            f"{SUPABASE_URL}/functions/v1/synthesize",
            json={"user_id": PHELLO_USER_ID, "query": prompt},
            headers=headers,
        )
        r.raise_for_status()

    data = r.json()
    return data["synthesized_prose"]


# ---------------------------------------------------------------------------
# Local data loaders
# ---------------------------------------------------------------------------

def load_claude_memory(path: Path) -> str:
    """Load ``memories.json`` and extract the ``conversations_memory`` field.

    Args:
        path: Path to the ``memories.json`` file.

    Returns:
        The raw conversations_memory string from the first element.
    """
    with open(path) as f:
        data = json.load(f)
    return data[0]["conversations_memory"]


# ---------------------------------------------------------------------------
# Payload builder functions
# ---------------------------------------------------------------------------

def build_p1(questionnaire_data: dict[str, Any]) -> str:
    """Synthesize questionnaire data into natural-flowing prose (P1 payload).

    Draws from three sources:
    - Raw Q&A responses for color and voice
    - Preference fragments for synthesized domain knowledge
    - Preference signals for specific data points

    Target: 500-800 tokens (~2000-3200 characters) of narrative paragraphs.
    NOT Q&A. NOT bullets. NOT the SOUL.
    """
    responses = questionnaire_data.get("responses", [])
    fragments = questionnaire_data.get("fragments", [])
    signals = questionnaire_data.get("signals", [])

    # ---- Extract raw material from all three sources ----

    # Gather response text by scanning all key-value pairs
    response_texts: dict[str, str] = {}
    if responses:
        raw = responses[0] if isinstance(responses[0], dict) else {}
        response_texts = {k: str(v) for k, v in raw.items()}

    # Organize fragments by domain
    fragments_by_domain: dict[str, list[str]] = {}
    for frag in fragments:
        domain = frag.get("domain", "general")
        fragments_by_domain.setdefault(domain, []).append(frag.get("prose_body", ""))

    # Organize signals by domain
    signals_by_domain: dict[str, list[dict]] = {}
    for sig in signals:
        domain = sig.get("domain", "general")
        signals_by_domain.setdefault(domain, []).append(sig)

    # ---- Build prose paragraphs ----

    parts: list[str] = []

    # Opening: who this person is broadly
    opening_lines: list[str] = []

    # Infer broad identity from responses
    lifestyle_clues: list[str] = []
    for key, val in response_texts.items():
        if "lifestyle" in key or "shift" in key:
            lifestyle_clues.append(val)
        if "decompression" in key or "decompress" in key:
            lifestyle_clues.append(val)

    opening = (
        "This person lives with intention and a developed sense of taste that extends "
        "across food, travel, fitness, and daily routine. He is someone who values craft "
        "and deliberateness — not in a precious way, but in the way of a person who has "
        "learned what he actually likes and organized his life around it."
    )

    # Enrich with lifestyle specifics
    if lifestyle_clues:
        morning_ref = ""
        for clue in lifestyle_clues:
            if "morning" in clue.lower() or "night owl" in clue.lower():
                morning_ref = (
                    " A self-described former night owl, he forced himself into mornings "
                    "and the change stuck — a detail that says something about his capacity "
                    "for deliberate reinvention."
                )
                break
        opening += morning_ref

    parts.append(opening)

    # Food and dining paragraph (draw from fragments + signals + responses)
    food_parts: list[str] = []
    food_frags = fragments_by_domain.get("food", [])
    food_sigs = signals_by_domain.get("food", [])

    # Start from raw response color
    food_response_keys = [
        k for k in response_texts if any(w in k for w in ["meal", "food", "dining", "aversion"])
    ]
    favorite_foods: list[str] = []
    aversions: list[str] = []
    dining_vibes: list[str] = []
    for k in food_response_keys:
        val = response_texts[k]
        if "aversion" in k or "allerg" in k.lower():
            aversions.append(val)
        elif "vibe" in k or "atmosphere" in k:
            dining_vibes.append(val)
        else:
            favorite_foods.append(val)

    # Extract specific signal data points
    food_prefs = [s["value"] for s in food_sigs if s.get("signal_type") == "preference"]
    food_allergies = [s["value"] for s in food_sigs if s.get("signal_type") == "allergy"]
    food_aversions = [s["value"] for s in food_sigs if s.get("signal_type") == "aversion"]

    food_text = "His relationship with food is central to how he experiences the world."
    if food_prefs:
        pref_str = " and ".join(food_prefs[:2])
        food_text += (
            f" He gravitates toward {pref_str} — choices that reflect an appetite "
            f"for quality ingredients handled with care rather than architectural plating "
            f"or trendy concepts."
        )
    if food_frags:
        # Pull the fragment's synthesized insight about dining style
        for frag_text in food_frags:
            if "chef-driven" in frag_text.lower() or "creative" in frag_text.lower():
                food_text += (
                    " He prefers chef-driven restaurants where the kitchen takes creative risks "
                    "over safe crowd-pleasers."
                )
                break
    if food_allergies or food_aversions:
        avoid_items = food_allergies + food_aversions
        avoid_str = " and ".join(avoid_items)
        food_text += (
            f" On the avoidance side, {avoid_str} are non-negotiable — "
            f"the former a genuine allergy, the latter a visceral aversion."
        )
    if dining_vibes or any("atmosphere" in f.lower() or "intimate" in f.lower() for f in food_frags):
        food_text += (
            " The ideal dining atmosphere shifts with context: buzzy and energetic "
            "with friends, intimate with great lighting for a date."
        )

    parts.append(food_text)

    # Travel paragraph
    travel_frags = fragments_by_domain.get("travel", [])
    travel_sigs = signals_by_domain.get("travel", [])
    travel_responses = [
        response_texts[k] for k in response_texts if "trip" in k or "travel" in k
    ]

    travel_text = ""
    if travel_frags or travel_sigs or travel_responses:
        travel_text = "Travel for him is not about volume of destinations but depth of experience."
        # Pull from raw response for voice
        for resp in travel_responses:
            if "central coast" in resp.lower() or "slo" in resp.lower() or "big sur" in resp.lower():
                travel_text += (
                    " A road trip along California's Central Coast — Big Sur rolling into "
                    "San Luis Obispo — stands out as a defining trip, the kind where landscape "
                    "and food scene converge into something greater than either alone."
                )
                break
        # Pull aspiration signals
        aspiration_sigs = [s for s in travel_sigs if s.get("signal_type") == "aspiration"]
        if aspiration_sigs:
            dest = aspiration_sigs[0]["value"]
            travel_text += (
                f" {dest} lives on the horizon as an aspirational destination — specifically "
                f"for the intersection of architecture, culture, and food."
            )
        elif travel_frags:
            for frag_text in travel_frags:
                if "spain" in frag_text.lower():
                    travel_text += (
                        " Spain lives on the horizon as an aspirational destination — specifically "
                        "for the intersection of architecture, culture, and food."
                    )
                    break
        parts.append(travel_text)

    # Lifestyle, decompression, and entertainment
    lifestyle_frags = fragments_by_domain.get("lifestyle", [])
    outdoor_sigs = signals_by_domain.get("outdoor", [])
    decomp_responses = [response_texts[k] for k in response_texts if "decompress" in k]
    entertainment_responses = [
        response_texts[k] for k in response_texts if "entertainment" in k
    ]
    reading_responses = [response_texts[k] for k in response_texts if "reading" in k]

    lifestyle_text = ""
    # Decompression and physical
    if decomp_responses or outdoor_sigs or lifestyle_frags:
        lifestyle_text = "His decompression patterns reveal as much as his active choices."
        climbing_ref = any(
            s.get("value", "").lower() in ("rock climbing", "climbing")
            for s in outdoor_sigs
        )
        if climbing_ref:
            lifestyle_text += (
                " Climbing is the primary physical outlet — a sport that demands "
                "presence and problem-solving in equal measure."
            )
        for resp in decomp_responses:
            if "without headphones" in resp.lower() or "quiet" in resp.lower():
                lifestyle_text += (
                    " He runs without headphones, deliberately choosing silence over stimulation."
                )
                break
        for resp in decomp_responses:
            if "mezcal" in resp.lower():
                lifestyle_text += (
                    " The evening wind-down is mezcal on the patio — unhurried, sensory, solitary."
                )
                break

    # Entertainment and intellectual consumption
    entertainment_text = ""
    ent_responses = entertainment_responses + reading_responses
    if ent_responses or lifestyle_frags:
        entertainment_text = (
            " Intellectually, he stays sharp through a mix of low-key and high-engagement pursuits."
        )
        for resp in entertainment_responses:
            resp_lower = resp.lower()
            mentions: list[str] = []
            if "crossword" in resp_lower:
                mentions.append("the NYT crossword most mornings")
            if "indie" in resp_lower and "movie" in resp_lower:
                mentions.append("indie films when he can find them")
            if "pottery" in resp_lower:
                mentions.append("pottery as a hands-on creative practice")
            if mentions:
                entertainment_text += " " + ", ".join(mentions).capitalize() + "."
                break
        for resp in reading_responses:
            resp_lower = resp.lower()
            if "unreasonable hospitality" in resp_lower:
                entertainment_text += (
                    " Unreasonable Hospitality reshaped how he thinks about service and craft"
                )
                if "tbpn" in resp_lower or "podcast" in resp_lower:
                    entertainment_text += (
                        ", and the TBPN podcast is a regular companion."
                    )
                else:
                    entertainment_text += "."
                break
        # Check for The Bear reference from lifestyle clues
        for key, val in response_texts.items():
            if "the bear" in val.lower():
                entertainment_text += (
                    " The Bear is a current favorite — a show about the collision of "
                    "craft, pressure, and care that clearly resonates."
                )
                break

    if lifestyle_text or entertainment_text:
        parts.append((lifestyle_text + entertainment_text).strip())

    return "\n\n".join(parts)


def build_p2(claude_memory_raw: str) -> str:
    """Trim Claude conversation memory into a behavioral profile (P2 payload).

    Keeps: Work context, personal context, reasoning patterns, project themes.
    Removes: "Top of mind" section, dated events, specific contact names,
    old job search details.

    Target: 500-800 tokens (~2000-3200 characters).
    """
    lines = claude_memory_raw.split("\n")

    # ---- Section-based filtering ----
    # Parse into sections delimited by **Header** lines
    sections: dict[str, list[str]] = {}
    current_section = "_preamble"
    for line in lines:
        header_match = re.match(r"^\*\*(.+?)\*\*\s*$", line.strip())
        if header_match:
            current_section = header_match.group(1).strip().lower()
            sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(line)

    # ---- Remove ephemeral / excluded sections ----
    excluded_sections = {"top of mind"}
    for excluded in excluded_sections:
        sections.pop(excluded, None)

    # ---- Reconstruct and filter lines ----
    kept_lines: list[str] = []

    for section_name, section_lines in sections.items():
        if section_name == "_preamble":
            # Skip any preamble content before the first header
            continue

        content = "\n".join(section_lines).strip()
        if not content:
            continue

        # Filter out specific exclusions within kept sections
        filtered_lines: list[str] = []
        for line in section_lines:
            line_lower = line.lower()

            # Remove dated events and specific contact names
            if any(phrase in line_lower for phrase in [
                "march madness",
                "stanford gsb",
                "gmat prep",
            ]):
                continue

            # Remove old job search specifics
            if "job search" in line_lower and "after leaving" in line_lower:
                continue
            if "targeting vc associate" in line_lower and "strategy roles" in line_lower:
                continue

            filtered_lines.append(line)

        section_content = "\n".join(filtered_lines).strip()
        if section_content:
            kept_lines.append(section_content)

    result = "\n\n".join(kept_lines)

    # ---- Clean up formatting artifacts ----
    # Remove markdown bold headers (we want flowing prose, not structured sections)
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    # Remove markdown italic markers
    result = re.sub(r"\*(.+?)\*", r"\1", result)
    # Collapse multiple blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    # ---- Trim to target length if needed ----
    # Target ~3200 chars max. If over, trim from the end of the brief history
    # section, preserving work and personal context fully.
    max_chars = 3200
    if len(result) > max_chars:
        # Try to cut at a paragraph boundary
        paragraphs = result.split("\n\n")
        trimmed: list[str] = []
        char_count = 0
        for para in paragraphs:
            if char_count + len(para) + 2 > max_chars:
                break
            trimmed.append(para)
            char_count += len(para) + 2  # +2 for the \n\n join
        result = "\n\n".join(trimmed)

    return result.strip()


def build_combined(p1: str, p2: str, p3: str | None = None) -> str:
    """Concatenate payloads with natural paragraph transitions.

    Args:
        p1: The P1 questionnaire-based payload.
        p2: The P2 Claude-memory-based payload.
        p3: Optional P3 Spotify-taste payload.

    Returns:
        Combined payload text with transition passages between sections.
    """
    transition_p1_p2 = (
        "\n\nBeyond stated preferences, his work and intellectual patterns "
        "reveal additional dimensions.\n\n"
    )

    result = p1 + transition_p1_p2 + p2

    if p3 is not None:
        transition_p2_p3 = (
            "\n\nHis listening habits add an implicit layer to this profile.\n\n"
        )
        result += transition_p2_p3 + p3

    return result


def save_all_payloads(output_dir: Path, payloads: dict[str, str]) -> None:
    """Write each payload as a ``.txt`` file to the output directory.

    Args:
        output_dir: Directory to write payload files into.
        payloads: Mapping of payload name to content string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = output_dir / f"{name}.txt"
        path.write_text(content)
