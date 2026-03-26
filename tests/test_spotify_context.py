from __future__ import annotations

import pytest

from substrate.spotify_context import synthesize_taste_profile


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_SPOTIFY_DATA = {
    "top_artists": [
        {"name": "Radiohead", "genres": ["alternative rock", "art rock", "melancholic", "oxford indie", "permanent wave"]},
        {"name": "Aphex Twin", "genres": ["ambient", "electronic", "intelligent dance music", "techno"]},
        {"name": "Kendrick Lamar", "genres": ["conscious hip hop", "rap", "west coast rap"]},
        {"name": "Nick Drake", "genres": ["folk", "acoustic", "chamber folk", "singer-songwriter"]},
        {"name": "Talking Heads", "genres": ["art punk", "new wave", "post-punk", "funk rock"]},
    ],
    "top_tracks": [
        {"name": "Exit Music (For a Film)", "artist": "Radiohead"},
        {"name": "Avril 14th", "artist": "Aphex Twin"},
        {"name": "HUMBLE.", "artist": "Kendrick Lamar"},
        {"name": "Pink Moon", "artist": "Nick Drake"},
        {"name": "Once in a Lifetime", "artist": "Talking Heads"},
    ],
    "top_genres": [
        "alternative rock",
        "electronic",
        "art rock",
        "ambient",
        "indie",
        "hip hop",
        "folk",
        "new wave",
        "funk",
        "post-punk",
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_synthesize_returns_non_empty_string() -> None:
    """synthesize_taste_profile returns a non-empty string."""
    result = synthesize_taste_profile(MOCK_SPOTIFY_DATA)
    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesize_references_source_data() -> None:
    """Output prose references at least one artist name or genre from the input."""
    result = synthesize_taste_profile(MOCK_SPOTIFY_DATA)
    result_lower = result.lower()

    artist_names = [a["name"].lower() for a in MOCK_SPOTIFY_DATA["top_artists"]]
    genres = [g.lower() for g in MOCK_SPOTIFY_DATA["top_genres"]]

    found = any(name in result_lower for name in artist_names) or any(
        genre in result_lower for genre in genres
    )
    assert found, (
        "Expected the synthesis to reference at least one artist name or genre keyword; "
        f"got:\n{result}"
    )


def test_synthesize_output_length_reasonable() -> None:
    """Output is at least 100 characters long."""
    result = synthesize_taste_profile(MOCK_SPOTIFY_DATA)
    assert len(result) > 100, (
        f"Expected output longer than 100 chars; got {len(result)} chars"
    )


def test_synthesize_output_is_prose_not_bullets() -> None:
    """Output should not be a bullet-point list (no lines starting with '-' or '*')."""
    result = synthesize_taste_profile(MOCK_SPOTIFY_DATA)
    lines = result.strip().splitlines()
    bullet_lines = [line for line in lines if line.strip().startswith(("-", "*", "•"))]
    assert len(bullet_lines) == 0, (
        f"Expected flowing prose, not bullet points. Found bullet lines:\n"
        + "\n".join(bullet_lines)
    )


def test_synthesize_with_minimal_data() -> None:
    """synthesize_taste_profile handles minimal input without crashing."""
    minimal = {
        "top_artists": [{"name": "Artist A", "genres": ["pop"]}],
        "top_tracks": [{"name": "Track 1", "artist": "Artist A"}],
        "top_genres": ["pop"],
    }
    result = synthesize_taste_profile(minimal)
    assert isinstance(result, str)
    assert len(result) > 0
