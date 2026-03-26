"""Spotify OAuth, data fetch, and taste synthesis for the keel substrate.

P3 payload: captures implicit taste — what a person actually consumes.
The synthesis function converts raw Spotify data into flowing prose that
can be injected as context during LLM activation-capture experiments.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------

def create_oauth(
    client_id: str,
    client_secret: str,
    redirect_uri: str = "http://localhost:8888/callback",
    cache_path: Optional[str] = None,
) -> SpotifyOAuth:
    """Create a SpotifyOAuth manager with the read scopes required by keel.

    Args:
        client_id: Spotify app client ID.
        client_secret: Spotify app client secret.
        redirect_uri: OAuth redirect URI (must match app settings).
        cache_path: Optional path to the token cache file.

    Returns:
        A configured SpotifyOAuth instance.
    """
    kwargs: dict = dict(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope="user-top-read user-read-recently-played",
    )
    if cache_path is not None:
        kwargs["cache_path"] = cache_path
    return SpotifyOAuth(**kwargs)


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_spotify_data(sp: spotipy.Spotify) -> dict:
    """Fetch top artists, tracks, and aggregated genres from the Spotify API.

    Args:
        sp: An authenticated spotipy.Spotify client.

    Returns:
        A dict with keys:
          - ``top_artists``: list of ``{"name": str, "genres": list[str]}``
          - ``top_tracks``: list of ``{"name": str, "artist": str}``
          - ``top_genres``: top-10 genres by frequency across all artists
    """
    artists_response = sp.current_user_top_artists(limit=20, time_range="medium_term")
    tracks_response = sp.current_user_top_tracks(limit=20, time_range="medium_term")

    top_artists = [
        {"name": item["name"], "genres": item.get("genres", [])}
        for item in artists_response.get("items", [])
    ]

    top_tracks = [
        {"name": item["name"], "artist": item["artists"][0]["name"]}
        for item in tracks_response.get("items", [])
    ]

    genre_counter: Counter = Counter()
    for artist in top_artists:
        for genre in artist["genres"]:
            genre_counter[genre] += 1

    top_genres = [genre for genre, _ in genre_counter.most_common(10)]

    return {
        "top_artists": top_artists,
        "top_tracks": top_tracks,
        "top_genres": top_genres,
    }


# ---------------------------------------------------------------------------
# Aesthetic inference helpers
# ---------------------------------------------------------------------------

_AESTHETIC_SIGNALS: list[tuple[list[str], str]] = [
    (
        ["indie", "indie rock", "indie pop", "oxford indie"],
        "a valuing of authenticity over polish",
    ),
    (
        ["psychedelic", "psychedelic rock", "dream pop", "shoegaze"],
        "an attraction to atmospheric textures and layered sonic worlds",
    ),
    (
        ["funk", "soul", "funk rock", "r&b"],
        "a groove-oriented sensibility where rhythm and feel come first",
    ),
    (
        ["ambient", "drone", "new age"],
        "comfort with slowness and immersive, non-narrative sound",
    ),
    (
        ["electronic", "techno", "intelligent dance music", "idm", "experimental electronic"],
        "curiosity about the boundary between machine precision and human expression",
    ),
    (
        ["hip hop", "rap", "conscious hip hop", "west coast rap"],
        "an engagement with lyrical density and the storytelling power of rhythm",
    ),
    (
        ["folk", "acoustic", "chamber folk", "singer-songwriter"],
        "intimacy and stripped-back craft — the voice and the song above all else",
    ),
    (
        ["jazz", "jazz fusion", "bebop"],
        "an appreciation for improvisation and the tension between structure and freedom",
    ),
    (
        ["classical", "contemporary classical", "neo-classical"],
        "patience with long-form compositional arcs and dynamic restraint",
    ),
    (
        ["post-punk", "art punk", "new wave", "art rock"],
        "restlessness with convention and a tendency toward angular, self-aware music",
    ),
    (
        ["alternative rock", "alternative", "permanent wave"],
        "a taste for emotional directness wrapped in distortion and minor keys",
    ),
    (
        ["metal", "heavy metal", "doom metal", "black metal"],
        "an appetite for extremity and cathartic intensity",
    ),
]


def _infer_aesthetics(genres: list[str]) -> list[str]:
    """Return a list of aesthetic signal strings triggered by the given genres."""
    genre_set = {g.lower() for g in genres}
    signals: list[str] = []
    seen: set[str] = set()
    for keywords, signal in _AESTHETIC_SIGNALS:
        if any(kw in genre_set for kw in keywords) and signal not in seen:
            signals.append(signal)
            seen.add(signal)
    return signals


# ---------------------------------------------------------------------------
# Taste synthesis
# ---------------------------------------------------------------------------

def synthesize_taste_profile(data: dict) -> str:
    """Produce a prose description of a listener's taste from Spotify data.

    Targets 300-500 tokens of flowing, natural-language prose — no bullet
    points. The output is designed to be injected as LLM context during
    activation-capture experiments.

    Args:
        data: Dict as returned by :func:`fetch_spotify_data`.

    Returns:
        A multi-sentence prose string describing the listening profile.
    """
    top_artists: list[dict] = data.get("top_artists", [])
    top_tracks: list[dict] = data.get("top_tracks", [])
    top_genres: list[str] = data.get("top_genres", [])

    artist_names = [a["name"] for a in top_artists]
    track_names = [f'"{t["name"]}" by {t["artist"]}' for t in top_tracks]
    aesthetic_signals = _infer_aesthetics(top_genres)

    # ---- Build prose paragraphs ----------------------------------------

    parts: list[str] = []

    # Opening: artist landscape
    if artist_names:
        if len(artist_names) == 1:
            artist_phrase = artist_names[0]
        elif len(artist_names) == 2:
            artist_phrase = f"{artist_names[0]} and {artist_names[1]}"
        else:
            artist_phrase = (
                ", ".join(artist_names[:-1]) + f", and {artist_names[-1]}"
            )
        parts.append(
            f"This listener's recent listening gravitates heavily toward {artist_phrase}. "
            f"The selection spans a range of sounds, but a consistent sensibility runs "
            f"through each choice — these are not random discoveries but curated anchors "
            f"of a long-developed taste."
        )

    # Genre landscape
    if top_genres:
        if len(top_genres) >= 3:
            genre_phrase = (
                ", ".join(top_genres[:3])
                + (" and related territory" if len(top_genres) > 3 else "")
            )
        else:
            genre_phrase = " and ".join(top_genres)
        parts.append(
            f"Genre-wise, the profile clusters around {genre_phrase}. "
            f"These categories often overlap and bleed into one another, suggesting "
            f"someone who moves fluidly across adjacent sounds rather than staying "
            f"loyal to a single scene."
        )

    # Aesthetic signals
    if aesthetic_signals:
        if len(aesthetic_signals) == 1:
            signal_intro = f"The strongest aesthetic signal here is {aesthetic_signals[0]}."
        else:
            signal_intro = (
                "Several aesthetic threads emerge from these patterns. "
                + " ".join(
                    f"There is {s}." if i > 0 else f"There is {s},"
                    for i, s in enumerate(aesthetic_signals[:4])
                )
            )
        parts.append(signal_intro)

    # Track-level texture
    if track_names:
        sample = track_names[:3]
        track_phrase = (
            ", ".join(sample[:-1]) + f", and {sample[-1]}"
            if len(sample) > 1
            else sample[0]
        )
        parts.append(
            f"Among the most-played tracks are {track_phrase}. "
            f"These specific choices reinforce the broader picture: this is a listener "
            f"drawn to work with internal coherence — where every element earns its place."
        )

    # Synthesis / closing inference
    if aesthetic_signals and top_genres:
        closing_signals = aesthetic_signals[:2]
        if closing_signals:
            closing = (
                f"Taken together, the listening history sketches someone with a "
                f"developed and fairly specific set of values: {closing_signals[0]}. "
            )
            if len(closing_signals) > 1:
                closing += (
                    f"Alongside that runs {closing_signals[1]}, suggesting the taste "
                    f"is not monolithic but carries productive internal tensions. "
                )
            closing += (
                f"The breadth across {', '.join(top_genres[:3]) if len(top_genres) >= 3 else ' and '.join(top_genres)} "
                f"points to intellectual curiosity about sound itself, not just "
                f"comfort-seeking in familiar territory."
            )
            parts.append(closing)
    elif top_artists:
        parts.append(
            "The listening history points to a listener with a developed point of view — "
            "someone whose choices reflect accumulated aesthetic convictions rather than "
            "passive discovery."
        )

    return " ".join(parts)
