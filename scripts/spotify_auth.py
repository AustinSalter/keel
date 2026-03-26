"""One-time Spotify auth + P3 payload generation."""

import argparse
from pathlib import Path

import spotipy

from substrate.spotify_context import create_oauth, fetch_spotify_data, synthesize_taste_profile


def main():
    parser = argparse.ArgumentParser(description="Spotify OAuth + generate P3 payload.")
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
