# Sprint 1: Context Payloads & Prompt Design

## Purpose

Prepare 8 context payloads and 30 prompts for the substrate geometry experiment. The payloads represent different modalities and compression levels of personal context. The prompts span 5 interest categories to test where personal context creates measurable geometric signal vs where it doesn't.

## Data Sources

All personal data is already available:

| Source | Location | Format |
|--------|----------|--------|
| Phello questionnaire (10 Q&A) | Supabase `questionnaire_responses` (user `d09855bd`) | JSONB array |
| Phello SOUL document | Supabase `souls` | Prose (~300 tokens) |
| Phello preference fragments | Supabase `preference_fragments` (9 domains) | Domain-scoped prose |
| Phello preference signals | Supabase `preference_signals` (44 signals) | Structured data |
| Claude conversation memory | `Austin's Claude Profile/memories.json` | JSON with prose sections |
| Spotify listening data | Spotify API (OAuth required) | Top artists, tracks, genres |

Supabase service role key required for queries (stored in phello-mobile `.env.local`).

## Payloads (8 variants)

### P_null — No context baseline
- Empty string. Model sees only the prompt.
- Purpose: establish the default activation geometry.

### P1 — Questionnaire (explicit declarative)
- Synthesize the 10 Q&A responses into natural prose.
- NOT Q&A format. NOT the SOUL verbatim. A purpose-built context document: "Here is who this person is."
- Draw from raw answers, preference fragments, and signals to create a rich but natural-sounding profile.
- Target: 500-800 tokens.
- Source: Supabase `questionnaire_responses` + `preference_fragments` + `preference_signals` for Austin's user ID.

### P2 — Claude memory (revealed behavioral)
- Extract the `conversations_memory` section from `memories.json`.
- Trim to 500-800 tokens. Focus on: work context, intellectual patterns, reasoning style, project themes.
- Remove ephemeral details (specific upcoming calls, dated events).
- This captures "who he is in practice" — interaction patterns, recurrent themes, how he thinks.

### P3 — Spotify (implicit taste)
- OAuth connection to Spotify API.
- Pull: top artists (short/medium/long term), top tracks, genre distribution, recently played.
- Synthesize into a taste profile document: "Listens primarily to X genres. Top artists suggest Y aesthetic. Listening patterns suggest Z energy/work style."
- Target: 300-500 tokens.
- Implementation: `substrate/spotify_context.py` using `spotipy` library.

### P_soul — SOUL document only
- The Phello SOUL document verbatim from Supabase `souls` table.
- ~300 tokens of distilled identity prose.
- Tests: does a maximally compressed identity document carry geometric signal?

### P_phello — Full Phello MCP pipeline
- For each prompt, call the Phello MCP's synthesis pipeline:
  1. Route fragments by query domain (Claude Haiku classifies into 1-3 domains)
  2. Select fragments within token budget (~400 tokens)
  3. Synthesize SOUL + selected fragments + query into a steering document
- This payload is **query-dependent** — different prompts get different steering documents.
- Implementation: call the Supabase `synthesize` edge function directly, or replicate the routing + synthesis logic locally.
- Target: 400-600 tokens per prompt.

### P1+P2 — Declarative + behavioral combined
- Concatenate P1 and P2 with a natural transition.
- Target: 1000-1600 tokens.

### P1+P2+P3 — Full context stack
- Concatenate P1, P2, and P3 with natural transitions.
- Target: 1300-2100 tokens.

## Prompts (30 total, 5 categories x 6 prompts)

### Category 1: AI/ML & Preference Infrastructure
Personal context should strongly shape reasoning.

1. (A) "What's the strongest version of the case against building Phello as a standalone company vs integrating into an existing AI platform?"
2. (A) "If you had to pick one metric to prove substrate geometry works to a skeptical investor, what would it be and why?"
3. (B) "What's the most important unsolved problem in AI agent memory and personalization?"
4. (B) "Evaluate the strategic position of preference infrastructure as a category — is this a platform or a feature?"
5. (B) "What are the strongest arguments for and against open-weight models for activation geometry research?"
6. (B) "How should context injection for AI agents evolve over the next two years?"

### Category 2: Strategic Reasoning & Entrepreneurship
Personal context should shape the frame but not dominate the domain knowledge.

7. (A) "What should I prioritize this quarter — research validation, product development, or fundraising?"
8. (A) "What's the most honest assessment of my strengths and blind spots as a solo technical founder?"
9. (B) "What separates the founders who successfully navigate the pre-revenue phase from those who don't?"
10. (B) "How should an early-stage startup think about pricing for a developer infrastructure product?"
11. (B) "What's the strongest case for bootstrapping vs raising a seed round for a developer tools company?"
12. (B) "What role does conviction play in early-stage company building, and when does it become a liability?"

### Category 3: Food, Hospitality & Taste
Personal context should heavily shape both the reasoning frame and specific recommendations.

13. (A) "Plan a 5-day food-focused trip to a city I haven't been to but would love."
14. (A) "What's a dinner party menu that would feel authentically me — both the food and the vibe?"
15. (B) "What makes a restaurant experience feel genuinely personal rather than just well-executed?"
16. (B) "How is the relationship between chef and diner changing in the age of social media and AI-driven recommendations?"
17. (B) "What can hospitality teach the tech industry about building products people actually love?"
18. (B) "What distinguishes a great food city from a good one?"

### Category 4: Outdoor & Physical Practice
Personal context should shape the frame; climbing/surfing knowledge is domain.

19. (A) "Design a 12-week training block for breaking into the next grade in both climbing and surfing."
20. (A) "What outdoor trip would push me the most this year — and why that one specifically?"
21. (B) "What does the science say about the relationship between physical practice and creative thinking?"
22. (B) "How should someone balance structured training with unstructured outdoor play?"
23. (B) "What makes climbing culture different from other outdoor sports, and what does it get right?"
24. (B) "What's the case for analog, physical hobbies as essential counterweights to knowledge work?"

### Category 5: Generic Control
Personal context should NOT change the activation geometry. If it does, that's over-anchoring.

25. (C) "Explain the Krebs cycle and why it's central to cellular metabolism."
26. (C) "What causes inflation, and what are the main schools of thought on how to control it?"
27. (C) "How does TCP/IP work, and what makes it so resilient?"
28. (C) "Explain the basic principles of orbital mechanics."
29. (C) "What is the significance of Godel's incompleteness theorems?"
30. (C) "How do vaccines work at the cellular level?"

## Experimental Hypotheses by Category

- **Categories 1-2**: Large geometric difference between P_null and context payloads. Low rotation within contextual turns (responses stay coherent to the personal frame).
- **Category 3**: Largest geometric shift — food/taste is deeply personal and the context is rich.
- **Category 4**: Moderate shift — physical practice has personal context but also strong domain knowledge.
- **Category 5**: Minimal geometric difference. Context injection shouldn't change how the model reasons about biology or economics. If it does, that's over-anchoring.

## Key Comparisons

| Comparison | What it answers |
|------------|-----------------|
| P_null vs P1 | Does explicit declarative context create geometric signal? |
| P_null vs P2 | Does behavioral/revealed context create geometric signal? |
| P_null vs P3 | Does implicit taste data create geometric signal? |
| P1 vs P2 | Are declarative and behavioral preferences geometrically different? |
| P_soul vs P_phello | Does Phello's routing + synthesis add signal beyond raw SOUL? |
| P_soul vs P1 | Is distilled identity (300 tokens) as effective as verbose context (800 tokens)? |
| P1+P2 vs P1 alone | Does adding behavioral data to declarative create complementary geometry? |
| P1+P2+P3 vs P1+P2 | Does implicit taste data change substrate dimensionality? |

## Layer Configuration (from Sprint 0 layer sweep)

Based on the 32-layer sweep on Trinity Mini, capture at layers **{7, 11, 24}**:
- Layers 7, 11: in the optimal zone (high EVR 0.87+, low MoE noise, SNR 12-18x)
- Layer 24: post-noise-wall recovery zone for comparison

This reduces data collection from 4 layers to 3, saving ~25% compute.

## Sprint 2 Impact

8 payloads x 30 prompts x 5 completions x 3 layers = **3,600 capture points** (up from 900 in the original spec due to more payloads, fewer layers).

Estimated Modal compute: ~4-6 hours on A100, ~$6-9.

## Design Decisions & Known Issues

### Scope change from original spec
The original Sprint 1 spec called for 6 payloads x 30 prompts = 180 conditions. This design expands to 8 payloads (adding P_soul and P_phello) for 240 conditions. Rationale: the P_soul vs P_phello comparison directly tests Phello's synthesis value — the most commercially relevant question in the experiment.

### P_phello execution strategy
P_phello is query-dependent (different steering document per prompt). Two options:
- **Option A (recommended)**: Pre-generate all 30 P_phello payloads locally before Sprint 2 by calling the Supabase `synthesize` edge function. Save as static files. This avoids runtime dependencies on Supabase during the Modal GPU run.
- **Option B**: Call the edge function live during Sprint 2. Risk: rate limits, cold starts, edge function failures mid-run.

Decision: **Option A**. Generate and cache all P_phello payloads as part of Sprint 1, stored in `data/payloads/phello/`.

### Spotify OAuth for headless execution
Spotify OAuth requires a browser redirect. Approach:
1. Run the initial OAuth flow locally (browser-based, one-time).
2. Store the refresh token in `data/.secrets/spotify_refresh_token` (gitignored).
3. `spotify_context.py` uses the refresh token to get an access token programmatically.
4. The Spotify payload (P3) is generated once and saved as a static file — no Spotify API calls needed during Sprint 2 Modal runs.

### Generation capture (post-response activations)
The V1 experiment design requires capturing activations both BEFORE and AFTER model generation to compute subspace rotation. The current `capture.py` only handles pre-response (forward pass of context+prompt). Post-response capture requires hooking into autoregressive decoding (model.generate), which is a different code path. This is **Sprint 2 engineering work**, not Sprint 1. Sprint 1 delivers the payloads and prompts; Sprint 2 extends the capture pipeline for generation.

### Storage estimate
3,600 capture points at 3 layers. Trinity Mini hidden_size=2048 (not 3072 — checked config.json). Average sequence ~600 tokens. Per capture: 600 x 2048 x 4 bytes (float32) = ~4.7 MB. Total raw: 3,600 x 3 x 4.7 MB ≈ **50 GB**. With HDF5 gzip compression (~4x): **~12 GB**. Manageable on local disk. Consider storing only PCA components (not raw activations) for the full run, reducing to ~500 MB.

### P1 source composition
P1 draws from questionnaire responses, preference fragments, AND preference signals. This is intentionally richer than P_soul (which is pure identity prose). The P_soul vs P1 comparison is therefore confounded by both compression level AND source representation. This is acceptable — the question isn't "which source is better" but "does more context create more geometric signal." If a clean compression-only comparison is needed, a future experiment can hold source constant.

### Category 3 food coverage
The Phello questionnaire is heavily food-focused (wagyu/ribeye preferences, dining atmosphere, dietary restrictions including shellfish allergy, truffle oil aversion, Japanese food culture, hospitality identity). The `dining`, `dietary`, and `decompression` fragments all contain food data. P1 will include food preferences. Category 3 prompts are well-supported.

## Implementation Tasks

1. Build `substrate/spotify_context.py` — Spotify OAuth + data pull + synthesis
2. Build `substrate/payloads.py` — payload assembly (query Supabase, load Claude memory, compose all 8 variants)
3. Write `data/prompts.json` — all 30 prompts with category metadata
4. Write `data/payloads/` — save all static payloads as text files
5. Pre-generate P_phello payloads — call synthesize edge function for each of the 30 prompts, save to `data/payloads/phello/`
6. Update `substrate/config.py` — new layer indices {7, 11, 24} (acceptance criterion: verified before Sprint 2)
7. Verify token counts for all payloads before saving
8. Validate: pilot run with P_null, P1, P_phello against prompts from categories 1, 3, and 5 (one each)
