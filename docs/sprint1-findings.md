# Sprint 0-1 Findings: Substrate Geometry Experiments

## Executive Summary

Personal context injection measurably changes activation geometry in Trinity Mini (26B MoE). The SOUL document — 300 tokens of distilled identity prose — produces the most geometrically stable context injection, outperforming payloads 5x its length. PCA is the wrong decomposition for contextual geometry; CKA is the correct primary metric. Layer 24 is the diagnostic layer for over-anchoring detection.

---

## Sprint 0: Infrastructure Validation

### Code vs Philosophy Sanity Check

Two maximally different prompts (quicksort code vs Socrates philosophy) tested on both models.

**Trinity Mini Base (26B MoE):**

| Comparison | Grassmann Distance | Mean Angle | Signal/Noise |
|-----------|-------------------|-----------|-------------|
| Code vs Philosophy | **2.99** | **76°** | — |
| Code vs Code (self) | 0.19 | 2.9° | — |
| **Ratio** | — | — | **16x** |

**Qwen 2.5 7B (dense):**

| Comparison | Grassmann Distance | Mean Angle | Signal/Noise |
|-----------|-------------------|-----------|-------------|
| Code vs Philosophy | **2.93** | **73°** | — |
| Code vs Code (self) | 0.06 | 0.6° | — |
| **Ratio** | — | — | **49x** |

Different prompts produce nearly orthogonal activation subspaces (~75° mean angle). MoE architecture has higher self-comparison noise (expert routing non-determinism) but signal is still overwhelming.

### VRAM Profile

| Model | Peak VRAM (seq 4096 + hooks) | Verdict |
|-------|------------------------------|---------|
| Trinity Mini Base | 34.6 GB | Fits on A100-80GB |
| Qwen 2.5 7B | 15.6 GB | Fits comfortably |

### Trinity Mini Compatibility

Trinity Mini uses custom `afmoe` architecture requiring `transformers==4.57.6` (incompatible with 5.x due to strict config attribute access for missing `pad_token_id`). TransformerLens does not support MoE — custom `register_forward_hook` on decoder layers is the path.

---

## Layer Sweep: 32-Layer Analysis on Trinity Mini

### Three Zones in the MoE Architecture

**Zone 1: Layers 0-3 (Embedding / Dense)**
- Layer 0 has artificially perfect SNR (deterministic token lookup)
- EVR moderate (0.60-0.68) — representation hasn't crystallized
- Signal is surface features (tokenization patterns), not reasoning

**Zone 2: Layers 4-12 (The Sweet Spot)**

| Layer | Framing GD | Self Noise | SNR | EVR k=10 |
|-------|-----------|-----------|-----|----------|
| 4 | 1.58 | 0.09 | 17x | **0.85** |
| 7 | 1.67 | 0.14 | 12x | **0.88** |
| 8 | 1.70 | 0.12 | 14x | **0.87** |
| 10 | 1.62 | 0.11 | 15x | **0.87** |
| 11 | 1.62 | 0.13 | 13x | **0.88** |

- Highest EVR concentration (0.85-0.88) — model has clear low-dimensional structure
- Low MoE noise (0.09-0.14) — expert routing settled, downstream noise not accumulated
- Best SNR (12-17x)

**Zone 3: Layers 13-19 (The MoE Noise Wall)**

| Layer | Self Noise | SNR |
|-------|-----------|-----|
| 13 | **0.37** | 4.4x |
| 15 | **0.43** | 3.9x |
| 16 | **0.94** | **1.9x** |
| 18 | **0.36** | 4.8x |

Layer 16 is catastrophic — self-comparison noise of 0.94 means the same prompt twice produces nearly a full radian of rotation. MoE expert routing non-determinism concentrated in mid-network.

**Zone 4: Layers 20-31 (Recovery)**
- Noise settles (0.12-0.29), signal moderate (1.68-1.80)
- EVR drops (0.69-0.80) — representation more distributed
- CKA diagnostic signal lives here (L24)

**Selected capture layers:** {7, 11, 24}

---

## Context Injection: Geometric Effects

### Pilot Run (3 payloads × 3 prompts)

Rotation between context-only and context+prompt activations:

| Prompt | P_null GD | P1 GD | P_phello GD |
|--------|-----------|-------|-------------|
| AI/ML (personal) | 2.42 | **0.64** | **0.60** |
| Food (personal) | 2.40 | **0.97** | **0.69** |
| Control (Krebs) | 2.42 | **1.02** | **0.94** |

Without context, any prompt causes massive rotation (~2.4) from the bare system prompt. With personal context, rotation drops — context establishes a geometric frame the prompt extends rather than replaces.

### Cross-Prompt Geometry Under P_null

Different prompts (no context) produce different geometries:

| Comparison | GD | Mean Angle |
|-----------|-----|-----------|
| AI/ML vs Food | 2.85 | 70° |
| AI/ML vs Krebs | 2.96 | 74° |
| Food vs Krebs | 2.93 | 73° |

Evaluative reasoning prompts (AI/ML, Food) are more similar to each other than to factual recall (Krebs).

### Text Generation Comparison (Krebs with/without context)

- **P_null**: Clean factual answer about the Krebs cycle
- **P1**: Generates follow-up questions instead of answering (model shifts to conversational mode)
- **P_phello**: Follow-up questions then hallucinated C# code

Base model (not instruct-tuned) pattern-matches on context style rather than answering the question. Personal context for generic prompts actively degrades response quality — the over-anchoring phenomenon.

---

## Context Decomposition: Context-Derived vs Prompt-Derived Geometry

For each layer, measured what fraction of the full (context+prompt) geometry comes from context vs prompt:

| Prompt | Layer | Context→Full GD | Prompt→Full GD | Context Share |
|--------|-------|----------------|----------------|--------------|
| Krebs (generic) | L7 | 1.02 | 2.85 | **73.6%** |
| Krebs (generic) | L11 | 0.63 | 2.92 | **82.2%** |
| Krebs (generic) | L24 | 0.49 | 2.92 | **85.6%** |
| AI/ML (personal) | L7 | 0.65 | 2.78 | **80.9%** |
| AI/ML (personal) | L11 | 0.96 | 2.91 | **75.2%** |
| AI/ML (personal) | L24 | 1.04 | 2.90 | **73.7%** |

Geometry is 73-86% context-derived. But the gradient is opposite:
- **Generic prompts**: Context share INCREASES through layers (74% → 86%) — model can't escape the context frame. Over-anchoring.
- **Personal prompts**: Context share DECREASES through layers (81% → 74%) — prompt gradually blends with context. Healthy integration.

**Diagnostic signal**: If context share increases through layers → over-anchoring. If it decreases → healthy integration.

---

## EVR Collapse: PCA Is Wrong for Contextual Geometry

### Explained Variance Ratio at k=10

| Condition | L7 | L11 | L24 |
|-----------|-----|-----|-----|
| P_null + AI/ML | **0.92** | **0.92** | **0.85** |
| P_null + Food | **0.95** | **0.93** | **0.91** |
| P_null + Krebs | **0.99** | **0.99** | **0.97** |
| P1 + AI/ML | 0.37 | 0.33 | 0.30 |
| P1 + Food | 0.37 | 0.33 | 0.30 |
| P1 + Krebs | 0.37 | 0.33 | 0.30 |
| P_phello + Food | 0.33 | 0.32 | 0.27 |
| P1 context only | 0.38 | 0.34 | 0.31 |

Context injection collapses EVR from ~92% to ~33%. The substrate is genuinely high-dimensional.

### EVR Elbow Analysis (k = 10, 20, 30, 50, 100)

| k | P_null + food | P1 + food | P_phello + food |
|---|--------------|-----------|-----------------|
| 10 | **0.95** | 0.37 | 0.33 |
| 20 | **1.00** | 0.46 | 0.42 |
| 30 | 1.00 | 0.53 | 0.48 |
| 50 | 1.00 | 0.63 | 0.57 |
| 100 | 1.00 | **0.79** | **0.72** |

No elbow. Variance accumulates linearly. At k=100, still only 72-79% captured. PCA is fundamentally wrong for this signal — the substrate is not low-rank.

### Full-Dimensional Comparison (CKA)

| Comparison | L7 | L11 | L24 |
|-----------|-----|-----|-----|
| null vs P1 (food) | **0.87** | **0.84** | 0.33 |
| null vs P_phello (food) | **0.85** | **0.87** | 0.21 |
| P1 vs P_phello (food) | **0.75** | **0.75** | 0.08 |
| null vs P1 (krebs) | **0.86** | **0.85** | 0.34 |

CKA tells a different story: at L7/L11, covariance structure is largely preserved across conditions (0.75-0.87) — representations share internal organization but point in different directions. At L24, CKA drops (0.08-0.34) — late layers actually reshape the covariance structure.

**Conclusion**: CKA for measurement, PCA at k=50 for post-hoc diagnosis on interesting cases. Store raw activations.

---

## CKA Diagnostic: The Over-Anchoring Spectrum

CKA between context-only and context+prompt activations at each layer. Measures how much the prompt changes the representation established by context.

| Context | Prompt | L7 | L11 | L24 | Diagnosis |
|---------|--------|-----|-----|-----|-----------|
| **P1** (questionnaire) | ai_ml | 0.78 | 0.74 | **0.16** | weak context |
| **P1** | food | 0.78 | 0.74 | **0.16** | weak context |
| **P1** | krebs | 0.78 | 0.74 | **0.16** | weak context |
| **P2** (Claude memory) | ai_ml | 0.54 | 0.60 | **0.33** | healthy blend |
| **P2** | food | 0.53 | 0.54 | **0.33** | healthy blend |
| **P2** | krebs | 0.53 | 0.54 | **0.33** | healthy blend |
| **P_soul** | ai_ml | 0.70 | 0.65 | **0.44** | healthy blend |
| **P_soul** | food | 0.70 | 0.65 | **0.45** | healthy blend |
| **P_soul** | krebs | 0.70 | 0.65 | **0.45** | healthy blend |
| **P1+P2** | ai_ml | 0.41 | 0.45 | **0.16** | weak context |
| **P1+P2** | food | 0.38 | 0.37 | **0.15** | weak context |
| **P1+P2** | krebs | 0.38 | 0.37 | **0.15** | weak context |
| **P1+P2+P3** | ai_ml | 0.31 | 0.42 | **0.20** | weak context |
| **P1+P2+P3** | food | 0.29 | 0.42 | **0.23** | weak context |
| **P1+P2+P3** | krebs | 0.29 | 0.42 | **0.20** | weak context |
| **P_phello** (food) | food | 0.74 | 0.79 | **0.09** | CONTEXT OVERRIDE |
| **P_phello** (krebs) | krebs | 0.70 | 0.65 | **0.45** | healthy blend |

### Interpretation Scale

| L24 CKA | Meaning |
|---------|---------|
| > 0.8 | Over-anchored — prompt not reaching output layer |
| 0.3-0.6 | Healthy blend — context and prompt integrating |
| 0.1-0.3 | Weak context — context washes out by output layer |
| < 0.1 | Context override — prompt barely registers |

### Key Findings

**P_soul is the Goldilocks payload.** CKA = 0.44-0.45 at L24 for ALL prompts — AI/ML, food, and Krebs alike. Maintains structural influence uniformly without dominating. The ideal behavior for a preference passport: stable identity substrate that prompts blend into.

**P2 (Claude memory) is healthy blend (0.33).** Better than P1 (0.16) but below SOUL (0.44). Behavioral/intellectual profile maintains moderate structural influence. Like SOUL, it's prompt-agnostic.

**Combining payloads makes them WEAKER.** P1+P2 (0.15) < P2 alone (0.33). P1+P2+P3 (0.20) < P2 alone. More tokens ≠ more geometric persistence. Longer payloads dilute the identity signal.

**P_phello overrides for matched queries.** When the food-specific steering document meets the food prompt, CKA = 0.09 — the context has already specified the model's trajectory. The prompt barely registers. For mismatched queries, it falls back to SOUL-like behavior (0.45).

**Compression is the value.** 300 tokens of distilled identity prose outperforms 1620 tokens of raw context at maintaining structural influence at L24.

---

## Payload Token Counts

| Payload | Characters | ~Tokens | L24 CKA |
|---------|-----------|---------|---------|
| P_null | 0 | 0 | — |
| P1 (questionnaire) | 1,675 | ~419 | 0.16 |
| P2 (Claude memory) | 2,835 | ~709 | 0.33 |
| P3 (Spotify) | 1,842 | ~461 | — |
| P_soul | 2,280 | ~570 | **0.44** |
| P1+P2 | 4,605 | ~1,145 | 0.15 |
| P1+P2+P3 | 6,510 | ~1,620 | 0.20 |
| P_phello (avg) | ~2,800 | ~700 | 0.09-0.45 |

---

## Data Sources

| Source | What it captures | Supabase location |
|--------|-----------------|-------------------|
| Phello questionnaire | 10 Q&A pairs (travel, food, lifestyle, gifts, media) | `questionnaire_responses` |
| Phello SOUL | ~300 token identity prose | `souls` |
| Phello fragments | 9 domain-scoped prose blocks | `preference_fragments` |
| Phello signals | 44 structured preference data points | `preference_signals` |
| Claude memory | Work context, intellectual patterns, reasoning style | `Austin's Claude Profile/memories.json` |
| Spotify | Top 20 artists, 20 tracks, 10 genres (medium-term) | Spotify API via `spotipy` |

**Note**: The SOUL was generated only from questionnaire data (lifestyle/taste). It does not include Claude memory (work/intellectual) or Spotify (implicit taste). A "full SOUL" from all three sources is a Sprint 2 Track B experiment.

---

## Technical Infrastructure

- **Model**: Arcee Trinity Mini Base (26B, MoE, 128 experts, 8 active per token, Apache 2.0)
- **Compute**: Modal serverless A100-80GB (~$1.49/hr)
- **Framework**: Custom PyTorch forward hooks (TransformerLens incompatible with afmoe)
- **Capture layers**: {7, 11, 24} (0-indexed)
- **Dependencies**: transformers==4.57.6, torch, scikit-learn, scipy, h5py, modal, spotipy
- **Tests**: 26 unit tests, all passing locally (no GPU required)

---

## Sprint 2 Implications

1. **Primary metric**: CKA (context-only vs context+completion at L24)
2. **Secondary metric**: PCA at k=50 for post-hoc diagnosis
3. **Storage**: Raw activations (for PCA diagnosis) + CKA scalars + generated text
4. **Key test**: Does L24 CKA correlate with human-judged coherence? (ρ > 0.5 validates S)
5. **SOUL is the primary payload** for coherence testing
6. **Full SOUL** (all three sources) is a parallel Track B experiment
