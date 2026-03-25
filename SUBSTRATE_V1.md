# Substrate Geometry V1 — Activation-Based Coherence Measurement

## One-line thesis

A model's internal activation geometry when processing personal context IS the substrate — and coherence across turns can be measured as the angular velocity of that geometry's principal components.

---

## Why this is standalone

This experiment validates the foundational primitive that everything else depends on: can we define S from model internals, and does it track human-perceived coherence? If yes, this unlocks:

- **CPO reward signal** — the coherence reward for the dialectic RL project (Phase 4 of PROJECT_PLAN.md)
- **Substrate-aligned retrieval** — a structural re-ranking primitive for Chroma/vector DBs 
- **Phello's preference passport** — empirical proof that injecting a well-crafted context payload creates a measurably different reasoning geometry (validates the entire SOUL document premise)
- **Long-horizon agentic coherence** — a drift detection signal for agents operating across many turns

None of those require each other. Each can proceed independently once S is validated. This experiment does the validation.

---

## Experimental design

### Context payloads (the personal substrate)

Three payloads representing different preference modalities, tested individually and in combination:

| Payload | Modality | What it captures | Source |
|---------|----------|------------------|--------|
| P1: Questionnaire | Explicit declarative | Who I say I am — values, goals, constraints, stated preferences | Phello onboarding answers |
| P2: Claude profile | Revealed behavioral | Who I am in practice — interaction patterns, topics, reasoning style, recurrent themes | Claude profile export (issue #5) |
| P3: Spotify OAuth | Implicit taste | What I consume — aesthetic preferences, energy patterns, genre affinities, temporal rhythms | Spotify API listening history + top artists/tracks |

**Combination payloads:**
- P1+P2: declarative + behavioral (do stated and revealed preferences create complementary geometry?)
- P1+P2+P3: full context stack (does adding implicit taste data change the substrate dimensionality?)
- P_null: no context injection (baseline — model's default activation geometry)

### Model

Arcee Trinity Mini (26B, MoE, open-weight, Apache 2.0). Reasons:

- Open weights = full access to internal activations at every layer
- MoE architecture = interesting question about whether expert routing patterns change with context
- Base model available = no RLHF plausibility bias masking structural effects
- Small enough to run activation capture experiments on a single GPU

Fallback: Qwen 2.5 7B if Trinity Mini activation capture is too memory-intensive.

### Measurements

**Primary metric: Subspace rotation between turns.**

For each (payload, prompt, turn) triple:

1. Inject payload as context → capture residual stream activations at layers {8, 16, 24, 32}
2. Generate response → re-capture activations after response
3. Run PCA on activations → extract top-k principal components (k = 5, 10, 20)
4. Compute principal angle between pre-response and post-response subspaces
5. This angle IS the substrate rotation for that turn

**What we expect:**
- Low rotation = response stayed within the activated subspace = structurally coherent
- High rotation = response shifted the model into a different region of activation space = structural drift
- Moderate rotation with consistent direction = substrate extension (adding new but structurally germane dimensions)

**Secondary metrics:**
- Expert activation patterns (which MoE experts fire with/without context — does personal context create a stable expert "team"?)
- Layer-wise PCA explained variance (at which layer does the substrate become most concentrated?)
- Cross-payload subspace overlap (do P1, P2, P3 produce overlapping or orthogonal geometries?)

### Prompts

30 prompts across 3 categories:

**Category A: Personalized reasoning (10 prompts)**
Questions where personal context should shape the reasoning frame, not just the answer.
- "What should I prioritize this quarter?"
- "How should I think about pricing for my consulting work?"
- "What's the strongest version of the case against my current approach to Phello?"

**Category B: Domain reasoning with personal lens (10 prompts)**
Questions where the topic is external but personal context should constrain the reasoning substrate.
- "What's the most important trend in AI agent infrastructure right now?"
- "Evaluate the strategic position of a seed-stage startup in preference infrastructure."
- "What are the strongest arguments for and against open-weight models?"

**Category C: Generic reasoning — control (10 prompts)**
Questions where personal context should NOT materially change the substrate.
- "Explain the Krebs cycle."
- "What causes inflation?"
- "How does TCP/IP work?"

**Hypotheses by category:**
- Category A: large geometric difference between P_null and P1/P2/P3. Low rotation within contextual turns (coherent to personal frame).
- Category B: moderate geometric difference. Context should shape but not dominate.
- Category C: minimal geometric difference. Context injection shouldn't change how the model reasons about biology or economics. If it does, that's over-anchoring — the substrate is too strong.

### Human evaluation

For each (payload, prompt) pair, generate 5 completions. Score each on:

1. **Structural coherence** (1-5): Does this response operate within the reasoning frame established by the context? Not "is it good" — "is it germane?"
2. **Substrate extension** (1-3): Does it add new dimensions to the frame (3), stay within the frame (2), or trivially restate (1)?
3. **Drift detection** (binary): At any point does the response depart from the structural frame in a way that feels like a different conversation?

3 raters minimum. Compute Krippendorff's α for inter-rater reliability.

---

## Implementation plan

### Sprint 0 — Infrastructure (3-4 days)

**Goal:** Load Trinity Mini, capture activations, verify PCA pipeline works.

- [ ] Set up environment: PyTorch + TransformerLens (or custom hook-based activation capture)
- [ ] Load Trinity Mini with activation hooks at layers {8, 16, 24, 32}
- [ ] Verify: inject a simple prompt, capture residual stream, run PCA, visualize explained variance curve
- [ ] Memory profiling: how much VRAM does activation capture at 4 layers × full sequence length require?
- [ ] If memory-constrained: downsample to layers {16, 32} or use activation checkpointing
- [ ] Build `substrate/capture.py`: standardized activation capture → PCA → basis vector extraction
- [ ] Build `substrate/rotation.py`: compute principal angles between two subspaces (using `scipy.linalg.subspace_angles`)
- [ ] Sanity check: inject two very different prompts (code snippet vs philosophical essay), verify that PCA geometries are substantially different

**Acceptance:** Can capture activations, compute PCA, measure rotation. Pipeline runs end-to-end on a single prompt pair.

### Sprint 1 — Context payload preparation (3-4 days)

**Goal:** Three clean context payloads ready for injection.

- [ ] P1 (Questionnaire): compile Phello onboarding answers into a structured context document
  - Format: natural prose, not Q&A format (model should absorb it as context, not treat it as a form)
  - Target length: 500-1000 tokens
- [ ] P2 (Claude profile): export Claude conversation memory/profile data
  - Issue #5 dependency — if export isn't available, manually compile from Claude memory descriptions
  - Format: structured summary of behavioral patterns, topics, style preferences
  - Target length: 500-1000 tokens
- [ ] P3 (Spotify): OAuth connection → pull listening history, top artists, top tracks, genre distribution
  - `substrate/spotify_context.py`: fetch data, synthesize into a preference profile document
  - Don't just dump raw data — synthesize: "Listens primarily to X, Y, Z genres. Top artists suggest preference for A aesthetic. Listening patterns peak at B times, suggesting C work style."
  - Target length: 300-500 tokens
- [ ] Combination payloads: P1+P2, P1+P2+P3, concatenated with natural transitions
- [ ] P_null: empty context (just the system prompt, no personal data)
- [ ] Write 30 prompts across categories A, B, C

**Acceptance:** 6 payload variants × 30 prompts = 180 experimental conditions defined and ready.

### Sprint 2 — Activation capture run (1 week)

**Goal:** Full experimental data collected.

- [ ] For each (payload, prompt) pair:
  - Inject payload as system/context
  - Capture activations after context processing (pre-response geometry)
  - Generate 5 completions (temperature 0.8)
  - Capture activations after each completion (post-response geometry)
  - Compute PCA at each capture point
  - Compute subspace rotation between pre and post
  - Store: activations (compressed), PCA components, rotation angles, generated text
- [ ] Total runs: 6 payloads × 30 prompts × 5 completions = 900 generations
- [ ] Estimated time: ~2-4 hours on single GPU (depends on model inference speed + activation capture overhead)
- [ ] Also capture: expert routing patterns per token (which experts fire, how stable is the expert "team" across turns?)
- [ ] Data storage: `results/activations/` (HDF5 or numpy archives), `results/generations/` (JSONL)

**Acceptance:** 900 generation runs completed with activation data captured. No OOM errors. Data integrity verified.

### Sprint 3 — Human evaluation (1 week, partially parallel)

**Goal:** Human coherence scores for all 900 generations.

- [ ] Build simple evaluation interface: show context + prompt + completion, collect ratings
  - Can be a Streamlit app or even a structured spreadsheet
  - Blind to payload condition — rater doesn't know which context was injected
- [ ] Recruit 2 additional raters (can be colleagues, friends with analytical backgrounds)
- [ ] Each rater scores all 900 completions on the 3 dimensions (coherence, extension, drift)
- [ ] Compute inter-rater reliability (Krippendorff's α)
  - Target: α > 0.6 (moderate agreement) on coherence dimension
  - If α < 0.4: rating rubric needs revision, re-calibrate with examples
- [ ] Store ratings in `results/human_eval/ratings.csv`

**Acceptance:** Complete human ratings with acceptable inter-rater reliability.

### Sprint 4 — Analysis (1 week)

**Goal:** Answer the core question: does activation geometry rotation correlate with human coherence judgment?

- [ ] Primary analysis: Spearman correlation between subspace rotation angle and human coherence score
  - Compute per-layer (does one layer correlate better than others?)
  - Compute per-k (does explained variance at k=5 vs k=10 vs k=20 matter?)
  - Compute per-category (A vs B vs C — does correlation depend on how personal the question is?)
- [ ] Payload comparison:
  - Does P1+P2+P3 create a more concentrated substrate than any individual payload?
  - Do P1, P2, P3 produce overlapping or orthogonal subspaces? (cross-payload subspace angles)
  - Does P_null (no context) produce higher rotation than any contextual payload? (baseline test)
- [ ] Category C control:
  - If context injection changes substrate geometry even for generic questions: context is over-anchoring
  - If no change: substrate is appropriately topic-sensitive (only activates for relevant queries)
- [ ] Expert routing analysis (MoE-specific):
  - Do the same experts fire consistently when personal context is present?
  - Does expert stability correlate with coherence? (stable expert team = coherent reasoning)
- [ ] Layer-wise analysis:
  - At which layer does the substrate become most defined? (PCA explained variance by layer)
  - Early layers (8): likely capture surface/syntactic features
  - Mid layers (16-24): likely capture semantic/structural frame — hypothesis: best correlation here
  - Late layers (32): likely capture output formatting — may not correlate with structural coherence
- [ ] Visualizations:
  - Rotation angle distributions by payload and category (violin plots)
  - PCA explained variance curves by layer (determines optimal k)
  - Subspace overlap matrix across payloads (heatmap)
  - Correlation scatter: rotation angle vs human coherence score (the money plot)

**Acceptance:** Clear answer to "does activation geometry track coherence?" with effect sizes, p-values, and visualizations.

---

## Decision tree after Sprint 4

```
Does rotation correlate with coherence? (ρ > 0.5)
├── YES → S is validated. Multiple downstream paths open:
│   ├── Feed S definition into CPO project (Sprint 17 of PROJECT_PLAN.md — already done)
│   ├── Build substrate-retrieval demo for Chroma pitch
│   ├── Write up as standalone paper: "Activation Geometry as Structural Coherence Metric"
│   └── Integrate into Phello: SOUL document → activation geometry → coherence monitoring
│
├── PARTIAL (0.3 < ρ < 0.5) → Signal exists but weak. Investigate:
│   ├── Is correlation stronger at specific layers? (→ use only that layer)
│   ├── Is correlation stronger for specific payload types? (→ some preference modalities matter more)
│   ├── Is PCA the wrong decomposition? (→ try ICA, sparse coding, or attention-derived basis)
│   └── Is Trinity Mini too small? (→ repeat on Trinity Large or Qwen 72B)
│
└── NO (ρ < 0.3) → Activation geometry doesn't track coherence at this scale.
    ├── Pivot: train a learned coherence classifier (RLHF-style reward model for structural fidelity)
    ├── Pivot: use attention patterns instead of residual stream (different internal signal)
    └── Pivot: content-based substrate extraction (fall back to PCA on embeddings, not activations)
```

---

## What this validates for each downstream project

| Project | What this experiment proves | What still needs work |
|---------|---------------------------|----------------------|
| **CPO (dialectic RL)** | S can be computed from model internals → coherence reward is feasible | Reward function design, training loop, DAPO integration |
| **Substrate retrieval** | Projection onto S discriminates structural relevance → re-ranking primitive works | Chroma integration, retrieval benchmark, A/B comparison vs cosine similarity |
| **Phello** | Context payload creates measurable geometric signature → SOUL document has empirical foundation | Multi-user validation, cross-model portability, privacy implications |
| **Long-horizon agents** | Rotation tracks drift → agents can self-monitor coherence | Integration with agent frameworks, threshold calibration, intervention design |
| **Transcript extraction** | Substrate construction from tangled sources → spine inference from interviews | Multi-spine graph structure, claim extraction from conversational data |

---

## Technical dependencies

- PyTorch 2.x
- Arcee Trinity Mini weights (HuggingFace, Apache 2.0)
- TransformerLens or custom activation hooks (for residual stream capture)
- scikit-learn (PCA, correlation analysis)
- scipy (subspace_angles, statistical tests)
- spotipy (Spotify OAuth + API)
- Streamlit (evaluation interface)
- matplotlib/seaborn (visualization)

---

## Timeline

| Sprint | Duration | Deliverable |
|--------|----------|-------------|
| 0: Infrastructure | 3-4 days | Activation capture + PCA pipeline working |
| 1: Payloads | 3-4 days | 6 context variants × 30 prompts ready |
| 2: Data collection | 1 week | 900 generations with activation data |
| 3: Human eval | 1 week | Coherence ratings from 3 raters |
| 4: Analysis | 1 week | Correlation results, visualizations, decision |

**Total: ~5 weeks to answer the core question.**

If the answer is yes, every downstream project has its foundational primitive validated. If no, we know exactly where to pivot before investing in CPO training or Chroma integration.

---

## References

- Anthropic, 2025 — Circuit Tracing: Revealing Computational Graphs in Language Models
- Lanham et al., 2023 (Anthropic) — Measuring Faithfulness in Chain-of-Thought Reasoning
- Arcee AI, 2026 — Trinity Large Technical Report
- Elhage et al., 2022 — Toy Models of Superposition (Anthropic, residual stream geometry)
- EMNLP 2025 — LCoT2Tree: What Makes a Good Reasoning Chain?
