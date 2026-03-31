# Keel

**Can a language model's activation geometry measure structural coherence — without generating or evaluating text?**

Keel investigates whether the internal representations of a transformer encode enough about conversational structure to distinguish coherent multi-turn reasoning from degraded reasoning, using a model that didn't produce the traces.

## Why it matters

Current coherence evaluation requires either human judges or LLM-as-judge pipelines — both expensive, slow, and opaque. If structural coherence has a geometric correlate in activation space, it becomes a measurable signal: useful as a reward for RL, a retrieval filter, or a drift detector for long-horizon agents.

## Key findings so far

- **Distilled identity context (SOULs) outperform raw data.** 570 tokens of compressed personal context produce stronger geometric signatures than 1,620 tokens of concatenated source material. Density per token matters, not volume.
- **Geometry tracks structural sequence.** Coherent dialectic traces show damping (oscillations converge); shuffled traces show divergence. Significant at displacement curvature (Cohen's d = −0.54, p = 0.057).
- **CKA at layer 20–24 is the informative band.** Qwen 2.5 7B's dense architecture gives clean signal where Trinity Mini's MoE routing introduced noise.
- **Plausible degradation (D2) doesn't yet separate from coherent traces.** The current metric captures sequence structure but not reasoning quality — Sprint 2.6 ablations target this gap.

## Architecture

```
substrate/          Core measurement pipeline (Modal + PyTorch)
  capture.py        Activation hooks on Qwen 2.5 7B residual stream
  payloads.py       Context payload construction (questionnaire, profile, Spotify)
  rotation.py       PCA subspace rotation + Grassmann metrics
  modal_app.py      Remote GPU orchestration

scripts/            Analysis & experimentation (~30 scripts)
traces/             Standardized dialectic reasoning transcripts (JSONL)
data/               Prompts, payloads, preference documents
notebooks/          Jupyter analysis (Sprint 2.5 results)
```

**Stack:** Python 3.11, PyTorch, Transformers, scikit-learn, Modal (serverless A100)
**Model:** Qwen 2.5 7B (dense, 28-layer) in receptive mode (no generation)
**Metrics:** CKA, cosine similarity, Grassmann distance, displacement curvature, EVR

## Roadmap

- [x] Sprint 0–1: Layer sweep, payload generation, SOUL vs raw context comparison
- [x] Sprint 2: Coherence correlation (CKA tracks frame preservation)
- [x] Sprint 2.5: Dialectic trace geometry (damping separates coherent from shuffled)
- [ ] **Sprint 2.6**: Ablations — discourse markers, reverse order, coherent-but-wrong
- [ ] Sprint 3: Germaneness separation (reasoning quality vs. structural order)
- [ ] Sprint 4: RL reward signal feasibility (geometric coherence as training signal)

## License

MIT
