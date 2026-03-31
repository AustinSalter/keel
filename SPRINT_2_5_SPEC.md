# Sprint 2.5 — Dialectic Trace Geometry Mapping

## One-line goal

Does Qwen 2.5 7B's activation geometry distinguish coherent multi-turn reasoning from degraded reasoning when processing traces it didn't generate?

## Why this sits between Sprint 2 and 3

Sprint 2 tested whether Trinity Mini's geometry tracks coherence in its *own* generations. The result was partial — CKA correlated negatively with coherence (ρ = -0.22) because base model generation quality and structural departure are confounded: the only way Trinity produces substantive responses is by breaking away from the context geometry. This experiment removes the confound entirely: the traces are Opus-quality (no generation quality issue), and the measurement is purely receptive (the model processes, doesn't generate).

**Model switch: Qwen 2.5 7B replaces Trinity Mini.** Three reasons:

1. **No MoE noise wall.** Trinity's L13-18 noise wall (self-comparison noise 0.94 at L16, SNR 1.9x) has been a confound in every experiment. Qwen is dense — 49x SNR, 0.06 self-comparison noise, clean signal at every layer. We can measure at L16-24 where abstract reasoning representations should live, without worrying about routing non-determinism smearing the signal.

2. **Distillation advantage.** Qwen was likely trained on outputs from frontier models. Its representations have learned to *encode the structure of high-quality reasoning* even though it can't fully reproduce it generatively. For this experiment — feeding Opus traces in and asking "does your geometry distinguish coherent from degraded?" — we want a model whose representations have been shaped by reasoning-quality signal. Trinity was trained on web data from scratch.

3. **Half the VRAM** (15.6 GB vs 34.6 GB). Faster iterations, cheaper compute, more experiments per dollar.

Trinity results are preserved as an architectural comparison point — the MoE noise wall finding, layer sweep, and EVR collapse are all publishable findings about how MoE models represent context differently from dense models.

Sprint 3 needs this result to determine the canonical measurement layer and to confirm the "thermometer" principle — that a small model can measure coherence in reasoning it couldn't produce.

---

## Pre-flight: Three quick experiments before the main run

### A. Qwen 7B sanity check (30 min)

Replicate Sprint 0's code-vs-philosophy test on Qwen 2.5 7B to confirm:
- SNR ≥ 40x (expected from Sprint 0: 49x)
- Clean signal at all layers including L14 and L21 (the layers Trinity's noise wall blocked)
- EVR profile across layers (expect no noise wall — EVR should be smooth, not the collapse Trinity showed at L13-18)

If Qwen's SNR or EVR profile is unexpectedly poor, investigate before proceeding.

### B. CKA base rate experiment (30 min)

**Critical prerequisite.** Before interpreting ANY Sprint 2.5 results, you need to know whether CKA distinguishes *any* two structurally different text sequences or specifically coherent from degraded ones.

Feed two *different coherent* dialectic sessions into Qwen 7B (e.g., thesis_geometry and harness_thesis). Compute turn-by-turn CKA trajectories for both. Then compute CKA between corresponding turns of the two sessions (coherent-vs-coherent baseline).

Three possible outcomes:

- **Coherent-vs-coherent CKA is high (>0.7) while coherent-vs-degraded is low (<0.4):** CKA specifically detects structural coherence. The metric works as intended. This is the best outcome.
- **Coherent-vs-coherent CKA is similar to coherent-vs-degraded:** CKA detects text *difference*, not coherence *quality*. Any two different texts look equally "different" geometrically. The metric cannot distinguish structural quality from surface dissimilarity. Sprint 2.5's primary analysis would be uninterpretable without this baseline.
- **Coherent-vs-coherent CKA is low (<0.4):** Different coherent traces produce very different geometric trajectories. This doesn't kill the experiment — it means the analysis should focus on *trajectory smoothness* (variance of turn-to-turn CKA) rather than *absolute CKA level*. Coherent traces should have smooth trajectories regardless of their absolute CKA level; degraded traces should have jagged ones.

This takes one extra pair of forward pass sequences (~15 minutes) and prevents a false positive interpretation of the main results.

### C. Qwen scaling ladder (2-3 hours)

Run the Sprint 0 sanity check (code-vs-philosophy) on Qwen 2.5 at multiple sizes: 0.5B, 1.5B, 3B, 7B, 14B. Same test — inject code snippet, inject philosophy passage, measure Grassmann distance, self-comparison noise, SNR, and EVR at all layers.

This produces a scaling curve that answers: at what model size does the geometric signal emerge? If the signal holds down to 0.5B, the thermometer principle works with very small models. If it disappears below 3B, that constrains which models can serve as measurement instruments.

The Qwen family is ideal because all models are architecturally identical except for size — no confound from training data, architecture, or MoE-vs-dense. Any change is attributable to scale alone.

This also strengthens Claims 1-2 from the position paper: if information density determines geometric persistence across model scales (not just on Trinity Mini and Qwen 7B), the finding generalizes.

**Run this the same day as the sanity check — it's the same test at different sizes. The 14B model may need a larger Modal instance but the smaller ones run on T4.**

---

## Source material

### Coherent traces (4 sessions)

Extracted from Claude Code conversation logs into standardized JSONL format (see `traces/README.md`). All sessions capped at 5 iterations for length standardization.

1. **Thesis geometry dialectic** (abstract/theoretical) — 5 iterations, 2 thesis modifications, clear curvature detection and substrate reconstruction. Core research thesis for the keel project. Source: `~/.claude/projects/.../c93a366e-...jsonl` (1.8M).

2. **The Harness Thesis** (strategic/meta) — 6 iterations (capped to 5), 3 thesis modifications. Meta-thesis on context engineering and agentic loops as search. Longest input thesis (~4K words). Source: `~/.claude/projects/.../5d7eaf4d-...jsonl` (622K).

3. **Agentic commerce** (economics/platform dynamics) — 5 iterations, 2 thesis modifications. Analysis of how agentic commerce disrupts platform economics and advertising models. Source: `~/.claude/projects/.../9e1a9fd0-...jsonl` (831K).

4. **Kinnected** (real estate/business) — 7 iterations (capped to 5), 1 thesis modification. Coworking space evaluation at a former school facility in San Antonio. Maximally different domain from the other three — the strongest domain-invariance test. Source: `~/.claude/projects/.../dc9cec9f-...jsonl` (1.9M).

For each session, extract the turn-by-turn trace as a sequence of messages:

```
Turn 0: [system prompt + context]
Turn 1: [expansion output]
Turn 2: [compression output]
Turn 3: [critique output + decision]
Turn 4: [iteration 2 expansion]
...
Turn N: [final synthesis / conclude]
```

### Degraded traces (3 variants per coherent session, 12 total)

For each coherent session, construct three degraded versions that preserve surface plausibility but break structural coherence:

**D1: Shuffled turns.** Take the same turns, randomize their order (except turn 0 stays as context). The content is identical — same quality prose, same evidence, same vocabulary. But the structural progression is destroyed. Expansion after critique, compression before evidence gathering. If geometry tracks structure, this should show jagged CKA breaks where the coherent trace shows smooth evolution.

**D2: Plausible substitution.** Replace one key turn (ideally the elevation or a critical critique) with a plausible-but-drifting alternative. Use Opus to generate a response to the same prompt that sounds good but doesn't actually follow from the previous turn's evidence. This tests whether Qwen's geometry can detect a *local* coherence break — one turn that departs from the manifold while the rest stays on it.

**D3: Dilutory substitution.** Replace one key turn with a restating, hedging, non-committal version that stays on-topic but adds nothing structural. Use Opus with instructions like "respond to this without taking a position or advancing the argument." This tests whether Qwen's geometry distinguishes load-bearing from dilutory contributions — the hardest case.

---

## Measurement protocol

### For each trace (coherent + 3 degraded × 4 sessions = 16 traces):

1. Feed turn 0 (context) into Qwen 2.5 7B
2. Capture activations at layers {7, 14, 21, 27}
   - L7: early representation (baseline, should encode surface structure)
   - L14: mid-network (no noise wall in dense model — this is where Trinity was blind)
   - L21: deep processing (abstract reasoning representations should concentrate here)
   - L27: near-output (closest to generation, most transformed from input)
   - Note: Qwen has 28 layers. Dense architecture means clean signal at every depth — we can finally test the full layer stack without MoE confounds.
3. Feed turn 0 + turn 1, capture activations at same layers
4. Feed turn 0 + turn 1 + turn 2, capture activations
5. Continue through all turns, capturing at each accumulation point

This gives an activation trajectory: a sequence of geometric snapshots as the conversation builds.

### Compute per trace:

- **Turn-to-turn CKA at each layer**: CKA(activations after turn N, activations after turn N+1). This is the geometric "smoothness" of the conversation's evolution at each layer.
- **Cumulative CKA drift from turn 0**: CKA(activations after turn 0, activations after turn N) for all N. This shows how far the geometry has evolved from the starting context.
- **EVR at each turn**: explained variance ratio of top-k PCA at each accumulation point. Does the representation concentrate (substrate sharpening) or diffuse (substrate dilution) over turns?
- **Turn-to-turn Grassmann distance at k=50**: for comparison with CKA. Captures subspace rotation magnitude.

### Total compute estimate:

**Pre-flight:**
- A. Qwen 7B sanity check: ~30 min, single A10G
- B. CKA base rate (coherent-vs-coherent): ~15 min additional (2 coherent traces, same infrastructure)
- C. Qwen scaling ladder (0.5B/1.5B/3B/7B/14B × code-vs-philosophy): ~2-3 hours, T4 for small models, A10G for 14B

**Main experiment:**
- 4 sessions × 4 variants × ~15 turns × 4 layers = ~960 forward passes
- Each pass: context grows (turn 0 is ~500 tokens, by turn 10 it's ~5000+ tokens)
- Qwen 2.5 7B at 15.6 GB VRAM — runs on a single A10G or even T4
- Estimate: 1-2 hours on Modal

**Total: ~4-5 hours Modal compute, ~$15-25**

---

## Analysis

### Primary question: Does turn-to-turn CKA distinguish coherent from degraded traces?

**Prerequisite: the base rate comparison must be analyzed FIRST.** Before interpreting coherent-vs-degraded, establish the coherent-vs-coherent baseline from pre-flight B. All separation findings below are meaningful only if coherent-vs-degraded separation exceeds the coherent-vs-coherent baseline.

For each layer, plot the turn-to-turn CKA trajectory:

```
Turn:    0→1   1→2   2→3   3→4   4→5   ...
Coherent: 0.7   0.6   0.5   0.4   0.6   (smooth evolution, dip at elevation, recovery)
Shuffled: 0.7   0.2   0.8   0.1   0.5   (jagged, unpredictable breaks)
Plausible: 0.7   0.6   0.5   0.1   0.5  (smooth then sudden break at substitution)
Dilutory: 0.7   0.6   0.8   0.5   0.6   (spike at dilutory turn — too close, didn't depart)
```

Expected CKA sensitivity ordering (from the research agenda dialectic): D1 (shuffled) most detectable > D3 (dilutory) > D2 (plausible substitution) least detectable. Rationale: CKA is a global averaging metric — shuffled turns produce wholesale structural disruption (easy to detect), dilutory turns produce anomalous stasis (detectable as an outlier spike), but plausible substitutions that share surface-level structure with the real turn may produce CKA values within the normal range. If D2 IS distinguishable, that's the strongest evidence for geometric germaneness detection.

Hypotheses:
- Coherent traces: smooth CKA trajectory with gradual evolution, moderate departures at each turn, possible dip at elevation points followed by recovery
- Shuffled: high variance, no smooth trend, CKA jumps unpredictably
- Plausible substitution: smooth except for one sharp break at the substituted turn
- Dilutory substitution: CKA spike at the substituted turn (too high — stayed too close, didn't extend)

### Dual outcome interpretation (from research agenda dialectic)

Sprint 2.5 has TWO positive outcome paths, not one:

**Path A — Thermometer works.** Coherent traces produce measurably smoother geometric trajectories than degraded traces. CKA trajectory variance is the discriminating feature. This validates the thermometer principle and opens the germaneness separability experiment (Claim 5).

**Path B — Coherence-control tradeoff confirmed.** If the Sprint 2 negative correlation (stronger CKA departure = lower coherence) REPLICATES on Qwen with Opus traces — meaning even externally generated coherent reasoning shows geometric tension between preference-frame adherence and structural extension — then this is NOT a measurement failure. It's the first CKA-based measurement of the coherence-control tradeoff documented in the activation steering literature. This connects to a larger, well-funded research thread (IDS, FGAA, CAST) and produces a different but equally publishable paper: "Geometric Characterization of the Coherence-Preference Tradeoff in Context-Injected LLMs."

**The only failure is no signal at all** — if coherent and degraded traces produce indistinguishable CKA trajectories AND no systematic correlation between CKA and coherence quality in either direction.

### Layer comparison: where does the signal live?

With Qwen's dense architecture, we can finally test the full depth hypothesis without MoE confounds:

- If L7 separates best: coherence is a surface/representation-level signal, readable from how the model encodes the input
- If L14 separates best: coherence lives at mid-depth where semantic integration happens — the region Trinity's noise wall made unmeasurable
- If L21 separates best: coherence requires deep processing, abstract relational reasoning
- If L27 separates best: coherence is a near-output property, close to the generation distribution
- If separation is uniform across layers: coherence is a global property of the representation, not localized to a specific processing depth

The mid-layer result (L14) would be the most important finding — it would mean Trinity's noise wall was actively destroying the coherence signal at exactly the depth where it's strongest, explaining why Sprint 2's Trinity correlations were weak.

### Abstract vs concrete domain comparison:

- If separation is equally strong across domains at the same layer: coherence signal is domain-invariant (strongest finding for CPO generalization)
- If abstract traces show stronger separation at deeper layers (L21/L27) while concrete traces separate at shallower layers (L7/L14): coherence measurement needs to be layer-adaptive based on prompt complexity — a finding with direct implications for the CPO reward design

### EVR trajectory:

- Coherent sessions: EVR should evolve smoothly — possibly increasing (substrate sharpening as the argument focuses) or decreasing then recovering (substrate expansion during exploration, concentration during compression)
- Degraded sessions: EVR should be erratic or monotonically decreasing (substrate never concentrates because the argument doesn't build)

---

## What this unlocks

### Pre-flight findings (available before main experiment):

5. **Scaling curve**: at what model size does the geometric signal emerge? If code-vs-philosophy separation holds at 0.5B, the thermometer principle works with models small enough to run on a phone. If it requires 7B+, the measurement instrument needs real compute. Either way, this is the first published scaling curve for activation geometry across a controlled model family.

6. **Base rate calibration**: the coherent-vs-coherent CKA baseline determines how to interpret every subsequent result. This is the zero point of the measurement — without it, all separation findings are ambiguous.

### If Path A (thermometer works — coherent traces geometrically distinguishable):

1. **CPO reward design confirmed**: the coherence signal exists at the representation level, independent of generation quality. The reward can be computed from geometry.

2. **Measurement layer determined**: whichever layer best separates coherent from degraded becomes the canonical measurement layer for all downstream work (Sprint 3 human eval, CPO reward, substrate-aligned retrieval).

3. **Cross-model measurement validated**: if Qwen 2.5 7B's geometry tracks Opus-quality reasoning, you can use small dense models as coherence measurement instruments for any model's output. This is the "thermometer doesn't need to be hot" principle confirmed — and the distillation hypothesis validated (models trained on frontier outputs learn to *represent* reasoning quality they can't *generate*).

4. **Dense vs MoE comparison**: combined with Trinity's Sprint 0-2 results, you'll have a direct comparison of how dense and MoE architectures encode coherence. If Qwen shows clean coherence signals at L14 where Trinity's noise wall destroyed them, that's a publishable architectural finding about MoE limitations for geometric measurement.

### If Path B (coherence-control tradeoff confirmed — negative correlation replicates):

1. **Sprint 2's result reframed as a positive finding**: the negative correlation isn't a measurement error — it's the first CKA-based measurement of the coherence-control tradeoff in context-injected LLMs. This connects to the well-funded RepE literature (IDS, FGAA, CAST).

2. **Different paper, equally publishable**: "Geometric Characterization of the Coherence-Preference Tradeoff in Context-Injected LLMs." Experiments: vary SOUL injection strength, map the Pareto frontier of coherence vs preference adherence, compare context injection tradeoff curve with direct activation steering.

3. **Product implication for Phello**: the preference infrastructure becomes a "coupling dial" — Phello's value is in navigating the tradeoff optimally, not in eliminating it. The SOUL's CKA 0.44 may already be near the Pareto-optimal point for the coherence-preference frontier.

4. **Germaneness question redirected but not killed**: if coherence and preference adherence trade off, germaneness may still be measurable as the *angle* of departure (on-manifold vs off-manifold) rather than the *magnitude* of departure (which is what CKA captures). This motivates a different metric exploration rather than abandoning the concept.

### If traces are NOT distinguishable:

1. If Qwen with 49x SNR and no noise wall still can't distinguish coherent from degraded traces, the problem is not architectural — CKA may fundamentally not capture the kind of structural coherence the dialectic produces. Try attention pattern comparison or directed metrics that capture argumentative direction (the link turn problem).
2. The coherence signal may be generation-dependent (only measurable in the model's own outputs, not in processed inputs) — would mean the "thermometer" model must be the same as the generating model, which limits the architecture but doesn't kill the CPO thesis.
3. The degraded traces may be too close to the coherent ones — surface plausibility may share enough geometric structure with genuine coherence that CKA can't separate them. This would mean coherence is a *semantic* property not a *geometric* one, which would redirect the project toward learned coherence classifiers rather than geometric measurement.

### Dialectic operation signatures (exploratory analysis, not a gate):

Regardless of the primary result, check whether dialectic operations have characteristic geometric signatures in the coherent traces:
- Expansion: EVR spread increases (exploring the manifold)
- Compression: EVR concentrates (building the tangent plane)
- Critique: CKA dip between turns (curvature detection — representation shifts as the model processes a challenge)
- Elevation: CKA break followed by recovery at a different level (substrate reconstruction)

If visible, this geometrically validates the dialectic loop — each operation does something structurally distinct to the activation geometry. This is the Substack finding regardless of the primary result.

---

## Deliverables

- `scripts/trace_geometry.py`: processes dialectic sessions, computes turn-by-turn geometric trajectories
- `scripts/scaling_ladder.py`: runs code-vs-philosophy on Qwen 0.5B through 14B, produces scaling curve
- `results/preflight/qwen_sanity.json`: SNR, EVR, Grassmann distance at all layers for Qwen 7B
- `results/preflight/base_rate.json`: coherent-vs-coherent CKA comparison (the interpretability prerequisite)
- `results/preflight/scaling_ladder.json`: SNR, EVR, signal strength across Qwen model sizes
- `results/traces/`: per-session trajectory data (CKA, Grassmann, EVR at each turn × each layer)
- `results/traces/analysis.md`: layer comparison, coherent vs degraded separation, domain comparison, base rate comparison, dual-outcome assessment (Path A vs Path B)
- Visualizations: trajectory plots (CKA over turns, colored by condition), layer comparison heatmaps, EVR evolution curves, scaling ladder plot (SNR and signal strength vs model size)

---

## Timeline

| Day | Activity | Compute |
|-----|----------|---------|
| 1 morning | Trace preparation: truncate to 5 iter, format for Qwen chat template, build degraded variants for all 4 sessions | Opus API (~$2) |
| 1 afternoon | Pre-flight A+B: Qwen 7B sanity check + CKA base rate experiment | 45 min Modal |
| 1 evening | Pre-flight C: Qwen scaling ladder (0.5B → 14B) | 2-3 hours Modal |
| 2 morning | Analyze pre-flight results. Go/no-go: if Qwen 7B SNR < 20x or base rate shows CKA can't discriminate, stop and diagnose. If scaling ladder shows signal vanishes below 3B, note for paper but proceed with 7B. | None |
| 2 afternoon | Main experiment: run all 16 traces through Qwen 7B | 1-2 hours Modal |
| 3 | Analysis + visualization: trajectory plots, layer comparison, base rate comparison, dual-outcome assessment, dialectic operation signatures, domain comparison | None |

**Total: 3 days, ~$15-25 Modal compute**
