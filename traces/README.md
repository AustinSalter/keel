# Dialectic Session Traces

Standardized conversation transcripts from dialectic reasoning sessions, parsed from raw Claude Code JSONL logs into a uniform format for analysis in Sprint 2.5.

## Format

Each `coherent.jsonl` file contains one JSON object per line:

```json
{"turn": 0, "role": "user|assistant|system", "phase": "expansion|compression|critique|null", "iteration": 1, "content": "...full text..."}
```

- **turn**: Sequential 0-indexed across the whole conversation
- **role**: `user` (human input), `assistant` (model output), `system` (stop hook re-feeds)
- **phase**: Detected from `EXPANSION PASS` / `COMPRESSION PASS` / `CRITIQUE PASS` headers. `null` for non-phase turns (setup, tool results, inter-iteration messages).
- **iteration**: Which dialectic cycle (1-indexed). Each expansion pass starts a new iteration.
- **content**: Full reasoning text. Tool call metadata, file I/O, and git operations stripped. Web search results and evidence preserved.

## Sessions

| Session | Turns | Complete Iterations | ~Tokens | Thesis Mods | Source |
|---------|------:|-------------------:|--------:|:-----------:|--------|
| **thesis_geometry** | 103 | 5 | ~86K | Yes (2) | dialectic-rl project |
| **harness_thesis** | 44 | 6 | ~40K | Yes (3) | AI Study |
| **agentic_commerce** | 49 | 5 | ~49K | Yes (2) | Macro Theses |
| **kinnected** | 168 | 7 | ~124K | Yes (1) | Kinnected side project |

### thesis_geometry/
Thesis: *"When a model processes context, it establishes a reasoning geometry -- a high-dimensional manifold in activation space... If coherence to that manifold can be isolated as a signal, it can be used as an RL reward."*

5 complete iterations. Core research thesis for the keel project -- coherence-to-reasoning-manifold as RL reward signal.

### harness_thesis/
Thesis: *"The harness is the bottleneck, not the model. Current model intelligence is sufficient for asymmetric economic outcomes -- the problem is finding them. That's a search problem."*

6 complete iterations. Meta-thesis on context engineering, agentic loops as search, and why single-agent depth beats multi-agent handoffs. Longest input thesis (~4K words).

### agentic_commerce/
Thesis: *"The digital economy is at risk with agentic commerce... shifts power dynamics from platform back to brand. SEO + DTC become stronger. Search advertising takes a big hit."*

5 complete iterations. Analysis of how agentic commerce disrupts platform economics, advertising models, and consumer attention flows.

### kinnected/
Thesis: Evaluation of a coworking/meeting/event space at 1807 Centennial Blvd, San Antonio TX -- a former 58,000 SF school facility.

7 complete iterations (8 expansion passes, 1 incomplete cycle). Longest session by turn count. Real estate and business model analysis.

## Parser

`scripts/parse_dialectic_trace.py` converts raw Claude Code session JSONL into the standardized format. Handles both `# EXPANSION PASS (Iteration N)` and `## EXPANSION PASS` header formats.

## Stats

Each session directory also contains `stats.json` with machine-readable metadata (turn count, phase counts, token estimates, source file path).
