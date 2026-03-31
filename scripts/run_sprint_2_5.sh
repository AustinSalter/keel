#!/bin/bash
# Sprint 2.5 full pipeline runner
# Run from keel project root: ./scripts/run_sprint_2_5.sh
set -e

export PYTHONUNBUFFERED=1
VENV=".venv/bin/python"
LOG="results/sprint_2_5_run.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

mkdir -p results/preflight results/traces results/figures

log "=== Sprint 2.5 Pipeline Starting ==="

# Pre-flight A: Sanity check
if [ ! -f results/preflight/qwen_sanity.json ]; then
    log "Step 2b: Pre-flight A — Qwen 7B sanity check"
    $VENV scripts/preflight_sanity.py 2>&1 | tee -a "$LOG"
    log "Pre-flight A complete"
else
    log "SKIP Pre-flight A (results exist)"
fi

# Pre-flight B: Base rate
if [ ! -f results/preflight/base_rate.json ]; then
    log "Step 2c: Pre-flight B — CKA base rate"
    $VENV scripts/preflight_base_rate.py 2>&1 | tee -a "$LOG"
    log "Pre-flight B complete"
else
    log "SKIP Pre-flight B (results exist)"
fi

# Pre-flight C: Scaling ladder
if [ ! -f results/preflight/scaling_ladder.json ]; then
    log "Step 3: Pre-flight C — Scaling ladder"
    $VENV scripts/scaling_ladder.py 2>&1 | tee -a "$LOG"
    log "Pre-flight C complete"
else
    log "SKIP Pre-flight C (results exist)"
fi

# Pre-flight analysis
log "Step 4: Pre-flight analysis"
$VENV scripts/analyze_preflight.py 2>&1 | tee -a "$LOG"

# Go/no-go check
if grep -q "NO-GO" results/preflight/analysis.md 2>/dev/null; then
    log "NO-GO: Pre-flight failed. Check results/preflight/analysis.md"
    exit 1
fi

# Main experiment
log "Step 5: Main experiment — 16 traces"
$VENV scripts/run_trace_experiment.py 2>&1 | tee -a "$LOG"
log "Main experiment complete"

# Analysis
log "Step 6: Analysis + visualization"
$VENV scripts/analyze_traces.py 2>&1 | tee -a "$LOG"

log "=== Sprint 2.5 Pipeline Complete ==="
log "Results: results/traces/analysis.md"
log "Figures: results/figures/"
