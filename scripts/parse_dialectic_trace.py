#!/usr/bin/env python3
"""
Parse raw Claude Code JSONL session files into standardized dialectic trace format.

Output format (one JSON object per line):
{"turn": 0, "role": "user|assistant|system", "phase": "expansion|compression|critique|null", "iteration": 1, "content": "..."}

Rules:
- Strips tool call metadata, file reads/writes, git operations
- Keeps reasoning content only
- Detects phases from EXPANSION PASS / COMPRESSION PASS / CRITIQUE PASS markers
- Splits multi-phase assistant turns into separate entries
- Concatenates consecutive assistant text chunks into single turns
- Keeps stop hook re-feeds as role: "system"
- Includes web search results and evidence
"""

import json
import re
import sys
from pathlib import Path


def extract_text_content(message: dict) -> str:
    """Extract text content from a message, stripping tool calls and thinking blocks."""
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    # Keep tool results that contain substantive content (web search, evidence)
                    # but skip file read/write results
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        # Skip file operation results
                        if any(skip in result_content for skip in [
                            "File content (", "Added ", "lines to", "Task #",
                            "Updated task", "Created file", "Wrote file",
                            "tool_use_error", "successfully", "No files found"
                        ]):
                            continue
                        # Keep substantive tool results (web search, evidence)
                        if len(result_content) > 50:
                            text_parts.append(result_content)
                    elif isinstance(result_content, list):
                        for sub in result_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                sub_text = sub.get("text", "")
                                if len(sub_text) > 50 and not any(skip in sub_text for skip in [
                                    "File content (", "Added ", "Task #", "Updated task",
                                    "tool_use_error"
                                ]):
                                    text_parts.append(sub_text)
                # Skip: thinking, tool_use, image blocks
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts)

    return str(content) if content else ""


def is_tool_noise(text: str) -> bool:
    """Check if text is purely tool operation noise with no reasoning content."""
    if not text.strip():
        return True

    noise_patterns = [
        r"^Let me (read|check|create|write|update|initialize|look)",
        r"^I'll (read|check|create|write|start|initialize|set up)",
        r"^Starting dialectic reasoning session",
        r"^Now (let me|I'll|reading|writing|checking)",
        r"^\[Request interrupted",
        r"^Reading the",
        r"^Checking (if|for|the)",
    ]

    stripped = text.strip()
    # Short messages that are just status updates
    if len(stripped) < 100:
        for pattern in noise_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True

    return False


def is_meta_injection(text: str) -> bool:
    """Check if this is a skill file injection (dialectic protocol, phase skills, etc.).

    Handles both raw text and line-numbered format (e.g., '  1→# Compression Pass').
    """
    # Strip line number prefixes for matching (format: '  123→content')
    stripped = re.sub(r'^\s*\d+→', '', text, flags=re.MULTILINE)

    # Main dialectic protocol
    if "# Dialectic Reasoning" in stripped and "You are executing a multi-pass dialectic" in stripped:
        return True
    if "## Step 1: Initialize or Resume" in stripped and "state.json" in stripped:
        return True
    # Phase skill files (compression, critique, expansion, forge, distillation)
    if "# Compression Pass" in stripped and "Convergent synthesis phase" in stripped:
        return True
    if "name: dialectic-critique" in stripped and "description:" in stripped:
        return True
    if "name: dialectic-distillation" in stripped and "description:" in stripped:
        return True
    if "name: dialectic-compression" in stripped:
        return True
    if "name: dialectic-expansion" in stripped and "description:" in stripped:
        return True
    # Forge / build spec skill
    if "# Forge" in stripped and ("You are executing the forge phase" in stripped or
                                   "You are synthesizing the output of a dialectic" in stripped):
        return True
    # Forge report (structured output, not reasoning)
    if "# Forge Report:" in stripped and "## Decision" in stripped:
        return True
    # Dialectic Distillation protocol
    if "# Dialectic Distillation" in stripped and "You are executing the distillation phase" in stripped:
        return True
    # Strategic patterns library (reference file, not reasoning)
    if "# Strategic Patterns Library" in stripped:
        return True
    # Semantic markers reference
    if "# Semantic Markers" in stripped and "Use these markers to structure" in stripped:
        return True
    # Philosophical foundations reference
    if "PHILOSOPHICAL-FOUNDATIONS" in text:
        return True
    # File listing results from reading skill directories
    if re.match(r'^\s*/?Users/.*/(plugins|skills)/', text.strip()):
        return True
    return False


def is_stop_hook_refeed(text: str) -> bool:
    """Check if this is a stop hook re-feed message."""
    return text.strip().startswith("Stop hook feedback:")


def is_command_invocation(text: str) -> bool:
    """Check if this is the initial command invocation."""
    return "<command-message>dialectic:dialectic</command-message>" in text


def detect_phase(text: str) -> list:
    """Detect dialectic phases in text. Returns list of (phase, start_idx, iteration) tuples."""
    phases = []

    # Match both formats: "# EXPANSION PASS (Iteration 3)" and "## EXPANSION PASS" and "## EXPANSION PASS (Final)"
    for match in re.finditer(r'#+\s*EXPANSION PASS(?:\s*\((?:Iteration\s*)?(\d+|Final)\))?', text):
        iter_val = match.group(1)
        iteration = int(iter_val) if iter_val and iter_val.isdigit() else None
        phases.append(("expansion", match.start(), iteration))

    for match in re.finditer(r'#+\s*COMPRESSION PASS(?:\s*\((?:Iteration\s*)?(\d+|Final)\))?', text):
        iter_val = match.group(1)
        iteration = int(iter_val) if iter_val and iter_val.isdigit() else None
        phases.append(("compression", match.start(), iteration))

    for match in re.finditer(r'#+\s*CRITIQUE PASS(?:\s*\((?:Iteration\s*)?(\d+|Final)\))?', text):
        iter_val = match.group(1)
        iteration = int(iter_val) if iter_val and iter_val.isdigit() else None
        phases.append(("critique", match.start(), iteration))

    phases.sort(key=lambda x: x[1])
    return phases


def split_by_phases(text: str, phases: list) -> list:
    """Split text into segments by phase boundaries. Returns list of (phase, iteration, content)."""
    if not phases:
        return [(None, None, text)]

    segments = []
    # Content before first phase marker
    pre_content = text[:phases[0][1]].strip()
    if pre_content:
        segments.append((None, None, pre_content))

    for i, (phase, start, iteration) in enumerate(phases):
        end = phases[i + 1][1] if i + 1 < len(phases) else len(text)
        segment_text = text[start:end].strip()
        if segment_text:
            segments.append((phase, iteration, segment_text))

    return segments


def parse_session(input_path: str) -> list:
    """Parse a raw JSONL session file into standardized trace entries."""
    entries = []
    current_turn = 0
    current_iteration = 1

    # First pass: collect all messages in order, merging consecutive assistant chunks
    raw_messages = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")

            # Skip non-message types
            if msg_type in ("file-history-snapshot", "progress", "queue-operation"):
                continue

            # System messages (stop hook re-feeds)
            if msg_type == "system":
                text = ""
                msg = obj.get("message", {})
                if isinstance(msg, dict):
                    text = extract_text_content(msg)
                elif isinstance(msg, str):
                    text = msg

                # Also check for system content in other fields
                if not text:
                    text = obj.get("content", "")

                if text and not is_tool_noise(text):
                    raw_messages.append({
                        "role": "system",
                        "content": text,
                        "is_meta": obj.get("isMeta", False)
                    })
                continue

            if msg_type not in ("user", "assistant"):
                continue

            msg = obj.get("message", {})
            if not msg:
                continue

            role = msg.get("role", msg_type)
            is_meta = obj.get("isMeta", False)
            text = extract_text_content(msg)

            if not text:
                continue

            # Map human -> user
            if role == "human":
                role = "user"

            raw_messages.append({
                "role": role,
                "content": text,
                "is_meta": is_meta
            })

    # Second pass: merge consecutive same-role messages and filter
    merged = []
    for msg in raw_messages:
        # Skip the command invocation line
        if msg["role"] == "user" and is_command_invocation(msg["content"]):
            # Extract the thesis from the command args
            match = re.search(r'<command-args><(.+?)>.*?</command-args>', msg["content"], re.DOTALL)
            if match:
                thesis_text = match.group(1)
                merged.append({
                    "role": "user",
                    "content": thesis_text,
                    "is_meta": False
                })
            continue

        # Skip meta/skill injections (protocol text, phase skill files, etc.)
        # Check all messages — skill content can arrive via meta injection OR tool results
        if is_meta_injection(msg["content"]):
            continue

        # Reclassify stop hook re-feeds as system role
        if is_stop_hook_refeed(msg["content"]):
            msg["role"] = "system"

        # Skip tool noise
        if is_tool_noise(msg["content"]):
            continue

        # Merge consecutive same-role messages
        if merged and merged[-1]["role"] == msg["role"] and msg["role"] == "assistant":
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    # Third pass: split by phases and assign turn numbers
    # Track iteration by counting expansion phases (each expansion = new iteration)
    expansion_count = 0

    for msg in merged:
        phases = detect_phase(msg["content"]) if msg["role"] == "assistant" else []

        if phases:
            segments = split_by_phases(msg["content"], phases)
            for phase, explicit_iter, content in segments:
                if phase == "expansion":
                    expansion_count += 1
                    # Use explicit iteration number if provided, otherwise use count
                    current_iteration = explicit_iter if explicit_iter is not None else expansion_count

                elif explicit_iter is not None:
                    current_iteration = explicit_iter

                if content.strip():
                    entries.append({
                        "turn": current_turn,
                        "role": msg["role"],
                        "phase": phase,
                        "iteration": current_iteration,
                        "content": content.strip()
                    })
                    current_turn += 1
        else:
            entries.append({
                "turn": current_turn,
                "role": msg["role"],
                "phase": None,
                "iteration": current_iteration,
                "content": msg["content"].strip()
            })
            current_turn += 1

    return entries


def count_thesis_modifications(entries: list) -> tuple:
    """Count thesis modifications. Returns (has_modifications, count).

    Looks for explicit thesis modification markers in critique/compression passes,
    and also checks for CONTINUE decisions paired with thesis updates.
    """
    mod_iterations = set()
    for entry in entries:
        content = entry["content"]
        content_lower = content.lower()

        # Explicit modification markers
        if any(marker in content for marker in [
            "THESIS MODIFIED", "Thesis Modified", "MODIFIED →",
            "**Updated Thesis**", "**Revised Thesis**", "**Modified Thesis**",
        ]):
            mod_iterations.add(entry["iteration"])
            continue

        # In critique/compression passes, look for thesis update language
        if entry["phase"] in ("critique", "compression"):
            # Look for "current thesis:" or "updated thesis:" patterns
            if re.search(r'(?:current|updated|revised|new|modified)\s+thesis\s*:', content_lower):
                mod_iterations.add(entry["iteration"])
            elif any(w in content_lower for w in [
                "thesis should be", "thesis needs to", "refine the thesis",
                "modify the thesis", "update the thesis", "amend the thesis",
                "thesis has evolved", "thesis now"
            ]):
                mod_iterations.add(entry["iteration"])

    return (len(mod_iterations) > 0, len(mod_iterations))


def estimate_tokens(entries: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = sum(len(e["content"]) for e in entries)
    return total_chars // 4


def count_complete_iterations(entries: list) -> int:
    """Count iterations that have all three phases (expansion, compression, critique)."""
    iteration_phases = {}
    for entry in entries:
        if entry["phase"]:
            it = entry["iteration"]
            if it not in iteration_phases:
                iteration_phases[it] = set()
            iteration_phases[it].add(entry["phase"])

    complete = sum(1 for phases in iteration_phases.values()
                   if {"expansion", "compression", "critique"} <= phases)
    return complete


def write_trace(entries: list, output_path: str):
    """Write entries as JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    sessions = [
        {
            "name": "thesis_geometry",
            "input": str(Path.home() / ".claude/projects/-Users-austinsalter-Documents-Projects-dialectic-rl/c93a366e-c1a3-4a45-bc1b-782726a3b1d9.jsonl"),
            "output": "traces/thesis_geometry/coherent.jsonl",
        },
        {
            "name": "harness_thesis",
            "input": str(Path.home() / ".claude/projects/-Users-austinsalter-Documents-Life-AI-Study/5d7eaf4d-a332-4a2e-ba73-fd4af1dea3ea.jsonl"),
            "output": "traces/harness_thesis/coherent.jsonl",
        },
        {
            "name": "agentic_commerce",
            "input": str(Path.home() / ".claude/projects/-Users-austinsalter-Documents-Life-Business-Study-Macro-Theses/9e1a9fd0-596f-40f9-a331-c9f3b8146418.jsonl"),
            "output": "traces/agentic_commerce/coherent.jsonl",
        },
        {
            "name": "kinnected",
            "input": str(Path.home() / ".claude/projects/-Users-austinsalter-Documents-Life-Side-Project-Kinnected/dc9cec9f-b9ed-4de4-b183-8e6cd6013eaa.jsonl"),
            "output": "traces/kinnected/coherent.jsonl",
        },
    ]

    # Resolve output paths relative to keel project
    keel_root = Path("/Users/austinsalter/Documents/Projects/keel")

    print(f"{'Session':<20} {'Turns':>6} {'Iterations':>11} {'~Tokens':>8} {'Thesis Mods':>12}")
    print("-" * 65)

    for session in sessions:
        input_path = session["input"]
        output_path = keel_root / session["output"]

        if not Path(input_path).exists():
            print(f"MISSING: {input_path}")
            continue

        entries = parse_session(input_path)
        write_trace(entries, str(output_path))

        turns = len(entries)
        iterations = count_complete_iterations(entries)
        tokens = estimate_tokens(entries)
        has_mods, mod_count = count_thesis_modifications(entries)

        mod_str = f"Yes ({mod_count})" if has_mods else "No"
        print(f"{session['name']:<20} {turns:>6} {iterations:>11} {tokens:>8,} {mod_str:>12}")

        # Write per-session stats
        stats = {
            "session_name": session["name"],
            "source_file": input_path,
            "turns": turns,
            "complete_iterations": iterations,
            "estimated_tokens": tokens,
            "thesis_modifications": mod_count,
            "phases": {}
        }
        for entry in entries:
            if entry["phase"]:
                phase = entry["phase"]
                stats["phases"][phase] = stats["phases"].get(phase, 0) + 1

        stats_path = output_path.parent / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
