#!/usr/bin/env python3
"""Step 02: Extract dialogue JSON objects from raw LLM answers.

Each input record looks like ``{"question": "...", "answer": ["<raw>"], ...}``.
The raw LLM output is expected to contain a JSON object describing the
multi-round dialogue (the prompt asks for this). LLMs sometimes return
JSON that is *almost* valid (missing trailing commas, stray Markdown
fences, etc.), so this script applies the original ``fix_json_format``
heuristic and surrounding ```json ... ``` fence stripping before parsing.

Successfully parsed dialogue objects are written one per line to ``--out``.

Ported from ``prompt_0917/Step1_extract_answer_from_output.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def fix_json_format(json_string: str) -> str:
    """Repair common LLM-emitted JSON glitches (missing commas between fields)."""
    lines = json_string.split("\n")
    corrected = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith('"') and line.endswith("}"):
                line += ","
            elif next_line.startswith('"') and line.endswith('"'):
                line += ","
        corrected.append(line)
    return "\n".join(corrected)


def strip_code_fences(text: str) -> str:
    """Remove leading/trailing ```json fences that some models emit."""
    return _FENCE_RE.sub("", text).strip()


def maybe_parse(raw: str) -> Optional[dict]:
    """Try to parse ``raw`` as JSON, applying the same fix-ups as the original
    pipeline."""
    candidates = []
    raw = strip_code_fences(raw)
    candidates.append(raw)
    candidates.append(fix_json_format(raw))
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract dialogue JSON from raw LLM answers.")
    p.add_argument("--in", dest="inp", required=True, help="Input JSONL (output of step 01).")
    p.add_argument("--out", required=True, help="Output JSONL of parsed dialogues.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_total = 0
    num_correct = 0
    with in_path.open("r", encoding="utf-8") as infile, \
            out_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            num_total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[02] outer JSON decode error on line {num_total}")
                continue
            answers = data.get("answer") or []
            for ans in answers:
                if not isinstance(ans, str) or not ans.strip():
                    continue
                parsed = maybe_parse(ans)
                if parsed is None:
                    print(f"[02] could not parse dialogue from item {num_total}")
                    continue
                json.dump(parsed, outfile, ensure_ascii=False)
                outfile.write("\n")
                num_correct += 1

    print(f"[02_extract_answers] parsed {num_correct}/{num_total} records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
