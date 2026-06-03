#!/usr/bin/env python3
"""Step 00: Generate prompts that will be sent to an LLM.

Writes one JSON object per line. Each line follows the schema expected by
``scripts/01_call_llm.py``::

    {"messages": [{"role": "user", "content": "<the generated prompt>"}],
     "meta": {"style": "...", "rounds": "...", "index": N}}

Usage examples
--------------
Generate 1000 GPT-4o-style short prompts:

    python scripts/00_generate_prompts.py --style gpt4o --rounds short \\
        --num 1000 --out outputs/prompts.jsonl

Generate a mix of short and long GPT-4o-style prompts:

    python scripts/00_generate_prompts.py --style gpt4o --rounds mixed \\
        --num 5000 --out outputs/prompts.jsonl

Generate Doubao-style Chinese prompts:

    python scripts/00_generate_prompts.py --style doubao --num 2000 \\
        --out outputs/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Make the local package importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fullduplex.prompt_builders import build_gpt4o_prompt, build_doubao_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate prompts for full-duplex dialogue data.")
    p.add_argument("--style", choices=["gpt4o", "doubao"], default="gpt4o",
                   help="Prompt-builder style. 'gpt4o' is bilingual with control tokens "
                        "already in the prompt; 'doubao' is Chinese-only and adds tokens "
                        "in post-processing.")
    p.add_argument("--rounds", choices=["short", "long", "mixed"], default="short",
                   help="Only used by --style gpt4o. 'short'=1-3 rounds, 'long'=4-8 rounds, "
                        "'mixed'=uniform mix of both.")
    p.add_argument("--num", type=int, default=1000,
                   help="Number of prompts to generate.")
    p.add_argument("--out", type=str, required=True,
                   help="Path to output JSONL file. Parent directories are created.")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional random seed for reproducibility.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed) if args.seed is not None else random

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(args.num):
            if args.style == "gpt4o":
                length = args.rounds
                if length == "mixed":
                    length = rng.choice(["short", "long"])
                prompt = build_gpt4o_prompt(length=length, rng=rng)
                meta = {"style": "gpt4o", "rounds": length, "index": i}
            else:
                prompt = build_doubao_prompt(rng=rng)
                meta = {"style": "doubao", "rounds": "variable", "index": i}

            record = {
                "messages": [{"role": "user", "content": prompt}],
                "meta": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[00_generate_prompts] wrote {written} prompts to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
