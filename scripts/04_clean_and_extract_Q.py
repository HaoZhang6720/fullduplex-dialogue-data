#!/usr/bin/env python3
"""Step 04: Clean user-side artifacts and optionally extract interruption queries.

Two things happen here:

1. **Clean** every round's ``user`` field of artifacts that the prompt phrasing
   sometimes causes the LLM to leak into the user turn, such as
   ``（interruption）``, ``表达否认和不满``, ``进一步询问或更换话题``. Records
   that end up with an empty ``user`` or that lack a ``type`` field are
   dropped.
2. **Extract** the user turns whose ``type == interruption`` and write them to
   a side file (``--extract-out``). This file is the raw ingredient for the
   custom "fake-Q pool" referenced in step 05; reviewing it lets you build a
   high-quality, in-domain pool of barge-in utterances over time.

Ported from ``prompt_0917/Step3_answers_filter_Q_extraction.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_ARTIFACTS = ["（interruption）", "(interruption)", "表达否认和不满", "进一步询问或更换话题"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean user artifacts and extract Q pool.")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extract-out", default=None,
                   help="Optional path to write the extracted interruption-user utterances "
                        "(one per line). If omitted, extraction is skipped.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    interrupt_user_lines: list[str] = []
    num_total = 0
    num_kept = 0

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
                print(f"[04] JSON decode error on line {num_total}")
                continue

            valid = True
            for _, round_value in data.items():
                if not isinstance(round_value, dict):
                    valid = False
                    break
                if "type" not in round_value or "user" not in round_value:
                    valid = False
                    break
                user_text = round_value["user"]
                if not isinstance(user_text, str):
                    valid = False
                    break
                for artifact in _ARTIFACTS:
                    if artifact in user_text:
                        user_text = user_text.replace(artifact, "")
                user_text = user_text.strip()
                if not user_text:
                    valid = False
                    break
                round_value["user"] = user_text

            if not valid:
                continue
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            num_kept += 1

            if args.extract_out:
                for _, round_value in data.items():
                    if round_value.get("type") == "interruption":
                        interrupt_user_lines.append(round_value["user"])

    if args.extract_out:
        extract_path = Path(args.extract_out)
        extract_path.parent.mkdir(parents=True, exist_ok=True)
        with extract_path.open("w", encoding="utf-8") as fout:
            for text in interrupt_user_lines:
                fout.write(text + "\n")
        print(f"[04_clean_and_extract_Q] extracted {len(interrupt_user_lines)} "
              f"interruption-user utterances -> {extract_path}")

    print(f"[04_clean_and_extract_Q] kept {num_kept}/{num_total} dialogues -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
