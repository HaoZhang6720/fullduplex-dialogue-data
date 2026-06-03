#!/usr/bin/env python3
"""Step 07: Finalize the HY YiZhangShi training file.

Three things happen here, mirroring what the original repo's
``Combine_jsonl_files_then_shuffle.py`` + ``modify_punctuation.py``
chain did:

1. **Concatenate** all input files (``--in`` repeatable).
2. **Punctuation augmentation** of the ``input`` field, controlled by
   ``--punctuation``:

   * ``none``  -> emit the records as-is.
   * ``strip`` -> remove trailing punctuation from each ``[CHAT_SEP]``-separated
                  segment of ``input``.
   * ``add``   -> add a Chinese period ``。`` to any segment that lacks
                  trailing punctuation.
   * ``all3``  -> emit *three* copies of every record: base, stripped, added.
                  This is what produced the canonical
                  ``train_fullduplex_HYyzs_Cleaned_PunctuaionCombined_shuffled.jsonl``.

3. **Shuffle** the (concatenated and possibly augmented) records when
   ``--shuffle`` is passed, so different punctuation variants are interleaved.

Input is the HY YiZhangShi format produced by ``06_to_chat_jsonl.py --format
hy_yzs``: one line per dialogue with ``{"input": "...[CHAT_SEP]...",
"output": "...[CHAT_SEP]..."}``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, List

CHAT_SEP = "[CHAT_SEP]"

_TRAILING_PUNCT_RE = re.compile(r"[.?!,:;\u3002\uff01\uff0c\uff1b\uff1f]$")


def _strip_trailing_punct(text: str) -> str:
    parts = text.split(CHAT_SEP)
    parts = [_TRAILING_PUNCT_RE.sub("", p.strip()) for p in parts]
    return CHAT_SEP.join(parts)


def _add_trailing_period(text: str) -> str:
    parts = text.split(CHAT_SEP)
    parts = [(p.strip() + ("。" if not _TRAILING_PUNCT_RE.search(p.strip()) else ""))
             for p in parts]
    return CHAT_SEP.join(parts)


def punct_variants(record: dict, mode: str) -> List[dict]:
    if mode == "none":
        return [record]
    base_input = record.get("input", "")
    if mode == "strip":
        out = dict(record)
        out["input"] = _strip_trailing_punct(base_input)
        return [out]
    if mode == "add":
        out = dict(record)
        out["input"] = _add_trailing_period(base_input)
        return [out]
    if mode == "all3":
        base = dict(record)
        stripped = dict(record)
        stripped["input"] = _strip_trailing_punct(base_input)
        added = dict(record)
        added["input"] = _add_trailing_period(base_input)
        return [base, stripped, added]
    raise ValueError(f"Unknown punctuation mode: {mode!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize HY YiZhangShi training JSONL.")
    p.add_argument("--in", dest="inp", action="append", required=True,
                   help="Input HY YiZhangShi JSONL (output of step 06 with "
                        "--format hy_yzs). Pass multiple times to concatenate.")
    p.add_argument("--out", required=True, help="Output JSONL.")
    p.add_argument("--punctuation",
                   choices=["none", "strip", "add", "all3"],
                   default="all3",
                   help="How to augment the trailing punctuation of each user "
                        "segment in 'input'. Default 'all3' reproduces the "
                        "canonical 3x augmentation used in the original repo.")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle the (concatenated and augmented) records "
                        "before writing. Recommended when --punctuation=all3 "
                        "so that the three variants interleave.")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional integer seed for the shuffle.")
    return p.parse_args()


def iter_inputs(paths: Iterable[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed) if args.seed is not None else random

    records: List[dict] = []
    n_in = 0
    for record in iter_inputs(args.inp):
        n_in += 1
        records.extend(punct_variants(record, args.punctuation))

    if args.shuffle:
        rng.shuffle(records)

    with out_path.open("w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[07_finalize] read {n_in} records from {len(args.inp)} file(s); "
          f"punctuation={args.punctuation} shuffle={args.shuffle} "
          f"-> wrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
