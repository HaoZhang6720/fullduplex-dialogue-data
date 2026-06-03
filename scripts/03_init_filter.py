#!/usr/bin/env python3
"""Step 03: Initial schema filter.

Drops dialogue records that do not match the expected ``{Round N: {user, assistant, ...}}``
schema. A record is kept only if **every** round contains both a ``user``
and an ``assistant`` field (these are the minimum requirements for the
downstream token-insertion steps).

The script accepts either lowercase ``user``/``assistant`` (gpt4o / doubao
flow after extraction) or the original GPT-4o JSON schema with ``User``/``Sys``
keys -- in the latter case the keys are normalized to lowercase before further
processing.

Ported from ``prompt_0917/Step2_init_filter.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalize_round(round_value: dict) -> dict:
    """Rename ``User``/``Sys`` -> ``user``/``assistant`` if present."""
    if not isinstance(round_value, dict):
        return round_value
    out = dict(round_value)
    if "User" in out and "user" not in out:
        out["user"] = out.pop("User")
    if "Sys" in out and "assistant" not in out:
        out["assistant"] = out.pop("Sys")
    return out


def _normalize_dialogue(data: dict) -> dict:
    """If the dialogue has top-level language wrappers (``"English version"``,
    ``"Chinese version"``), flatten them and merge round keys; otherwise return
    the dialogue as-is with per-round key normalization applied."""
    if not isinstance(data, dict):
        return data
    if "English version" in data or "Chinese version" in data:
        flat: dict = {}
        for lang_key, lang_val in data.items():
            if not isinstance(lang_val, dict):
                continue
            for k, v in lang_val.items():
                flat[f"{lang_key.split()[0]} {k}"] = _normalize_round(v)
        return flat
    return {k: _normalize_round(v) for k, v in data.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initial schema filter.")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
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
                print(f"[03] JSON decode error on line {num_total}")
                continue

            data = _normalize_dialogue(data)
            if not isinstance(data, dict) or not data:
                continue

            is_valid = True
            for _, round_value in data.items():
                if (not isinstance(round_value, dict)
                        or "user" not in round_value
                        or "assistant" not in round_value):
                    is_valid = False
                    break
            if not is_valid:
                continue
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            num_correct += 1

    print(f"[03_init_filter] kept {num_correct}/{num_total} records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
