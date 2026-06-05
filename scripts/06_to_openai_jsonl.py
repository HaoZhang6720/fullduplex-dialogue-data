#!/usr/bin/env python3
"""Step 06: Flatten tokenized dialogues into a chat-style JSONL.

Each input dialogue is a dict ``{Round 1: {...}, Round 2: {...}, ...}`` whose
rounds may carry the special continuation fields produced by step 05:

* ``assistant_continuation``   - one extra (assistant, user) pair, used by
                                 real interruption (assistant emits
                                 ``<|switch-to-listen|>`` mid-utterance) and
                                 by single-split continue-listen (assistant
                                 emits ``<|continue-listen|>`` mid-utterance).
* ``assistant_continuation3``  - two extra (assistant, user) pairs, used by
                                 double-split continue-listen.
* ``assistant_continuation2``  - one extra (user, assistant) pair, used by
                                 fake interruption (user barges in, assistant
                                 emits ``<|continue_speak|>`` and finishes).

Two output formats are supported via ``--format``:

* ``openai`` (default): standard OpenAI ``messages`` format -- friendly for
  open-source SFT frameworks. One line per dialogue::

      {"messages": [
          {"role": "user",      "content": "..."},
          {"role": "assistant", "content": "<|switch-to-speak|> ... <|switch-to-listen|>"},
          ...
      ]}

This merges the original four scripts:

* ``Step_final_OpenAI_format_norm.py``
* ``Step_final_OpenAI_format_real_fake.py`` / ``Step_final_OpenAI_format_real_fake_json.py``
* ``Step_final_HY_YiZhangShi_normal_json.py``
* ``Step_final_HY_YiZhangShi_real_fake_json.py`` /
  ``Step_final_HY_YiZhangShi_continueListen_only_json.py``

Multi-file concatenation (the role of ``Combine_jsonl_files.py``) is done by
passing ``--in`` multiple times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


CHAT_SEP = "[CHAT_SEP]"

# Artifact strings that occasionally leak from the LLM into the user side and
# that we want to strip on the way out.
_ARTIFACT_STRINGS = [
    " <|interrupted|>", "<|interrupted|>",
    " interruption", "interruption",
    " （interruption）", "（interruption）",
    " （interrupt", "（interrupt",
    " （interrup",
]


def flatten_round(round_info: dict) -> List[dict]:
    """Flatten one round into an ordered list of ``{role, content}`` messages.

    The order follows the original
    ``Step_final_OpenAI_format_real_fake_json.py`` /
    ``Step_final_HY_YiZhangShi_continueListen_only_json.py`` logic:

    1. ``user``                        (initial query, possibly truncated)
    2. ``assistant_continuation``      assistant + user
       (real INTR's <|switch-to-listen|> midway, OR single-split continue-listen)
    3. ``assistant_continuation3``     assistant + user + assistant + user
       (double-split continue-listen)
    4. ``assistant``                   main assistant turn
    5. ``assistant_continuation2``     user + assistant
       (fake INTR's barge-in then <|continue_speak|>)
    """
    messages: List[dict] = []
    if "user" in round_info:
        messages.append({"role": "user", "content": round_info["user"]})

    if "assistant_continuation" in round_info:
        cont = round_info["assistant_continuation"]
        messages.append({"role": "assistant", "content": cont["assistant"]})
        messages.append({"role": "user", "content": cont["user"]})

    if "assistant_continuation3" in round_info:
        cont = round_info["assistant_continuation3"]
        messages.append({"role": "assistant", "content": cont["assistant"]})
        messages.append({"role": "user", "content": cont["user"]})
        messages.append({"role": "assistant", "content": cont["assistant2"]})
        messages.append({"role": "user", "content": cont["user2"]})

    if "assistant" in round_info:
        messages.append({"role": "assistant", "content": round_info["assistant"]})

    if "assistant_continuation2" in round_info:
        cont = round_info["assistant_continuation2"]
        messages.append({"role": "user", "content": cont["user"]})
        messages.append({"role": "assistant", "content": cont["assistant"]})

    return messages


def _scrub(text: str, strip_interrupted: bool, strip_artifacts: bool) -> str:
    if strip_interrupted:
        text = text.replace("<|interrupted|>", "")
    if strip_artifacts:
        for a in _ARTIFACT_STRINGS:
            text = text.replace(a, "")
    return text.strip() if (strip_interrupted or strip_artifacts) else text


def dialogue_to_messages(
    data: dict,
    strip_interrupted: bool = False,
    strip_artifacts: bool = False,
) -> List[dict]:
    """Convert a step-05 dialogue dict into an ordered list of OpenAI messages."""
    messages: List[dict] = []
    for _, round_info in data.items():
        if not isinstance(round_info, dict):
            continue
        for m in flatten_round(round_info):
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = _scrub(content, strip_interrupted, strip_artifacts)
            messages.append(m)
    return messages



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flatten tokenized dialogues to chat JSONL.")
    p.add_argument("--in", dest="inp", action="append", required=True,
                   help="Input JSONL (output of step 05). Pass multiple times to "
                        "concatenate several files into one output (replaces "
                        "Combine_jsonl_files.py).")
    p.add_argument("--out", required=True, help="Output chat-format JSONL.")
    p.add_argument("--format", choices=["openai", "hy_yzs"], default="openai",
                   help="Output schema. 'openai' = standard messages JSONL "
                        "(default, friendly for OSS SFT frameworks). ")
    p.add_argument("--strip-interrupted", action="store_true",
                   help="Remove the <|interrupted|> marker from every message "
                        "before writing.")
    p.add_argument("--strip-interruption-artifacts", action="store_true",
                   help="Also remove residual 'interruption' / '（interruption）' "
                        "strings that occasionally leak from the LLM.")
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

    n_in = 0
    n_out = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for data in iter_inputs(args.inp):
            n_in += 1
            messages = dialogue_to_messages(
                data,
                strip_interrupted=args.strip_interrupted,
                strip_artifacts=args.strip_interruption_artifacts,
            )
            if not messages:
                continue
            if args.format == "openai":
                record = {"messages": messages}
            else:  # hy_yzs
                record = messages_to_hy_yzs(messages)
                if not record["input"] and not record["output"]:
                    continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[06_to_chat_jsonl] read {n_in} dialogues from {len(args.inp)} file(s); "
          f"wrote {n_out} ({args.format}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
