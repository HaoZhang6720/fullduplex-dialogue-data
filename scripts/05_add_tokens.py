#!/usr/bin/env python3
"""Step 05: Insert full-duplex control tokens into cleaned dialogues.

This step merges the original scripts:

* ``Step4_add_tokens_norm.py``                              -> mode ``norm``
* ``Step5_add_tokens_real_fake_1.py`` (modify_content_fake) -> mode ``fake``
* ``Step5_add_tokens_real_fake_1.py`` (modify_content_real) /
  ``Step5_add_tokens_real_fake_2.py``                       -> mode ``real``
* ``add_continue_listen/Step5_add_tokens_continueListen_only.py``
  (modify_content_continue2)                                -> mode ``continue_listen``

Token vocabulary added here
---------------------------
* ``<|switch-to-speak|>`` ... ``<|switch-to-listen|>``  - wrap normal assistant turns
* ``<|interrupted|>``        - marks the assistant's truncated utterance
* ``<|continue_speak|>``     - prefix for the assistant continuation after a fake interruption
* ``<|continue-listen|>``    - assistant emits this between halves of a user
                              utterance to signal "I am still listening"

Four randomized augmentations
-----------------------------
Each normal round can be probabilistically rewritten as either:

* a **fake interruption** (probability ``--fake-intr-prob``, default 0.4):
  the assistant text is split at a random point with ``<|interrupted|>`` /
  ``<|continue_speak|>``, and a random barge-in utterance is inserted as a
  new user turn (an affirmation or an unrelated comment, ratio controlled by
  ``--affirm-vs-unrelated``).

* a **real interruption** (probability ``--real-intr-prob``, default 0.7):
  applied only on rounds whose original ``type == 'interruption'``; the
  previous assistant turn is truncated with ``<|interrupted|>`` and the
  current user query is split so the assistant must emit ``<|switch-to-listen|>``
  before the user finishes.

* a **continue-listen augmentation** (probability ``--continue-listen-prob``,
  default 0.7): split the user query into two (or three) parts and insert
  ``<|continue-listen|>`` between them on the assistant side. Within the
  rounds that get augmented, a fraction (``--continue-listen-double-frac``,
  default 0.286 ~= original (1 - 0.8) / (1 - 0.3) ratio) get a double split
  (two ``<|continue-listen|>`` insertions).

The remaining rounds stay as normal ``<|switch-to-speak|> ... <|switch-to-listen|>``.

``--mode`` selects which transformations to apply:

* ``all`` (default): wrap-norm + fake + real + continue_listen, in that order.
* ``norm``: only wrap normal turns with switch-to-speak/listen.
* ``fake``: only apply fake-interruption rewriting.
* ``real``: only apply real-interruption rewriting.
* ``continue_listen``: only apply the continue-listen user-split augmentation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

# Make local package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fullduplex.affirmations import (  # noqa: E402
    sample_affirmation,
    sample_unrelated_speech,
    load_fake_query_pool,
)


# Control tokens.
TOK_SPEAK = "<|switch-to-speak|>"
TOK_LISTEN = "<|switch-to-listen|>"
TOK_INTERRUPTED = "<|interrupted|>"
TOK_CONTINUE = "<|continue_speak|>"
TOK_CONTINUE_LISTEN = "<|continue-listen|>"


def _letter_count(text: str) -> int:
    """Count alphabetic characters (Unicode category starts with 'L').
    Used to drop too-short user utterances from continue-listen splits."""
    return sum(1 for c in text if unicodedata.category(c).startswith("L"))


# ---------------------------------------------------------------------------
# Mode: wrap normal turns with speak/listen tokens
# ---------------------------------------------------------------------------

def apply_norm_tokens(dialogues: Iterable[dict]) -> Iterable[dict]:
    for data in dialogues:
        for round_info in data.values():
            if isinstance(round_info, dict) and "assistant" in round_info:
                a = round_info["assistant"]
                if not a.startswith(TOK_SPEAK):
                    round_info["assistant"] = f"{TOK_SPEAK} {a} {TOK_LISTEN}"
        yield data


# ---------------------------------------------------------------------------
# Mode: fake interruption augmentation
# ---------------------------------------------------------------------------

def _cut_index(text_len: int, low: float, high: float, rng: random.Random) -> int:
    """Return a cut index that respects the original 20-char prefix offset and
    leaves at least one character of suffix."""
    ratio = rng.uniform(low, high)
    base = max(text_len - 42, 1)
    cut = int(base * ratio) - 1
    return max(0, min(cut, base - 1))


def apply_fake_interruption(
    dialogues: Iterable[dict],
    fake_intr_prob: float = 0.4,
    affirm_vs_unrelated: float = 0.8,
    affirm_low: float = 0.3,
    affirm_high: float = 1.0,
    unrelated_low: float = 0.1,
    unrelated_high: float = 0.9,
    rng: Optional[random.Random] = None,
    fake_query_pool: Optional[List[str]] = None,
) -> Iterable[dict]:
    """Rewrite some ``normal`` rounds as fake interruptions.

    Args:
        fake_intr_prob: probability of converting a normal round.
        affirm_vs_unrelated: probability of using an affirmation (vs. an
            unrelated comment) when converting.
        affirm_low / affirm_high / unrelated_low / unrelated_high: bounds on
            the random cut position within the assistant text.
        fake_query_pool: optional list of barge-in utterances overriding the
            built-in unrelated-speech pool.
    """
    rng = rng or random
    for data in dialogues:
        for round_key, round_val in list(data.items()):
            if not isinstance(round_val, dict):
                continue
            if round_val.get("type") != "normal":
                continue
            assistant_text = round_val.get("assistant")
            if not isinstance(assistant_text, str):
                continue
            if rng.random() >= fake_intr_prob:
                continue

            text_len = len(assistant_text)
            if rng.random() < affirm_vs_unrelated:
                # Affirmation flavor.
                user_content = sample_affirmation(rng)
                cut = _cut_index(text_len, affirm_low, affirm_high, rng)
                cut_content_1 = assistant_text[:20 + cut] + f" {TOK_INTERRUPTED}"
                cut_content_2 = f"{TOK_CONTINUE} " + assistant_text[20 + cut:]
            else:
                # Unrelated speech flavor.
                if fake_query_pool:
                    user_content = rng.choice(fake_query_pool)
                else:
                    user_content = sample_unrelated_speech(rng)
                cut = _cut_index(text_len, unrelated_low, unrelated_high, rng)
                cut_content_1 = assistant_text[:20 + cut] + f" {TOK_INTERRUPTED}"
                cut_content_2 = f" {TOK_CONTINUE}" + assistant_text[20 + cut:]

            data[round_key] = {
                "type": "Fake INTR",
                "user": round_val["user"],
                "assistant": cut_content_1,
                "assistant_continuation2": {
                    "user": user_content,
                    "assistant": cut_content_2,
                },
            }
        yield data


# ---------------------------------------------------------------------------
# Mode: real interruption augmentation
# ---------------------------------------------------------------------------

def apply_real_interruption(
    dialogues: Iterable[dict],
    real_intr_prob: float = 0.7,
    cut_low: float = 0.5,
    cut_high: float = 1.0,
    q_cut_low: float = 0.5,
    q_cut_high: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Iterable[dict]:
    """Rewrite some ``interruption`` rounds as real interruptions.

    The previous assistant turn is truncated and tagged with ``<|interrupted|>``;
    the current user query is split so the assistant must emit
    ``<|switch-to-listen|>`` between the two halves.
    """
    rng = rng or random
    for data in dialogues:
        previous_assistant: Optional[str] = None
        previous_type: Optional[str] = None
        prev_key: Optional[str] = None

        for i, (round_key, round_info) in enumerate(data.items()):
            if not isinstance(round_info, dict):
                continue

            should_apply = (
                i > 0
                and round_info.get("type") == "interruption"
                and previous_assistant is not None
                and previous_type != "Fake INTR"
                and rng.random() < real_intr_prob
            )

            if should_apply:
                round_info["type"] = "Real INTR"

                # Truncate previous assistant turn.
                text_len = len(previous_assistant)
                cut = _cut_index(text_len, cut_low, cut_high, rng)
                cut_content = previous_assistant[:20 + cut] + TOK_INTERRUPTED
                data[prev_key]["assistant"] = cut_content

                # Split current user query into two parts.
                user_text = round_info["user"]
                q_ratio = rng.uniform(q_cut_low, q_cut_high)
                q_cut = max(1, int((len(user_text) - 1) * q_ratio) - 1)
                first_part = user_text[:q_cut]
                second_part = user_text[q_cut:]
                data[round_key]["user"] = first_part
                data[round_key]["assistant_continuation"] = {
                    "assistant": TOK_LISTEN,
                    "user": second_part,
                }

            previous_assistant = round_info.get("assistant")
            previous_type = round_info.get("type")
            prev_key = round_key
        yield data


# ---------------------------------------------------------------------------
# Mode: continue-listen augmentation
# ---------------------------------------------------------------------------

def apply_continue_listen(
    dialogues: Iterable[dict],
    cl_prob: float = 0.7,
    cl_double_frac: float = 0.286,
    min_letters: int = 5,
    single_low: float = 0.1,
    single_high: float = 0.9,
    double_low_1: float = 0.1,
    double_high_1: float = 0.5,
    double_low_2: float = 0.5,
    double_high_2: float = 0.9,
    max_attempts: int = 20,
    rng: Optional[random.Random] = None,
) -> Iterable[dict]:
    """Split user turns and insert ``<|continue-listen|>`` between the halves.

    With probability ``cl_prob``, the user turn of a round is split. Of those
    augmented turns, a fraction ``cl_double_frac`` get a *double* split (two
    ``<|continue-listen|>`` insertions yielding three user fragments); the rest
    get a *single* split (one ``<|continue-listen|>``, two fragments).

    Turns whose user text has fewer than ``min_letters`` alphabetic characters
    are skipped, as in the original ``modify_content_continue2`` heuristic.

    Skips rounds that already had their user turn split by an earlier pass
    (i.e. ``assistant_continuation`` already present from the real-INTR pass)
    to avoid clobbering.

    Ported from
    ``add_continue_listen/Step5_add_tokens_continueListen_only.py``
    (the ``modify_content_continue2`` variant, which is what produced the
    canonical training data).
    """
    rng = rng or random
    for data in dialogues:
        for round_key, round_info in data.items():
            if not isinstance(round_info, dict):
                continue
            user_text = round_info.get("user")
            if not isinstance(user_text, str):
                continue

            # Avoid double-augmenting a turn already split by a real INTR.
            if "assistant_continuation" in round_info:
                continue

            if rng.random() >= cl_prob:
                continue

            if _letter_count(user_text) < min_letters:
                continue

            text_len = len(user_text)
            if rng.random() >= cl_double_frac:
                # Single split.
                cut = _safe_single_cut(text_len, single_low, single_high, rng, max_attempts)
                if cut is None:
                    continue
                first, second = user_text[:cut], user_text[cut:]
                round_info["user"] = first
                round_info["assistant_continuation"] = {
                    "assistant": TOK_CONTINUE_LISTEN,
                    "user": second,
                }
            else:
                # Double split.
                cuts = _safe_double_cut(
                    text_len, double_low_1, double_high_1,
                    double_low_2, double_high_2, rng, max_attempts,
                )
                if cuts is None:
                    continue
                c1, c2 = cuts
                first = user_text[:c1]
                second = user_text[c1:c2]
                third = user_text[c2:]
                round_info["user"] = first
                round_info["assistant_continuation3"] = {
                    "assistant": TOK_CONTINUE_LISTEN,
                    "user": second,
                    "assistant2": TOK_CONTINUE_LISTEN,
                    "user2": third,
                }
        yield data


def _safe_single_cut(text_len: int, low: float, high: float,
                     rng: random.Random, max_attempts: int) -> Optional[int]:
    """Pick a cut index that leaves at least 2 chars on each side."""
    for _ in range(max_attempts):
        cut = int((text_len - 1) * rng.uniform(low, high)) - 1
        if 1 < cut < text_len - 2:
            return cut
    return None


def _safe_double_cut(text_len: int, low1: float, high1: float,
                     low2: float, high2: float,
                     rng: random.Random, max_attempts: int
                     ) -> Optional[tuple]:
    """Pick two cut indices c1 < c2 each leaving >= 2 chars on each side."""
    for _ in range(max_attempts):
        c1 = int(text_len * rng.uniform(low1, high1))
        c2 = int(text_len * rng.uniform(low2, high2))
        if 1 < c1 < c2 < text_len - 1:
            return c1, c2
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Insert full-duplex control tokens.")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode",
                   choices=["all", "norm", "fake", "real", "continue_listen"],
                   default="all")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--fake-intr-prob", type=float, default=0.4,
                   help="Probability of turning a normal round into a fake-interruption "
                        "round (default: 0.4, matching the original 1 - 0.6 threshold).")
    p.add_argument("--affirm-vs-unrelated", type=float, default=0.8,
                   help="Probability of using an affirmation (vs. unrelated comment) "
                        "for fake interruptions (default: 0.8).")
    p.add_argument("--real-intr-prob", type=float, default=0.7,
                   help="Probability of turning an interruption-typed round into a "
                        "real-interruption round (default: 0.7).")
    p.add_argument("--fake-query-pool", default=None,
                   help="Optional plain-text file (one query per line) to use as the "
                        "unrelated-speech pool. If omitted, the built-in pool is used.")

    p.add_argument("--continue-listen-prob", type=float, default=0.7,
                   help="Probability of splitting a user turn and inserting "
                        "<|continue-listen|> between the halves (default: 0.7, "
                        "matching the original ratio2 > 0.3 threshold).")
    p.add_argument("--continue-listen-double-frac", type=float, default=0.286,
                   help="Fraction of continue-listen augmentations that use a "
                        "double split (two <|continue-listen|> insertions, three "
                        "user fragments) instead of a single split. Default: 0.286, "
                        "which matches the original (0.2 / 0.7) ratio.")
    p.add_argument("--continue-listen-min-letters", type=int, default=5,
                   help="Skip continue-listen augmentation when the user text has "
                        "fewer than this many alphabetic characters. Default: 5.")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed) if args.seed is not None else random

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fake_pool: Optional[List[str]] = None
    if args.fake_query_pool:
        fake_pool = load_fake_query_pool(args.fake_query_pool)
        print(f"[05_add_tokens] loaded {len(fake_pool)} entries from {args.fake_query_pool}")

    dialogues = read_jsonl(in_path)
    print(f"[05_add_tokens] loaded {len(dialogues)} dialogues from {in_path}")

    # Original pipeline order: norm-wrap -> fake -> real -> continue_listen.
    # 1) norm-wrap inserts <|switch-to-speak|> ... <|switch-to-listen|> around
    #    every assistant turn (those tokens contribute the ~20/22-char prefix
    #    and suffix that the cut formulas in fake/real depend on).
    # 2) fake-interruption rewrites some normal rounds with <|interrupted|> /
    #    <|continue_speak|>; it also retags the round type as 'Fake INTR'.
    # 3) real-interruption rewrites some 'interruption'-typed rounds with
    #    <|interrupted|> (overwriting the trailing <|switch-to-listen|>) and
    #    splits the current user query so an inline <|switch-to-listen|>
    #    appears mid-utterance via assistant_continuation.
    # 4) continue-listen splits user turns with <|continue-listen|> (single or
    #    double cut), populating assistant_continuation / assistant_continuation3.
    if args.mode in ("all", "norm"):
        dialogues = list(apply_norm_tokens(dialogues))
    if args.mode in ("all", "fake"):
        dialogues = list(apply_fake_interruption(
            dialogues,
            fake_intr_prob=args.fake_intr_prob,
            affirm_vs_unrelated=args.affirm_vs_unrelated,
            rng=rng,
            fake_query_pool=fake_pool,
        ))
    if args.mode in ("all", "real"):
        dialogues = list(apply_real_interruption(
            dialogues,
            real_intr_prob=args.real_intr_prob,
            rng=rng,
        ))
    if args.mode in ("all", "continue_listen"):
        dialogues = list(apply_continue_listen(
            dialogues,
            cl_prob=args.continue_listen_prob,
            cl_double_frac=args.continue_listen_double_frac,
            min_letters=args.continue_listen_min_letters,
            rng=rng,
        ))

    n = write_jsonl(out_path, dialogues)
    print(f"[05_add_tokens] wrote {n} dialogues -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
