#!/usr/bin/env python3
"""Step 01: Send prompts to an OpenAI-compatible LLM and store raw answers.

**You must supply your own LLM credentials.** This script reads from a ``.env``
file (or the environment) and uses the ``openai`` Python client, which works
with any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, Doubao Ark, vLLM,
DeepInfra, Together, ...). The endpoint is selected via ``LLM_BASE_URL`` and
the model via ``LLM_MODEL``.

Required env vars
-----------------
- ``LLM_API_KEY``     The API key for your provider.
- ``LLM_MODEL``       The model name, e.g. ``gpt-4o``, ``doubao-pro-32k``.

Optional env vars
-----------------
- ``LLM_BASE_URL``    Base URL of the endpoint (omit for OpenAI's default).
- ``LLM_CONCURRENCY`` Number of concurrent requests (default: 8).
- ``LLM_TEMPERATURE`` Sampling temperature (default: 1.0).
- ``LLM_MAX_TOKENS``  Max output tokens (default: 4096).

Input schema (per line of --in)
-------------------------------
``{"messages": [{"role": "user", "content": "..."}], "meta": {...}}``

Output schema (per line of --out)
---------------------------------
``{"question": "...", "answer": ["..."], "meta": {...}, "error": "..."}``

The ``answer`` field is a list to remain compatible with downstream
``02_extract_answers.py`` which expects ``data['answer']`` to be iterable
(some providers can be configured to return multiple choices).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is optional but recommended
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    print("ERROR: the `openai` package is required. Install with `pip install openai`.",
          file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore[no-redef]
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Call an OpenAI-compatible LLM on a JSONL of prompts.")
    p.add_argument("--in", dest="inp", required=True, help="Input prompts JSONL.")
    p.add_argument("--out", required=True, help="Output answers JSONL.")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Override LLM_CONCURRENCY env var.")
    p.add_argument("--max-retries", type=int, default=5,
                   help="Max retries per request on transient failures.")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only process the first N prompts (useful for smoke tests).")
    p.add_argument("--resume", action="store_true",
                   help="If set, append to --out and skip already-processed prompt indices.")
    return p.parse_args()


def load_env_config() -> Dict[str, Any]:
    load_dotenv()  # loads ./.env if present
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not api_key:
        raise SystemExit("ERROR: LLM_API_KEY is not set. Copy .env.example to .env and edit it.")
    if not model:
        raise SystemExit("ERROR: LLM_MODEL is not set. Copy .env.example to .env and edit it.")
    return {
        "api_key": api_key,
        "base_url": os.environ.get("LLM_BASE_URL") or None,
        "model": model,
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "1.0")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "4096")),
        "concurrency": int(os.environ.get("LLM_CONCURRENCY", "8")),
    }


def call_with_retry(client: OpenAI, model: str, messages: List[Dict[str, str]],
                    temperature: float, max_tokens: int, max_retries: int) -> str:
    """Send a single chat completion request with exponential backoff."""
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # broad catch is intentional; provider errors vary
            last_err = e
            sleep_for = delay * (2 ** attempt)
            time.sleep(min(sleep_for, 60.0))
    raise RuntimeError(f"Exhausted retries: {last_err!r}") from last_err


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def already_done_indices(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                idx = rec.get("meta", {}).get("index")
                if idx is not None:
                    done.add(idx)
            except json.JSONDecodeError:
                continue
    return done


def main() -> int:
    args = parse_args()
    cfg = load_env_config()
    if args.concurrency is not None:
        cfg["concurrency"] = args.concurrency

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = read_jsonl(in_path)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    skip = already_done_indices(out_path) if args.resume else set()
    if skip:
        prompts = [p for p in prompts
                   if p.get("meta", {}).get("index") not in skip]
        print(f"[01_call_llm] resuming; skipping {len(skip)} already-done items.")

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    open_mode = "a" if args.resume and out_path.exists() else "w"
    print(f"[01_call_llm] model={cfg['model']} base_url={cfg['base_url']} "
          f"concurrency={cfg['concurrency']} prompts={len(prompts)}")

    def work(item: Dict[str, Any]) -> Dict[str, Any]:
        messages = item["messages"]
        user_text = messages[-1]["content"] if messages else ""
        try:
            answer_text = call_with_retry(
                client, cfg["model"], messages, cfg["temperature"],
                cfg["max_tokens"], args.max_retries,
            )
            return {
                "question": user_text,
                "answer": [answer_text],
                "meta": item.get("meta", {}),
            }
        except Exception as e:  # pylint: disable=broad-except
            return {
                "question": user_text,
                "answer": [],
                "meta": item.get("meta", {}),
                "error": str(e),
            }

    with out_path.open(open_mode, encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
        futures = [pool.submit(work, item) for item in prompts]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM"):
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[01_call_llm] done. wrote {len(prompts)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
