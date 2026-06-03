#!/usr/bin/env bash
# End-to-end driver for the full-duplex dialogue data pipeline.
#
# Usage:
#   bash run_pipeline.sh [STYLE] [NUM] [ROUNDS]
#
#   STYLE   gpt4o | doubao             (default: gpt4o)
#   NUM     number of prompts          (default: 1000)
#   ROUNDS  short | long | mixed       (default: short; only used for gpt4o)
#
# Requires a `.env` file at the project root with LLM_API_KEY / LLM_MODEL /
# optionally LLM_BASE_URL. See `.env.example`.
#
# Produces the following artifacts in $OUT/:
#   prompts.jsonl                       (step 00)
#   raw_answers.jsonl                   (step 01)
#   answers.jsonl                       (step 02)
#   answers_filtered.jsonl              (step 03)
#   answers_clean.jsonl                 (step 04)
#   interrupt_Q.txt                     (step 04, optional barge-in pool seed)
#   answers_tokenized.jsonl             (step 05, with all 4 control tokens)
#   fullduplex_sft_openai.jsonl         (step 06, OpenAI messages format)
#   fullduplex_sft_hy_yzs.jsonl         (step 06, canonical HY YiZhangShi format)
#   fullduplex_sft_final.jsonl          (step 07, the actual training file:
#                                        3x punctuation-augmented + shuffled)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present, so child Python processes also see the vars.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

STYLE=${1:-gpt4o}
NUM=${2:-1000}
ROUNDS=${3:-short}

OUT=outputs
mkdir -p "$OUT"

echo "[run_pipeline] style=$STYLE num=$NUM rounds=$ROUNDS"

python scripts/00_generate_prompts.py \
    --style "$STYLE" \
    --rounds "$ROUNDS" \
    --num "$NUM" \
    --out "$OUT/prompts.jsonl"

python scripts/01_call_llm.py \
    --in  "$OUT/prompts.jsonl" \
    --out "$OUT/raw_answers.jsonl"

python scripts/02_extract_answers.py \
    --in  "$OUT/raw_answers.jsonl" \
    --out "$OUT/answers.jsonl"

python scripts/03_init_filter.py \
    --in  "$OUT/answers.jsonl" \
    --out "$OUT/answers_filtered.jsonl"

python scripts/04_clean_and_extract_Q.py \
    --in  "$OUT/answers_filtered.jsonl" \
    --out "$OUT/answers_clean.jsonl" \
    --extract-out "$OUT/interrupt_Q.txt"

# Insert all four control tokens (norm, fake INTR, real INTR, continue-listen).
python scripts/05_add_tokens.py \
    --in  "$OUT/answers_clean.jsonl" \
    --out "$OUT/answers_tokenized.jsonl" \
    --mode all

# Flatten to OpenAI messages format (good for most OSS SFT frameworks).
python scripts/06_to_openai_jsonl.py \
    --in  "$OUT/answers_tokenized.jsonl" \
    --out "$OUT/fullduplex_sft_openai.jsonl" \
    --format openai \
    --strip-interruption-artifacts

# Also flatten to the canonical HY YiZhangShi {input,output} [CHAT_SEP] format,
# which is what was actually used to train the paper's model.
python scripts/06_to_openai_jsonl.py \
    --in  "$OUT/answers_tokenized.jsonl" \
    --out "$OUT/fullduplex_sft_hy_yzs.jsonl" \
    --format hy_yzs \
    --strip-interruption-artifacts

# Apply the 3x punctuation augmentation + shuffle to produce the final training
# file. Comment this out if you do not want the augmentation.
python scripts/07_finalize.py \
    --in  "$OUT/fullduplex_sft_hy_yzs.jsonl" \
    --out "$OUT/fullduplex_sft_final.jsonl" \
    --punctuation all3 \
    --shuffle

echo "[run_pipeline] done."
echo "  OpenAI format:    $OUT/fullduplex_sft_openai.jsonl"
echo "  HY YiZhangShi:    $OUT/fullduplex_sft_hy_yzs.jsonl"
echo "  Final training:   $OUT/fullduplex_sft_final.jsonl"
