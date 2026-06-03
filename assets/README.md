# Assets

This folder ships small, illustrative resources that the pipeline can use out
of the box. None of these files are required -- you can replace or extend them.

## `fake_Q_examples.txt`

A small (~200 lines) curated pool of barge-in utterances in Chinese and
English, suitable for the **fake interruption** ("user makes a side comment
while the assistant is talking") scenario.

Pass it to step 05 with:

```bash
python scripts/05_add_tokens.py \
    --in outputs/answers_clean.jsonl \
    --out outputs/answers_tokenized.jsonl \
    --fake-query-pool assets/fake_Q_examples.txt
```

### How to extend

1. Run the pipeline once at small scale (e.g. 100 prompts). Step 04 will write
   every user utterance from `interruption` rounds to `outputs/interrupt_Q.txt`.
2. Review that file; copy the lines that look like natural side-comments /
   acknowledgments into your own pool.
3. Curate by domain (customer service, smart-home, productivity, ...) for
   better data-distribution control.

Lines are read verbatim (after stripping whitespace); empty lines are skipped.
