"""GPT-4o-style prompt builder.

Produces bilingual prompts (English + Chinese examples per call) that instruct
the target LLM to emit JSON-formatted multi-round dialogues with control
tokens ``[S.S]`` / ``[C.S]`` / ``[S.L]`` already inserted into the assistant
turns. This is the format used in the paper:
https://arxiv.org/pdf/2502.14145

Two length variants are supported:

- ``short``: 1-3 rounds, exactly one interruption round.
- ``long``: 4-8 rounds, with multiple interruption rounds.
"""

import random
from typing import Literal, Optional

from ..topics import EN_TOPICS
from ..scenarios import EN_SCENARIOS
from ..speaking_styles import select_style


Length = Literal["short", "long"]


def build_prompt(
    length: Length = "short",
    rng: Optional[random.Random] = None,
) -> str:
    """Build a single prompt string targeting an LLM (gpt-4o / doubao / ...).

    Args:
        length: ``"short"`` for 1-3 rounds with one interruption,
                ``"long"`` for 4-8 rounds with multiple interruptions.
        rng: Optional ``random.Random`` instance for deterministic sampling.
    """
    rng = rng or random
    if length == "short":
        return _build_short(rng)
    if length == "long":
        return _build_long(rng)
    raise ValueError(f"Unknown length: {length!r}. Use 'short' or 'long'.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADER = (
    "Objective: Generate a transcript of a conversation between a user and an AI voice "
    "assistant where the user can interrupt the assistant at any point. "
    "\\n\\nThe assistant operates in two modes: Speak (S) and Listen (L), and must handle "
    "both real and fake interruptions. "
    "\\n\\nAssistant's Behavior: "
    "\\n[S.S]: Start speaking, used at the beginning of the assistant's answer for normal QA "
    "or real interruptions; "
    "\\n[C.S]: At the beginning of the assistant's answer for fake interruptions, indicating "
    "the assistant continues speaking; "
    "\\n[S.L]: Switch to listening mode immediately after speaking or being interrupted by a "
    "real interruption; indicating it finishes answering or switches to listen mode. "
)

_OUTPUT_FORMAT = (
    "\\n\\nThe output is in JSON format with two top-level keys: \\\"English version\\\" and "
    "\\\"Chinese version\\\". Each version contains multiple rounds, e.g., \\\"Round 1\\\", "
    "\\\"Round 2\\\", etc. Inside each round there are keys: "
    "\\\"type\\\": Type of conversation (e.g., \\\"normal\\\", \\\"fake interruption\\\", or "
    "\\\"real interruption\\\"). "
    "\\\"User\\\": The user's query or comment. "
    "\\\"Sys\\\": The assistant's (system's) response, with special tokens like [S.S], [C.S], "
    "and [S.L]. "
)

_NORMAL_FORMAT = (
    "\\nFor normal rounds, there is one QA pair. The content within this round is: "
    "{\\\"type\\\": \\\"normal\\\", \\\"User\\\": \\\"query\\\", "
    "\\\"Sys\\\": \\\"[S.S] answer [S.L]\\\"}. "
)


def _build_short(rng: random.Random) -> str:
    """1-3 rounds, exactly one interruption round."""
    num_rounds = rng.choice([1, 2, 3])
    topic = rng.choice(EN_TOPICS)
    style = select_style("en", rng)
    scenario = rng.choice(EN_SCENARIOS)
    interruption_round = rng.randint(1, num_rounds)

    prompt = _HEADER
    prompt += (
        f"\\n\\nPlease generate {num_rounds} rounds of conversation. Generate two examples, "
        f"one in English, one in Chinese, labeled as \\\"English version\\\" and "
        f"\\\"Chinese version\\\", respectively. "
        f"\\n\\nThe user initiates the conversation with a topic related to {topic}. "
    )
    prompt += (
        f"The user is with a speaking style of {style}. "
        f"The assistant could use filler words and references to previous context for a "
        f"casual, friendly atmosphere. "
    )
    prompt += (
        f"\\n\\nIn round {interruption_round}, the user interrupts the system before it "
        f"finishes speaking with an interruption type of {scenario['name']}. "
        f"In such scenario, {scenario['description']} Example: {scenario['example']} "
    )
    prompt += "\\nAll other rounds are normal dialogue without interruption. "
    prompt += _OUTPUT_FORMAT
    prompt += _NORMAL_FORMAT
    prompt += (
        f"\\nFor round with {scenario['interrupt']}, there are two QA pairs: "
        f"{scenario['format']}. "
    )

    if "fake interruption" == scenario["interrupt"]:
        prompt += (
            "\\n\\nSpecial Notes: "
            "\\n1. If the next query from the user is a fake interruption: do not add [S.L] "
            "at the end of the assistant's partially completed answer. "
            "\\n2. Interruptions should occur before finishing a sentence, but after a "
            "complete word. "
        )
    else:
        prompt += (
            "\\n\\nSpecial Notes: "
            "\\n1. Ensure to add [S.L] at the end of the assistant's partially completed "
            "answer if the next query is a real interruption. "
            "\\n2. Interruptions should occur before finishing a sentence, but after a "
            "complete word."
        )
    return prompt


def _build_long(rng: random.Random) -> str:
    """4-8 rounds, with multiple interruption rounds."""
    num_rounds = rng.choice([4, 5, 6, 7, 8])
    interrupt_times = rng.randint(1, num_rounds - 1)
    interruption_rounds = sorted(rng.sample(range(1, num_rounds), interrupt_times))

    topic = rng.choice(EN_TOPICS)
    style = select_style("en", rng)

    prompt = _HEADER
    prompt += (
        f"\\n\\nPlease generate {num_rounds} rounds of conversation. Generate two examples, "
        f"one in English, one in Chinese, labeled as \\\"English version\\\" and "
        f"\\\"Chinese version\\\", respectively. "
        f"\\n\\nThe user initiates the conversation with a topic related to {topic}. "
    )
    prompt += (
        f"The user is with a speaking style of {style}. "
        f"The assistant could use filler words and references to previous context for a "
        f"casual, friendly atmosphere. "
    )

    seen_scenario = []
    seen_type = []
    for i in range(interrupt_times):
        scenario_i = rng.choice(EN_SCENARIOS)
        prompt += (
            f"\\nIn round {interruption_rounds[i]}, the user interrupts the system before it "
            f"finishes speaking with an interruption type of '{scenario_i['name']}'. "
        )
        if scenario_i not in seen_scenario:
            prompt += (
                f"In such a scenario, {scenario_i['description']} "
                f"Example: {scenario_i['example']} "
            )
            seen_scenario.append(scenario_i)
            seen_type.append(scenario_i["interrupt"])

    prompt += "\\nOther rounds are normal dialogue without interruption. "
    prompt += _OUTPUT_FORMAT
    prompt += _NORMAL_FORMAT

    if "real interruption" in seen_type:
        prompt += (
            "\\nFor real interruption round, there are two QA pairs: "
            "{\\\"type\\\": \\\"real interruption\\\", "
            "\\\"User\\\": \\\"User query\\\", "
            "\\\"Sys\\\": \\\"[S.S] partially completed answer [S.L]\\\", "
            "\\\"User\\\": \\\"new query or follow-up\\\", "
            "\\\"Sys\\\": \\\"[S.S] new answer [S.L]\\\"}. "
        )
    if "fake interruption" in seen_type:
        prompt += (
            "\\nFor fake interruption round, there are two QA pairs: "
            "{\\\"type\\\": \\\"fake interruption\\\", "
            "\\\"User\\\": \\\"User query\\\", "
            "\\\"Sys\\\": \\\"[S.S] partially completed answer\\\", "
            "\\\"User\\\": \\\"fake interruption with affirmation or an unrelated comment\\\", "
            "\\\"Sys\\\": \\\"[C.S] continuation of the answer [S.L]\\\"}. "
        )

    prompt += (
        "\\n\\nSpecial Notes: "
        "\\n1. Ensure to add [S.L] at the end of the assistant's partially completed answer "
        "if the next query is a real interruption; do not add [S.L] if the next query is a "
        "fake interruption. "
        "\\n2. Interruptions should occur before finishing a sentence, but after a complete "
        "word. "
    )
    return prompt
