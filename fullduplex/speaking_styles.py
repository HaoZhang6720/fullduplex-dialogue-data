"""Weighted speaking styles for the user persona in generated dialogues.

The first three styles receive most of the probability mass so that the
generated corpus skews toward natural conversational speech. The remaining
seven styles add diversity (urgency, confusion, excitement, ...).
"""

import random
from typing import List, Sequence


EN_STYLES: List[str] = [
    "Casual and conversational",
    "Polite and formal",
    "Calm and monotone",
    "Urgent and rushed",
    "Slow and deliberate",
    "Inquisitive and questioning",
    "Impatient and interruptive",
    "Confused or uncertain",
    "Excited and enthusiastic",
    "Affirmative and confirming",
]

ZH_STYLES: List[str] = [
    "随意且口语化",
    "礼貌且正式",
    "平静且单调",
    "紧急且匆忙",
    "缓慢且慎重",
    "好奇且质问",
    "不耐烦且打断",
    "困惑或不确定",
    "兴奋且热情",
    "肯定且确认",
]


def get_styles(language: str = "en") -> List[str]:
    """Return the style list for the requested language."""
    lang = language.lower()
    if lang in ("en", "english"):
        return EN_STYLES
    if lang in ("zh", "chinese", "cn", "zh-cn"):
        return ZH_STYLES
    raise ValueError(f"Unknown language: {language!r}. Use 'en' or 'zh'.")


def select_style(language: str = "en", rng: random.Random | None = None) -> str:
    """Sample a speaking style using the default skewed weighting.

    The first three styles share ~79% of probability mass; the remaining
    styles each get ~3%.
    """
    rng = rng or random
    styles = get_styles(language)
    weights = _default_weights(len(styles))
    return rng.choices(styles, weights=weights, k=1)[0]


def _default_weights(n: int) -> Sequence[float]:
    if n < 3:
        # Degenerate case: equal weights.
        return [1.0 / n] * n
    return [0.4, 0.195, 0.195] + [0.03] * (n - 3)
