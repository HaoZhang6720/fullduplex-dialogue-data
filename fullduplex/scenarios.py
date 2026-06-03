"""Interruption-scenario definitions for full-duplex dialogue generation.

Four scenarios are supported, two ``real`` interruptions (the assistant must
switch to listen and answer the new input) and two ``fake`` interruptions
(the assistant can continue speaking through the user's barge-in).

The full control-token vocabulary used by the pipeline is:

    [S.S] / <|switch-to-speak|>   Start speaking (begin assistant turn or new
                                  reply after a real interruption).
    [C.S] / <|continue_speak|>    Continue speaking after a fake interruption.
    [S.L] / <|switch-to-listen|>  Switch to listen (end of assistant turn or
                                  forced switch under a real interruption).
          <|continue-listen|>     Stay in listen mode -- emitted between
                                  fragments of a user utterance, added in the
                                  post-processing step 05 (not requested from
                                  the LLM, since LLMs can't reliably split
                                  user turns).
          <|interrupted|>         Internal marker on the assistant's
                                  truncated text; usually stripped at step 06.

Short-form ``[S.S] / [C.S] / [S.L]`` markers appear in the *prompt* shown to
the LLM (because the LLM emits them in its JSON output); they are converted
to the angle-bracket form (``<|switch-to-speak|>`` etc.) by step 05 when the
control tokens are inserted on the assistant side.
"""

EN_SCENARIOS = [
    {
        "name": "Denial or Discontent Interruption (real interruption)",
        "interrupt": "real interruption",
        "description": (
            "the user interrupts during the assistant's answer to express disagreement or "
            "dissatisfaction with the current response, requiring the assistant to switch to "
            "listening mode and respond to the new input."
        ),
        "format": (
            "{\\\"type\\\": \\\"real interruption\\\", "
            "\\\"User\\\": \\\"User query\\\", "
            "\\\"Sys\\\": \\\"[S.S] partially completed answer [S.L]\\\", "
            "\\\"User\\\": \\\"new query or follow-up\\\", "
            "\\\"Sys\\\": \\\"[S.S] new answer [S.L]\\\"} "
        ),
        "example": (
            "User asks \\\"How's the weather tomorrow?\\\". "
            "While the assistant says \\\"[S.S] The chance of rain tomorrow is ... [S.L]\\\" "
            "(not finished), the user interrupts with \\\"No, I want to know today's rainfall "
            "probability.\\\" The assistant updates the response to "
            "\\\"[S.S] Today's rainfall probability is 40%. [S.L]\\\"."
        ),
    },
    {
        "name": "Follow-up Question or Topic Change Interruption (real interruption)",
        "interrupt": "real interruption",
        "description": (
            "the user wants to interrupt with a new question or shifts the topic, requiring the "
            "assistant to switch to listening mode and respond to the new input."
        ),
        "format": (
            "{\\\"type\\\": \\\"real interruption\\\", "
            "\\\"User\\\": \\\"User query\\\", "
            "\\\"Sys\\\": \\\"[S.S] partially completed answer [S.L]\\\", "
            "\\\"User\\\": \\\"new query or follow-up\\\", "
            "\\\"Sys\\\": \\\"[S.S] new answer [S.L]\\\"} "
        ),
        "example": (
            "User asks \\\"How's the weather like tomorrow?\\\". "
            "As the assistant says \\\"[S.S] There will be light rain tomorrow with ... [S.L]\\\" "
            "(not finished), the user interrupts with \\\"What about the weekend?\\\". "
            "The assistant replies \\\"[S.S] It will be sunny this weekend with temperatures "
            "between 22 and 25 degrees. [S.L]\\\"."
        ),
    },
    {
        "name": "Agreement or Affirmation Acknowledgment (fake interruption)",
        "interrupt": "fake interruption",
        "description": (
            "the user does not want to interrupt; the user just barges in with an affirmation or "
            "agreement. The assistant evaluates that it can continue speaking [C.S] without "
            "being affected."
        ),
        "format": (
            "{\\\"type\\\": \\\"fake interruption\\\", "
            "\\\"User\\\": \\\"User query\\\", "
            "\\\"Sys\\\": \\\"[S.S] partially completed answer\\\", "
            "\\\"User\\\": \\\"fake interruption with affirmation or an unrelated comment\\\", "
            "\\\"Sys\\\": \\\"[C.S] continuation of the answer [S.L]\\\"} "
        ),
        "example": (
            "User asks \\\"Will the library open tomorrow?\\\". "
            "After the assistant says \\\"[S.S] The library is open tomorrow, you can enter ...\\\" "
            "(not finished), the user affirms with \\\"I see.\\\". "
            "The assistant proceeds to complete its original response "
            "\\\"[C.S] ... the library with a valid ID. [S.L]\\\"."
        ),
    },
    {
        "name": "Unrelated Comments/speech (fake interruption)",
        "interrupt": "fake interruption",
        "description": (
            "the user interrupts with a comment or statement that does not require a change in "
            "the assistant's response; the assistant continues speaking."
        ),
        "format": (
            "{\\\"type\\\": \\\"fake interruption\\\", "
            "\\\"User\\\": \\\"User query\\\", "
            "\\\"Sys\\\": \\\"[S.S] partially completed answer\\\", "
            "\\\"User\\\": \\\"fake interruption with affirmation or an unrelated comment\\\", "
            "\\\"Sys\\\": \\\"[C.S] continuation of the answer [S.L]\\\"} "
        ),
        "example": (
            "User asks \\\"Will it be windy tomorrow?\\\". "
            "While the assistant says \\\"[S.S] Tomorrow's wind strength will be ...\\\" "
            "(not finished), an unrelated statement from the user like \\\"I like Fei Wang's "
            "songs.\\\" or \\\"Can you pick up my son on your way back home?\\\" appears. "
            "The assistant proceeds to complete its original response "
            "\\\"[C.S] ... 4 to 5 levels. So be cautious if you are planning to go hiking. [S.L]\\\"."
        ),
    },
]

ZH_SCENARIOS = [
    {
        "name": "表达否认和不满（interruption）",
        "interrupt": "real interruption",
        "description": "用户表达对当前回答的不同意或不满，语音助手需要回复并相应地修改其回答。",
        "format": (
            "{\\\"type\\\": \\\"real interruption\\\", "
            "\\\"Dialog\\\": [{\\\"User\\\": \\\"用户提问\\\", "
            "\\\"Sys\\\": \\\"[S.S] 问题回答了一半 [S.L]\\\"}, "
            "{\\\"User\\\": \\\"用户打断\\\", "
            "\\\"Sys\\\": \\\"[S.S] 重新回复 [S.L]\\\"}]}"
        ),
        "example": (
            "user 问：\\\"请告诉我天气怎么样？\\\"。 "
            "assistant 回复说：\\\"明天的降雨概率是60%。\\\"，"
            "user 打断说：\\\"不对，我想知道的是今天的降雨概率。\\\"。 "
            "assistant 回复说：\\\"今天的降雨概率是40%。\\\"。"
        ),
    },
    {
        "name": "进一步询问或更换话题（interruption）",
        "interrupt": "real interruption",
        "description": "用户进一步询问提出新问题或转换到另一个随机话题，助手需要回应新的问题。",
        "format": (
            "{\\\"type\\\": \\\"real interruption\\\", "
            "\\\"Dialog\\\": [{\\\"User\\\": \\\"用户提问\\\", "
            "\\\"Sys\\\": \\\"[S.S] 问题回答了一半 [S.L]\\\"}, "
            "{\\\"User\\\": \\\"用户打断\\\", "
            "\\\"Sys\\\": \\\"[S.S] 重新回复 [S.L]\\\"}]}"
        ),
        "example": (
            "user 问：\\\"明天天气怎么样？\\\"。 "
            "assistant 回复说：\\\"明天会有小雨，日间气温大约在32摄氏度。\\\"。 "
            "user 打断说：\\\"那明天夜间温度如何？\\\"。 "
            "assistant 回复说：\\\"明天晚上相对于日间气温会有所下降，大约在16到20度之间。\\\"。"
        ),
    },
]


def get_scenarios(language: str = "en"):
    """Return the scenario pool for the requested language."""
    lang = language.lower()
    if lang in ("en", "english"):
        return EN_SCENARIOS
    if lang in ("zh", "chinese", "cn", "zh-cn"):
        return ZH_SCENARIOS
    raise ValueError(f"Unknown language: {language!r}. Use 'en' or 'zh'.")
