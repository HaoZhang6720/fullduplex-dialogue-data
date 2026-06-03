"""Full-duplex dialogue data generation package.

Provides topic pools, interruption scenarios, speaking styles, affirmation
pools, and prompt builders used to generate multi-round dialogue prompts
with control tokens for full-duplex SFT data.

Four control tokens describe the assistant's state transitions:

* ``<|switch-to-speak|>``   start speaking (beginning of an assistant turn).
* ``<|switch-to-listen|>``  switch to listen (end of an assistant turn, or
                            forced switch mid-utterance after a real
                            interruption).
* ``<|continue_speak|>``    continue speaking after a fake interruption.
* ``<|continue-listen|>``   stay in listen mode -- emitted between user
                            fragments when the user pauses mid-utterance.

Plus the internal marker ``<|interrupted|>`` on the assistant's truncated
text (usually stripped at the final flatten step).
"""

__version__ = "0.2.0"
