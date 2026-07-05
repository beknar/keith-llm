"""Domain vocabulary shared across the data pipeline, tokenizer, and CLI.

These value sets define the CTRL-style control tokens (`<|system:X|>`,
`<|doc:Y|>`) that condition generation, so changing them changes the
tokenizer's special-token set and invalidates existing token bins.
"""

SYSTEMS = ("dnd5e", "savage_worlds", "d6", "generic", "homebrew")
DOC_TYPES = ("adventure", "rules", "bestiary", "setting")
