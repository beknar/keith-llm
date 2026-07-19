"""Domain vocabulary shared across the data pipeline, tokenizer, and CLI.

These value sets define the CTRL-style control tokens (`<|system:X|>`,
`<|doc:Y|>`) that condition generation, so changing them changes the
tokenizer's special-token set and invalidates existing token bins.
"""

# Core systems plus per-system tokens for every folder above ~7M tokens in the
# converted corpus (systems below that threshold stay `generic`). Adding a
# system here changes the tokenizer's special-token set, so it requires
# retraining the tokenizer + re-binarizing.
SYSTEMS = (
    # original core set
    "dnd5e",
    "savage_worlds",
    "d6",
    "generic",
    "homebrew",
    # per-system split (>7M tokens each)
    "call_of_cthulhu",
    "dnd_magazines",
    "dnd_2e",
    "dnd_3e",
    "dnd_35",
    "fate",
    "dnd_becmi",
    "d20_modern",
    "shadowrun",
    "dnd_4e",
    "d20_variants",
    "add_1e",
    "cypher",
    "l5r",
    "warhammer_40k",
    "gurps",
    "starfire",
    "battletech",
    "deadlands",
    "exalted",
    "iron_kingdoms",
    "cyberpunk",
    "dcc",
    "world_of_darkness",
)
DOC_TYPES = ("adventure", "rules", "bestiary", "setting")
