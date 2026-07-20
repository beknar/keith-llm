import json

from keith_llm.sft.loader import (
    _record_to_turns,
    load_sft_examples,
    tokenize_conversation,
    tokenize_example,
)
from keith_llm.tokenizer.wrapper import KeithTokenizer


def test_single_turn_masks_prompt_loss_on_response(tiny_tokenizer_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    x, y = tokenize_conversation(tok, [("What is a goblin?", "A small humanoid.")], 256)
    assert y[0] == -1  # bos target is a prompt token -> masked
    # unmasked targets == assistant tokens + eos
    n_targets = sum(1 for t in y if t != -1)
    assert n_targets == len(tok.encode("A small humanoid.")) + 1
    # the wrapper matches the old single-turn entry point
    assert tokenize_example(tok, "What is a goblin?", "A small humanoid.", 256) == (x, y)


def test_multi_turn_computes_loss_on_every_assistant_turn(tiny_tokenizer_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    turns = [("Q one here", "A one answer"), ("Q two here", "A two answer")]
    x, y = tokenize_conversation(tok, turns, 256)
    n_targets = sum(1 for t in y if t != -1)
    expected = sum(len(tok.encode(a)) + 1 for _, a in turns)  # both responses + both eos
    assert n_targets == expected
    # earlier turns are present as context in the input
    dec = tok.decode(x)
    assert "Q one here" in dec and "A one answer" in dec and "Q two here" in dec
    # and it's strictly more loss-bearing than just the first turn
    _, y1 = tokenize_conversation(tok, turns[:1], 256)
    assert n_targets > sum(1 for t in y1 if t != -1)


def test_record_to_turns_both_formats():
    assert _record_to_turns({"instruction": "Q", "response": "A"}) == [("Q", "A")]
    msgs = {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
    }
    assert _record_to_turns(msgs) == [("Q1", "A1"), ("Q2", "A2")]


def test_record_to_turns_drops_unpaired_trailing_user():
    # a dangling user with no assistant reply is not a training turn
    msgs = {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "dangling"},
        ]
    }
    assert _record_to_turns(msgs) == [("Q1", "A1")]


def test_load_sft_examples_handles_mixed_formats(tmp_path, tiny_tokenizer_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    p = tmp_path / "sft.jsonl"
    p.write_text(
        json.dumps({"instruction": "single Q", "response": "single A"})
        + "\n"
        + json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "multi Q1"},
                    {"role": "assistant", "content": "multi A1"},
                    {"role": "user", "content": "multi Q2"},
                    {"role": "assistant", "content": "multi A2"},
                ]
            }
        )
        + "\n"
    )
    examples = load_sft_examples(p, tok, 256)
    assert len(examples) == 2  # both formats loaded
