import pytest


def build_pdf(page_texts: list[str]) -> bytes:
    """Build a minimal but structurally valid PDF (correct xref offsets) with
    one Helvetica text line per page. Enough for pypdf to extract text from.

    Page texts must not contain parentheses or backslashes (no PDF string
    escaping is done).
    """
    objs: dict[int, bytes] = {}
    kids = []
    next_num = 4
    for text in page_texts:
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        stream_num, page_num = next_num, next_num + 1
        next_num += 2
        objs[stream_num] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content)
        objs[page_num] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 3 0 R >> >> >>" % stream_num
        )
        kids.append(f"{page_num} 0 R")
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(page_texts)} >>".encode()
    objs[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += b"%d 0 obj\n" % num + objs[num] + b"\nendobj\n"
    xref_pos = len(out)
    size = max(objs) + 1
    out += b"xref\n0 %d\n" % size
    out += b"0000000000 65535 f \n"
    for num in range(1, size):
        out += b"%010d 00000 n \n" % offsets[num]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_pos)
    return bytes(out)


@pytest.fixture
def make_pdf():
    return build_pdf


_NOUNS = ["goblin", "wyvern", "lich", "barrow", "tavern", "relic", "warden", "mire", "keep"]
_VERBS = ["ambushes", "guards", "haunts", "curses", "seeks", "burns", "buries", "awakens"]
_TAILS = [
    "beneath the shattered bridge",
    "at the edge of the mirewood",
    "inside the sunken temple",
    "when the second moon rises",
    "for 2d6+3 fire damage",
    "unless a DC 15 Wisdom save succeeds",
    "while the caravan sleeps",
]


def synthetic_corpus_records(n_docs: int = 36) -> list[dict]:
    """Small but varied TTRPG-flavored corpus spread across all systems and
    doc types. Deterministic; ~50-80 unique sentences per document."""
    from keith_llm.constants import DOC_TYPES, SYSTEMS

    records = []
    for d in range(n_docs):
        sentences = []
        for s in range(60):
            i = d * 60 + s
            sentences.append(
                f"The {_NOUNS[i % len(_NOUNS)]} {_VERBS[(i // 3) % len(_VERBS)]} "
                f"the {_NOUNS[(i + 4) % len(_NOUNS)]} {_TAILS[(i // 2) % len(_TAILS)]}."
            )
        text = " ".join(sentences)
        import hashlib

        records.append(
            {
                "id": hashlib.sha1(text.encode()).hexdigest(),
                "source": f"doc{d}.txt",
                "system": SYSTEMS[d % len(SYSTEMS)],
                "doc_type": DOC_TYPES[d % len(DOC_TYPES)],
                "license": "CC-BY-4.0",
                "publishable": True,
                "text": text,
            }
        )
    return records


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory):
    import json

    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    with path.open("w") as fh:
        for rec in synthetic_corpus_records():
            fh.write(json.dumps(rec) + "\n")
    return path


@pytest.fixture(scope="session")
def tiny_tokenizer_path(tmp_path_factory, tiny_corpus):
    from keith_llm.tokenizer.train import train_bpe

    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    train_bpe(tiny_corpus, path, vocab_size=512)
    return path


@pytest.fixture(scope="session")
def tiny_bins(tmp_path_factory, tiny_corpus, tiny_tokenizer_path):
    """Binarized tiny corpus: (tokens_dir, meta dict)."""
    from keith_llm.data.binarize import binarize

    out_dir = tmp_path_factory.mktemp("tiny_tokens")
    meta = binarize(tiny_corpus, tiny_tokenizer_path, out_dir, val_mod=5)
    return out_dir, meta
