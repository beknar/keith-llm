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
