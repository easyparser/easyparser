from pathlib import Path

from easyparser.base import Chunk, Origin
from easyparser.load.pdf import SycamorePDF

pdf_path = str(Path(__file__).parent.parent / "assets" / "long.pdf")


def test_sycamore():
    original_chunk = Chunk(mimetype="file", origin=Origin(location=pdf_path))
    chunks = SycamorePDF.run(original_chunk, use_ocr=False)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
