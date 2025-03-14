from pathlib import Path

from easyparser.base import Chunk
from easyparser.load.pdf import DoclingPDF, SycamorePDF, UnstructuredPDF
from easyparser.mime.pdf import MimeTypePDF

pdf_path = str(Path(__file__).parent.parent / "assets" / "short.pdf")


def test_sycamore():
    root = MimeTypePDF.as_root_chunk(pdf_path)
    chunks = SycamorePDF.run(root, use_ocr=False)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)


def test_unstructured():
    root = MimeTypePDF.as_root_chunk(pdf_path)
    chunks = UnstructuredPDF.run(root)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)


def test_docling():
    root = MimeTypePDF.as_root_chunk(pdf_path)
    chunks = DoclingPDF.run(root)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
