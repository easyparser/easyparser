from pathlib import Path

from easyparser.base import Chunk, ChunkGroup
from easyparser.mime import mime_pdf
from easyparser.parser.pdf import DoclingPDF, FastPDF, SycamorePDF, UnstructuredPDF

pdf_path1 = str(Path(__file__).parent.parent / "assets" / "short.pdf")
pdf_path2 = str(Path(__file__).parent.parent / "assets" / "short_image.pdf")


def test_sycamore():
    root = mime_pdf.as_root_chunk(pdf_path1)
    chunks = SycamorePDF.run(root, use_ocr=False)
    assert len(chunks) > 0
    assert root.id in chunks.groups
    assert isinstance(chunks[0], Chunk)


def test_unstructured():
    root = mime_pdf.as_root_chunk(pdf_path1)
    chunks = UnstructuredPDF.run(root)
    assert len(chunks) > 0
    assert root.id in chunks.groups
    assert isinstance(chunks[0], Chunk)


def test_docling():
    root = mime_pdf.as_root_chunk(pdf_path1)
    chunks = DoclingPDF.run(root)
    assert len(chunks) > 0
    assert root.id in chunks.groups
    assert isinstance(chunks[0], Chunk)


def test_fastpdf():
    root = mime_pdf.as_root_chunk(pdf_path1)
    chunks = FastPDF.run(root)
    assert len(chunks) > 0
    assert root.id in chunks.groups
    assert isinstance(chunks[0], Chunk)


def test_sycamore_multiple():
    root1 = mime_pdf.as_root_chunk(pdf_path1)
    root2 = mime_pdf.as_root_chunk(pdf_path2)
    chunks = SycamorePDF.run(ChunkGroup(chunks=[root1, root2]))
    assert root1.id in chunks.groups
    assert root2.id in chunks.groups
    assert len(chunks.groups[root1.id]) > 0
    assert len(chunks.groups[root2.id]) > 0
