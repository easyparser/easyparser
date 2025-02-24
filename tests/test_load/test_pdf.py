from pathlib import Path

from easyparser.load.pdf import (
    pdf_by_extractous,
    pdf_by_pymupdf,
    pdf_by_pymupdf4llm,
    pdf_by_unstructured,
)

asset_dir = Path(__file__).parent.parent / "assets"


def test_unstructured():
    _ = pdf_by_unstructured(asset_dir / "test.pdf")
    assert True


def test_pymupdf():
    _ = pdf_by_pymupdf(asset_dir / "test.pdf")
    assert True


def test_pymupdf4llm():
    _ = pdf_by_pymupdf4llm(asset_dir / "test.pdf")
    assert True


def test_extractous():
    _ = pdf_by_extractous(asset_dir / "test.pdf")
    assert True
