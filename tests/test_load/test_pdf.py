from pathlib import Path

from easyparser.load.pdf import (
    pdf_by_extractous,
    pdf_by_pymupdf,
    pdf_by_unstructured,
    pdf_by_pymupdf4llm,
)

asset_dir = Path(__file__).parent.parent / "assets"


def test_unstructured():
    result = pdf_by_unstructured(asset_dir / "test.pdf")
    assert True


def test_pymupdf():
    result = pdf_by_pymupdf(asset_dir / "test.pdf")
    assert True


def test_pymupdf4llm():
    result = pdf_by_pymupdf4llm(asset_dir / "test.pdf")
    assert True


def test_extractous():
    result = pdf_by_extractous(asset_dir / "test.pdf")
    assert True
