from pathlib import Path

from easyparser.load.pdf import UnstructuredPDFLoader, PyMuPDFLoader, ExtractousLoader

asset_dir = Path(__file__).parent.parent / "assets"


def test_unstructured():
    loader = UnstructuredPDFLoader()
    result = loader.load(asset_dir / "test.pdf")
    assert True


def test_pymupdf():
    loader = PyMuPDFLoader()
    result = loader.load(asset_dir / "test.pdf")
    assert True


def test_pymupdf4llm():
    loader = PyMuPDFLoader(mode="4llm")
    result = loader.load(asset_dir / "test.pdf")
    assert True


def test_extractous():
    loader = ExtractousLoader()
    result = loader.load(asset_dir / "test.pdf")
    assert True

if __name__ == "__main__":
    test_extractous()
