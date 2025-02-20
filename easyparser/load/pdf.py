from pathlib import Path

from ..base import Document, Origin
from .base import Loader


class VLMPDFLoader(Loader):
    """Load PDFs using VLM."""

    ...


class DoclingLoader(Loader):
    pass


def pdf_by_pymupdf(file_path: Path | str) -> list[Document]:
    """Load a PDF file using PyMuPDF."""
    try:
        import pymupdf  # noqa
    except ImportError:
        raise ImportError("Please install `pip install pymupdf pymupdf4llm`")

    file_path = Path(file_path)
    document = pymupdf.open(file_path)
    text = ""
    for page in document:
        text += page.get_text()
    document.close()

    return [
        Document(
            id=file_path.stem,
            text=text,
            content=text,
            dtype="text",
            origin=Origin(
                source=file_path.as_posix(),
                location={},
                getter="",
            ),
            metadata={},
        )
    ]


def pdf_by_pymupdf4llm(file_path) -> list[Document]:
    try:
        from pymupdf4llm.llama.pdf_markdown_reader import PDFMarkdownReader
    except ImportError:
        raise ImportError("Please install `pip install pymupdf4llm`")

    reader = PDFMarkdownReader()
    outputs = reader.load_data(file_path)
    documents = [
        Document(
            id=file_path.stem,
            text=each.text,
            content=each.text,
            dtype="text",
            origin=Origin(
                source=file_path.as_posix(),
                location={},
                getter="",
            ),
            metadata={},
        )
        for each in outputs
    ]
    return documents


def pdf_by_extractous(file_path: Path | str) -> list[Document]:
    try:
        from extractous import Extractor
    except ImportError:
        raise ImportError("Please install `pip install extractous`")

    file_path = Path(file_path)
    extr = Extractor()
    result, metadata = extr.extract_file_to_string(str(file_path))
    return [
        Document(
            id=Path(file_path).stem,
            text=result,
            content=result,
            dtype="text",
            origin=Origin(
                source=file_path.as_posix(),
                location={},
                getter="",
            ),
            metadata=metadata,
        )
    ]


def pdf_by_unstructured(file_path: Path | str, **kwargs) -> list[Document]:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        raise ImportError('Please install `pip install "unstructured[pdf]"`')

    file_path = Path(file_path).resolve()
    file_path_str = str(file_path)
    if "chunking_strategy" not in kwargs:
        kwargs["chunking_strategy"] = "basic"
    result = partition_pdf(file_path_str, **kwargs)

    output = []
    for element in result:
        coord = (
            element.metadata.coordinates.to_dict()
            if element.metadata.coordinates
            else {}
        )
        output.append(
            Document(
                id=element.id,
                text=element.text,
                content=element.text,
                dtype="text",
                origin=Origin(
                    source=file_path_str,
                    location=coord,
                    getter="",
                ),
                metadata=element.metadata.to_dict(),
            )
        )
    return output


def pdf_by_docling(file_path: Path | str, **kwargs) -> list[Document]:
    ...


def pdf_by_vlm(file_path: Path | str, **kwargs) -> list[Document]:
    ...
