from pathlib import Path
from typing import Literal

from ..base import Document, Origin
from .base import Loader


class PyMuPDFLoader(Loader):
    def __init__(self, mode: Literal["normal", "4llm"] = "normal"):
        self._mode = mode
        if mode == "normal":
            try:
                import pymupdf    # noqa
                self._m = pymupdf
            except ImportError:
                raise ImportError("Please install `pip install pymupdf pymupdf4llm`")
        elif mode == "4llm":
            try:
                import pymupdf4llm  # noqa
                self._m = pymupdf4llm
            except ImportError:
                raise ImportError("Please install `pip install pymupdf4llm`")

    def load(self, file_path, **kwargs) -> list[Document]:
        if self._mode == "normal":
            return self._load_normal(file_path, **kwargs)
        elif self._mode == "4llm":
            return self._load_4llm(file_path, **kwargs)

    def _load_normal(self, file_path, **kwargs) -> list[Document]:
        if self._mode != "normal":
            raise ValueError("This loader is not configured for normal mode")

        document = self._m.open(file_path)
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

    def _load_4llm(self, file_path, **kwargs) -> list[Document]:
        if self._mode != "4llm":
            raise ValueError("This loader is not configured for 4llm mode")

        from pymupdf4llm.llama.pdf_markdown_reader import PDFMarkdownReader
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


class ExtractousLoader(Loader):
    def __init__(self):
        self._m = None
        try:
            from extractous import Extractor
            self._m = Extractor()
        except ImportError:
            raise ImportError('Please install `pip install extractous`')

    def load(self, file_path, *args, **kwargs):
        if self._m is None:
            raise ImportError('Please install `pip install extractous`')

        result, metadata = self._m.extract_file_to_string(str(file_path))
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


class UnstructuredPDFLoader(Loader):
    def __init__(self):
        try:
            from unstructured.partition.pdf import partition_pdf
            self.unstructured = partition_pdf
        except ImportError:
            raise ImportError('Please install `pip install "unstructured[pdf]"`')

    def load(self, file_path, **kwargs) -> list[Document]:
        if "chunking_strategy" not in kwargs:
            kwargs["chunking_strategy"] = "basic"
        result = self.unstructured(file_path, **kwargs)
        return [self._unstructured_element_to_document(element) for element in result]

    def _unstructured_element_to_document(self, element) -> Document:
        metadata = element.metadata
        return Document(
            id=element.id,
            text=element.text,
            content=element.text,
            dtype="text",
            origin=Origin(
                source=Path(metadata.file_directory, metadata.filename).as_posix(),
                location=metadata.coordinates.to_dict(),
                getter="",
            ),
            metadata=metadata.to_dict(),
        )


class VLMPDFLoader(Loader):
    """Load PDFs using VLM."""
    ...

class DoclingLoader(Loader):
    pass
