"""Collection of schemas, metadata and operations for chunks from PDF

- The element labels that chunks in PDF can be classified into.
- How to represent location of a chunk in a PDF.
- How to crop / capture a chunk from a PDF.
- How to render a chunk and its subsequent children in a PDF.
- How chunks can be combined, or broken down, ensuring relationship, metadata
and structure is preserved.
"""

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from easyparser.base import Chunk, Origin


class Label(Enum):
    """Possible label values for a chunk from PDF"""

    text = "text"  # block of text

    # important for structure analysis, inferring sections
    heading = "heading"

    # important for multi-modal usage
    image = "image"  # if contains image
    table = "table"  # if contains table
    formula = "formula"  # if contains formula
    checkbox = "checkbox"  # if contains checkbox


@dataclass
class ChildMetadata:
    """Metadata for a child chunk from PDF"""

    label: str = "text"  # one of mime_pdf.Label

    def as_dict(self):
        return asdict(self)


def as_root_chunk(path: str) -> Chunk:
    """From a pdf file to a base chunk"""
    path = str(Path(path).resolve())
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    chunk = Chunk(
        mimetype="application/pdf",
        origin=Origin(location=path),
        metadata={
            "file_hash": file_hash,
        },
    )
    chunk.id = f"pdf_{hashlib.sha256(path.encode()).hexdigest()}"
    return chunk


def to_origin(pdf_chunk, x1, x2, y1, y2, page_number) -> Origin:
    return Origin(
        source_id=pdf_chunk.id,
        location={
            "bbox": [x1, y1, x2, y2],
            "page": page_number,
        },
    )
