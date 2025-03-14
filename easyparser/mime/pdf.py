from easyparser.base import Chunk, Origin


class MimeTypePDF:
    """Collection of schemas, metadata and operations for chunks from PDF

    - The element types that chunks in PDF can be classified into.
    - How to represent location of a chunk in a PDF.
    - How to crop / capture a chunk from a PDF.
    - How to render a chunk and its subsequent children in a PDF.
    - How chunks can be combined, or broken down, ensuring relationship, metadata
    and structure is preserved.
    """

    TYPES = [
        "text",  # block of text
        # important for structure analysis, inferring sections
        "heading",
        # important for multi-modal usage
        "image",  # if contains image
        "table",  # if contains table
        "formula",  # if contains formula
        "checkbox",  # if contains checkbox
    ]

    @classmethod
    def to_chunk(cls, path: str) -> Chunk:
        """From a pdf file to a base chunk"""
        return Chunk(mimetype="application/pdf", origin=Origin(location=path))

    @classmethod
    def to_origin(cls, pdf_chunk, x1, x2, y1, y2, page_number) -> Origin:
        return Origin(source_id=pdf_chunk.id)
