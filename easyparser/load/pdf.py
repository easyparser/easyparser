from ..base import BaseOperation, Chunk, ChunkGroup, Origin


class SycamorePDF(BaseOperation):
    """Parsing PDF using sycamore

    Pros:
        - Maintain smaller semantic elements, don't lump into text
        - Have dedicated types for formulas, images, tables
    Cons:
        - Cannot specify extracting images for formulas and tables (data types that
        are easy to misrepresented in text)
    Need:
    """

    supported_mimetypes = ["application/pdf"]
    text_types = {
        "Page-header",
        "Section-header",
        "Title",
        "Caption",
        "Page-footer",
        "List-item",
        "Text",
        "Footnote",
    }
    image_types = {"Formula", "Image", "table", "Table"}

    def __init__(self, **kwargs):
        try:
            import sycamore  # noqa
        except ImportError:
            raise ImportError("Please install `pip install sycamore`")
        super().__init__()
        self._default_params.update(kwargs)

    @staticmethod
    def run(
        chunk: Chunk | ChunkGroup,
        use_partitioning_service: bool = False,
        extract_table_structure: bool = True,
        use_ocr: bool = True,
        extract_images: bool = True,
        **kwargs,
    ) -> ChunkGroup:
        """Load the PDF with Sycamore PDF extractor.

        This extractor parses all elements in the PDF and return them as chunks. They
        don't lump all text into a single text chunk, but instead maintain the original
        structure of the PDF. It also has dedicated types for formulas, images, and
        tables.

        Args:
            use_partitioning_service: If true, uses the online Aryn
                partitioning service to extract the text from the PDF.
            extract_table_structure: If true, runs a separate table extraction
                model to extract cells from regions of the document identified as
                tables.
            use_ocr: Whether to use OCR to extract text from the PDF. If false,
                we will attempt to extract the text from the underlying PDF.
            extract_images: If true, crops each region identified as an image.
        """
        import sycamore
        from sycamore.transforms.partition import ArynPartitioner

        partitioner = ArynPartitioner(
            use_partitioning_service=use_partitioning_service,
            extract_table_structure=extract_table_structure,
            use_ocr=use_ocr,
            extract_images=extract_images,
        )
        context = sycamore.init(exec_mode=sycamore.ExecMode.LOCAL)

        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])

        result = []
        for c in chunk:
            if c.origin is None:
                raise ValueError("Origin is not defined")
            docset = context.read.binary(
                paths=str(c.origin.location), binary_format="pdf"
            ).partition(partitioner=partitioner)
            doc = docset.take_all()[0]
            for e in doc.elements:
                origin = None
                if e.bbox:
                    origin = Origin(
                        source_id=c.id,
                        location={
                            "bbox": [e.bbox.x1, e.bbox.y1, e.bbox.x2, e.bbox.y2],
                            "page": e.properties["page_number"],
                        },
                    )

                text = e.text_representation.strip() if e.text_representation else ""
                if e.type in SycamorePDF.text_types:
                    mimetype = "text/plain"
                    content = e.text_representation
                elif e.type in SycamorePDF.image_types:
                    mimetype = "image/png"
                    content = e.binary_representation
                else:
                    raise ValueError(f"Unknown type: {e.type}")

                result.append(
                    Chunk(
                        mimetype=mimetype,
                        content=content,
                        text=text,
                        parent=c,
                        origin=origin,
                        metadata={"type": e.type, **e.properties},
                    )
                )

        for idx, c in enumerate(result[1:], start=1):
            c.prev = result[idx - 1]
            result[idx - 1].next = c

        return ChunkGroup(chunks=result)


# def pdf_by_extractous(file_path: Path | str) -> list[Snippet]:
#     try:
#         from extractous import Extractor
#     except ImportError:
#         raise ImportError("Please install `pip install extractous`")

#     file_path = Path(file_path)
#     extr = Extractor()
#     result, metadata = extr.extract_file_to_string(str(file_path))
#     return [
#         Snippet(
#             id=Path(file_path).stem,
#             text=result,
#             content=result,
#             dtype="text",
#             origin=Origin(
#                 source_id=file_path.as_posix(),
#                 location={},
#                 getter="",
#                 file_type="pdf",
#             ),
#             metadata=metadata,
#         )
#     ]


# def pdf_by_unstructured(file_path: Path | str, **kwargs) -> list[Snippet]:
#     try:
#         from unstructured.partition.pdf import partition_pdf
#     except ImportError:
#         raise ImportError('Please install `pip install "unstructured[pdf]"`')

#     file_path = Path(file_path).resolve()
#     file_path_str = str(file_path)
#     if "chunking_strategy" not in kwargs:
#         kwargs["chunking_strategy"] = "basic"
#     result = partition_pdf(file_path_str, **kwargs)

#     output = []
#     for element in result:
#         coord = (
#             element.metadata.coordinates.to_dict()
#             if element.metadata.coordinates
#             else {}
#         )
#         output.append(
#             Snippet(
#                 id=element.id,
#                 text=element.text,
#                 content=element.text,
#                 dtype="text",
#                 origin=Origin(
#                     source_id=file_path_str,
#                     location=coord,
#                     getter="",
#                     file_type="pdf",
#                 ),
#                 metadata=element.metadata.to_dict(),
#             )
#         )
#     return output


# def pdf_by_docling(file_path: Path | str, **kwargs) -> list[Snippet]: ...


# def pdf_by_vlm(file_path: Path | str, **kwargs) -> list[Snippet]: ...
