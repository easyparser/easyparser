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

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["sycamore"]


class UnstructuredPDF(BaseOperation):
    supported_mimetypes = ["application/pdf"]
    text_types = {
        "Title",
        "Text",
        "UncategorizedText",
        "NarrativeText",
        "BulletedText",
        "Paragraph",
        "Abstract",
        "Threading",
        "Form",
        "Field-Name",
        "Value",
        "Link",
        "CompositeElement",
        "FigureCaption",
        "Caption",
        "List",
        "ListItem",
        "List-item",
        "Checked",
        "Unchecked",
        "CheckBoxChecked",
        "CheckBoxUnchecked",
        "RadioButtonChecked",
        "RadioButtonUnchecked",
        "Address",
        "EmailAddress",
        "PageBreak",
        "Header",
        "Headline",
        "Subheadline",
        "Page-header",
        "Section-header",
        "Footer",
        "Footnote",
        "Page-footer",
        "PageNumber",
        "CodeSnippet",
        "FormKeysValues",
    }
    image_types = {
        "Image",
        "Picture",
        "Figure",
        "Formula",
        "Table",
    }

    @staticmethod
    def run(
        chunk: Chunk | ChunkGroup, strategy: str = "hi_res", **kwargs
    ) -> ChunkGroup:
        """Parse the PDF using the unstructured partitioning service.

        Args:
            strategy: The strategy to use for partitioning the PDF. Valid
                strategies are "hi_res", "ocr_only", and "fast". When using the
                "hi_res" strategy, the function uses a layout detection model to
                identify document elements. When using the "ocr_only" strategy,
                partition_pdf simply extracts the text from the document using OCR
                and processes it. If the "fast" strategy is used, the text is
                extracted directly from the PDF. The default strategy `auto` will
                determine when a page can be extracted using `fast` mode, otherwise
                it will fall back to `hi_res`.
        """
        import base64

        from unstructured.partition.pdf import partition_pdf

        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])

        result = []
        for c in chunk:
            if c.origin is None:
                raise ValueError("Origin is not defined")

            file_path = c.origin.location
            elements = partition_pdf(
                file_path,
                strategy=strategy,
                extract_image_block_types=list(UnstructuredPDF.image_types),
                extract_image_block_to_payload=True,
                **kwargs,
            )
            for e in elements:
                origin = None
                if coord := e.metadata.coordinates:
                    x1, y1 = coord.points[0]
                    x2, y2 = coord.points[2]
                    width, height = coord.system.width, coord.system.height
                    origin = Origin(
                        source_id=c.id,
                        location={
                            "bbox": [x1 / width, y1 / height, x2 / width, y2 / height],
                            "page": e.metadata.page_number,
                        },
                    )
                text = e.text
                if e.category in UnstructuredPDF.text_types:
                    mimetype = "text/plain"
                    content = e.text
                elif e.category in UnstructuredPDF.image_types:
                    if not e.metadata.image_base64 or not e.metadata.image_mime_type:
                        continue
                    mimetype = e.metadata.image_mime_type
                    content = base64.b64decode(e.metadata.image_base64)
                else:
                    raise ValueError(f"Unknown type: {e.category}")

                result.append(
                    Chunk(
                        mimetype=mimetype,
                        content=content,
                        text=text,
                        parent=c,
                        origin=origin,
                        metadata={
                            "type": e.category,
                            "languages": e.metadata.languages,
                        },
                    )
                )
        return ChunkGroup(chunks=result)

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["unstructured[pdf]"]


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


# def pdf_by_docling(file_path: Path | str, **kwargs) -> list[Snippet]: ...


# def pdf_by_vlm(file_path: Path | str, **kwargs) -> list[Snippet]: ...
