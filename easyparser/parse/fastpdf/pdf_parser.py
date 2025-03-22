"""Convert any document to Markdown."""

import re
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from pdftext.extraction import dictionary_output

from .pdf_table import img2table_get_tables


def scale_bbox(bbox: list[float], width: float, height: float) -> list[float]:
    return [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]


def parsed_pdf_to_markdown(
    pages: list[dict[str, Any]],
) -> list[str]:  # noqa: C901, PLR0915
    """Convert a PDF parsed with pdftext to Markdown."""

    def extract_font_size(span: dict[str, Any]) -> float:
        """Extract the font size from a text span."""
        font_size: float = 1.0
        if (
            span["font"]["size"] > 1
        ):  # A value of 1 appears to mean "unknown" in pdftext.
            font_size = span["font"]["size"]
        elif digit_sequences := re.findall(r"\d+", span["font"]["name"] or ""):
            font_size = float(digit_sequences[-1])
        elif (
            "\n" not in span["text"]
        ):  # Occasionally a span can contain a newline character.
            if round(span["rotation"]) in (0.0, 180.0, -180.0):
                font_size = span["bbox"][3] - span["bbox"][1]
            elif round(span["rotation"]) in (90.0, -90.0, 270.0, -270.0):
                font_size = span["bbox"][2] - span["bbox"][0]
        return font_size

    def get_mode_font_size(
        pages: list[dict[str, Any]],
    ) -> float:
        """Get the mode font size from a list of text spans."""
        font_sizes = np.asarray(
            [
                extract_font_size(span)
                for page in pages
                for block in page["blocks"]
                for line in block["lines"]
                for span in line["spans"]
            ]
        )
        font_sizes = np.round(font_sizes).astype(int)
        mode_font_size = np.bincount(font_sizes).argmax()
        return mode_font_size

    def add_emphasis_metadata(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add emphasis metadata such as
        bold and italic to a PDF parsed with pdftext."""
        # Copy the pages.
        pages = deepcopy(pages)
        # Add emphasis metadata to the text spans.
        for page in pages:
            for block in page["blocks"]:
                for line in block["lines"]:
                    if "md" not in line:
                        line["md"] = {}
                    for span in line["spans"]:
                        if "md" not in span:
                            span["md"] = {}
                        span["md"]["bold"] = (
                            span["font"]["weight"] > 500
                        )  # noqa: PLR2004
                        span["md"]["italic"] = (
                            "ital" in (span["font"]["name"] or "").lower()
                        )
                    line["md"]["bold"] = all(
                        span["md"]["bold"]
                        for span in line["spans"]
                        if span["text"].strip()
                    )
                    line["md"]["italic"] = all(
                        span["md"]["italic"]
                        for span in line["spans"]
                        if span["text"].strip()
                    )
        return pages

    def convert_to_markdown(
        pages: list[dict[str, Any]],
    ) -> list[str]:  # noqa: C901, PLR0912
        """Convert a list of pages to Markdown."""
        pages_md = []
        output_pages = []
        mode_font_size = get_mode_font_size(pages)

        for page in pages:
            page_md = ""
            output_blocks = []

            page_w, page_h = page["width"], page["height"]
            for block in page["blocks"]:
                block_text = ""
                for line in block["lines"]:
                    # Build the line text and style the spans.
                    line_text = ""
                    for span in line["spans"]:
                        if (
                            not line["md"]["bold"]
                            and not line["md"]["italic"]
                            and span["md"]["bold"]
                            and span["md"]["italic"]
                        ):
                            line_text += f"***{span['text']}***"
                        elif not line["md"]["bold"] and span["md"]["bold"]:
                            line_text += f"**{span['text']}**"
                        elif not line["md"]["italic"] and span["md"]["italic"]:
                            line_text += f"*{span['text']}*"
                        else:
                            line_text += span["text"]
                    # Add emphasis to the line (if it's not a heading or whitespace).
                    line_text = line_text.rstrip()
                    line_is_whitespace = not line_text.strip()

                    if not line_is_whitespace:
                        if line["md"]["bold"] and line["md"]["italic"]:
                            line_text = f"***{line_text}***"
                        elif line["md"]["bold"]:
                            line_text = f"**{line_text}**"
                        elif line["md"]["italic"]:
                            line_text = f"*{line_text}*"

                    line_text += "\n"
                    block_text += line_text
                block_text = block_text.rstrip()
                page_md += block_text + "\n\n"

                is_heading_block = all(line["md"]["bold"] for line in block["lines"])
                median_font_size = np.max(
                    np.round(
                        [
                            extract_font_size(span)
                            for line in block["lines"]
                            for span in line["spans"]
                        ]
                    )
                ).astype(int)
                is_heading_block = (
                    is_heading_block and median_font_size >= mode_font_size
                )
                if is_heading_block:
                    block_text = f"### {block_text}"

                output_blocks.append(
                    {
                        "text": block_text,
                        "bbox": scale_bbox(block["bbox"], page_w, page_h),
                        "type": "heading" if is_heading_block else "text",
                    }
                )

            page["blocks"] = output_blocks
            output_pages.append(page)
            pages_md.append(page_md.strip())
        return output_pages

    # Add emphasis metadata.
    pages = add_emphasis_metadata(pages)
    # Convert the pages to Markdown.
    pages = convert_to_markdown(pages)
    return pages


def parition_pdf(
    doc_path: Path,
    executor: ProcessPoolExecutor | None,
    extract_table=False,
) -> str:
    """Convert any document to GitHub Flavored Markdown."""
    # Parse the PDF with pdftext and convert it to Markdown.
    pages = dictionary_output(
        doc_path,
        sort=False,
        keep_chars=False,
        workers=None,
    )
    pages = parsed_pdf_to_markdown(pages)

    if extract_table:
        tables = img2table_get_tables(doc_path, executor=executor)

    for idx, page in enumerate(pages):
        if extract_table:
            text_blocks = page["blocks"]
            table_blocks = tables.get(idx, [])

            # filter blocks overlap with tables base on bbox
            for table in table_blocks:
                table_bbox = table["bbox"]
                text_blocks = [
                    block
                    for block in text_blocks
                    if (
                        block["bbox"][0] >= table_bbox[2]
                        or block["bbox"][1] >= table_bbox[3]
                        or block["bbox"][2] <= table_bbox[0]
                        or block["bbox"][3] <= table_bbox[1]
                    )
                ]
            page["blocks"] = text_blocks + table_blocks
            page.pop("refs", None)

    return pages


def pages_to_markdown(pages: list[dict[str, Any]]) -> list[str]:
    """Convert a list of pages to Markdown."""
    md_text = ""
    for page in pages:
        for block in page["blocks"]:
            md_text += block["text"] + "\n\n"
    return md_text
