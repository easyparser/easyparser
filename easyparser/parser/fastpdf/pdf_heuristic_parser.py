"""PDF heuristic parser for extracting text and images.
Loosely based on
https://github.com/superlinear-ai/raglite/blob/main/src/raglite/_markdown.py
(using pdftext and pdfium for metadata extraction from PDF).
"""

import re
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from pdftext.extraction import dictionary_output

from .pdf_image import get_images_pdfium
from .pdf_table import get_tables_img2table
from .util import merge_text_and_table_blocks, scale_bbox

DEFAULT_FONT_SIZE = 1.0
DEFAULT_MODE_FONT_SIZE = 10
DEFAULT_MODE_FONT_WEIGHT = 350
HEADER_MAX_LENGTH = 200
LINE_JOIN_CHAR = "\n"


def parsed_pdf_to_markdown(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a PDF parsed with pdftext to Markdown."""

    def extract_font_size(span: dict[str, Any]) -> float:
        """Extract the font size from a text span."""
        font_size: float = DEFAULT_FONT_SIZE
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

        try:
            mode_font_size = np.bincount(font_sizes).argmax()
        except ValueError:
            mode_font_size = DEFAULT_MODE_FONT_SIZE
        return mode_font_size

    def get_mode_font_weight(
        pages: list[dict[str, Any]],
    ) -> float:
        """Get the mode font size from a list of text spans."""
        font_weights = np.asarray(
            [
                span["font"]["weight"]
                for page in pages
                for block in page["blocks"]
                for line in block["lines"]
                for span in line["spans"]
                if span["font"]["weight"] > 0
            ]
        )
        font_weights = np.round(font_weights).astype(int)

        try:
            mode_font_weight = np.bincount(font_weights).argmax()
        except ValueError:
            mode_font_weight = DEFAULT_MODE_FONT_WEIGHT

        return mode_font_weight

    def add_emphasis_metadata(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add emphasis metadata such as
        bold and italic to a PDF parsed with pdftext."""
        # Copy the pages.
        pages = deepcopy(pages)
        mode_font_weight = max(
            get_mode_font_weight(pages),
            DEFAULT_MODE_FONT_WEIGHT,
        )

        # Add emphasis metadata to the text spans.
        for page in pages:
            for block in page["blocks"]:
                for line in block["lines"]:
                    if "md" not in line:
                        line["md"] = {}
                    for span in line["spans"]:
                        if "md" not in span:
                            span["md"] = {}
                        span["md"]["bold"] = span["font"]["weight"] > mode_font_weight
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

    def add_markdown_format(
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert a list of pages to Markdown."""
        output_pages = []
        mode_font_size = get_mode_font_size(pages)

        for page in pages:
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

                    # Add emphasis to the line (if it's not a whitespace).
                    line_text = line_text.rstrip()
                    line_is_whitespace = not line_text.strip()

                    if not line_is_whitespace:
                        if line["md"]["bold"] and line["md"]["italic"]:
                            line_text = f"***{line_text}***"
                        elif line["md"]["bold"]:
                            line_text = f"**{line_text}**"
                        elif line["md"]["italic"]:
                            line_text = f"*{line_text}*"

                    line["text"] = line_text
                    line_text += LINE_JOIN_CHAR
                    block_text += line_text
                block_text = block_text.rstrip()

                is_heading_block = (
                    all(line["md"]["bold"] for line in block["lines"])
                    and len(block_text.strip()) < HEADER_MAX_LENGTH
                )
                block_font_size = np.max(
                    np.round(
                        [
                            extract_font_size(span)
                            for line in block["lines"]
                            for span in line["spans"]
                        ]
                    )
                ).astype(int)
                is_heading_block = (
                    is_heading_block
                    and block_font_size >= mode_font_size
                    and block_text.strip()
                )

                output_blocks.append(
                    {
                        "text": block_text,
                        "bbox": scale_bbox(block["bbox"], page_w, page_h),
                        "lines": [
                            {
                                "bbox": scale_bbox(span["bbox"], page_w, page_h),
                                "text": span["text"],
                            }
                            for line in block["lines"]
                            for span in line["spans"]
                        ],
                        "type": "heading" if is_heading_block else "text",
                    }
                )

            page["blocks"] = output_blocks
            output_pages.append(page)
        return output_pages

    # Add emphasis metadata.
    pages = add_emphasis_metadata(pages)
    # Convert the pages to Markdown.
    pages = add_markdown_format(pages)
    return pages


def parition_pdf_heuristic(
    doc_path: Path | str,
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
    all_images = get_images_pdfium(doc_path)

    if extract_table:
        all_tables = get_tables_img2table(doc_path, executor=executor)

    for idx, page in enumerate(pages):
        image_blocks = all_images.get(idx, [])

        if extract_table:
            text_blocks = page["blocks"]
            table_blocks = all_tables.get(idx, [])
            page["blocks"] = merge_text_and_table_blocks(
                text_blocks,
                table_blocks,
            )

        page["blocks"] += image_blocks
        page.pop("refs", None)

    return pages


def pages_to_markdown(pages: list[dict[str, Any]]) -> list[str]:
    """Convert a list of pages to Markdown."""
    md_text = ""
    for page in pages:
        for block in page["blocks"]:
            md_text += block["text"] + "\n\n"
    return md_text
