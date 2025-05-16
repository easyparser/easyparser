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
MAX_PAGES_TO_CALCULATE_MODE = 5
LINE_JOIN_CHAR = "\n"


def extract_font_size(span: dict[str, Any]) -> float:
    """Extract the font size from a text span."""
    font_size: float = DEFAULT_FONT_SIZE
    if span["font"]["size"] > 1:  # A value of 1 appears to mean "unknown" in pdftext.
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
    pages = pages[:MAX_PAGES_TO_CALCULATE_MODE]
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
    pages: list[dict[str, Any]] | None = None,
    lines: list[dict[str, Any]] | None = None,
) -> float:
    """Get the mode font size from a list of text spans."""
    if pages:
        pages = pages[:MAX_PAGES_TO_CALCULATE_MODE]

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
    elif lines:
        font_weights = np.asarray(
            [
                span["font"]["weight"]
                for line in lines
                for span in line["spans"]
                if span["font"]["weight"] > 0
            ]
        )
    else:
        font_weights = []

    font_weights = np.round(font_weights).astype(int)
    try:
        mode_font_weight = np.bincount(font_weights).argmax()
    except ValueError:
        mode_font_weight = DEFAULT_MODE_FONT_WEIGHT

    return mode_font_weight


def render_emphasis(line: dict[str, Any]) -> str:
    """Render the emphasis for a line."""
    if "md" not in line:
        return "".join(span["text"] for span in line["spans"]).strip()

    line_text = ""
    is_line_special = line["md"]["bold"] or line["md"]["italic"]
    for span in line["spans"]:
        if not is_line_special:
            has_special_char = "*" in span["text"]
            if span["md"]["bold"] and span["md"]["italic"] and not has_special_char:
                line_text += f"***{span['text'].strip()}*** "
            elif span["md"]["bold"] and not has_special_char:
                line_text += f"**{span['text'].strip()}** "
            elif span["md"]["italic"] and not has_special_char:
                line_text += f"*{span['text'].strip()}* "
            else:
                line_text += span["text"].replace("*", "\\*")
        else:
            line_text += span["text"].replace("*", "\\*")

    # Add emphasis to the line (if it's not a whitespace).
    line_is_whitespace = not line_text.strip()

    if not line_is_whitespace:
        if line["md"]["bold"] and line["md"]["italic"]:
            line_text = f"***{line_text.strip()}*** "
        elif line["md"]["bold"]:
            line_text = f"**{line_text.strip()}** "
        elif line["md"]["italic"]:
            line_text = f"*{line_text.strip()}* "

    return line_text


def add_emphasis_metadata(
    pages: list[dict[str, Any]] | None = None,
    lines: list[dict[str, Any]] | None = None,
    mode_font_weight: int | None = None,
) -> list[dict[str, Any]]:
    """Add emphasis metadata such as
    bold and italic to a PDF parsed with pdftext."""
    if pages is None and lines is None:
        raise ValueError("Either pages or lines must be provided.")

    # Copy the pages.
    pages = deepcopy(pages)
    if mode_font_weight is None:
        mode_font_weight = max(
            get_mode_font_weight(
                pages=pages,
                lines=lines,
            ),
            DEFAULT_MODE_FONT_WEIGHT,
        )

    # Add emphasis metadata to the text spans.
    if pages is not None:
        all_lines = [
            line
            for page in pages
            for block in page["blocks"]
            for line in block["lines"]
        ]
    else:
        all_lines = lines

    for line in all_lines:
        if "md" not in line:
            line["md"] = {}
        for span in line["spans"]:
            if "md" not in span:
                span["md"] = {}
            span["md"]["bold"] = span["font"]["weight"] > mode_font_weight
            span["md"]["italic"] = "ital" in (span["font"]["name"] or "").lower()
        line["md"]["bold"] = all(
            span["md"]["bold"] for span in line["spans"] if span["text"].strip()
        )
        line["md"]["italic"] = all(
            span["md"]["italic"] for span in line["spans"] if span["text"].strip()
        )

    if pages is not None:
        return pages

    return all_lines


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
                line_text = render_emphasis(line)
                line_text += LINE_JOIN_CHAR
                block_text += line_text

            block_text = block_text.replace("\n", " ").replace("\r", "").strip()

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


def parsed_pdf_to_markdown(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a PDF parsed with pdftext to Markdown."""
    # Add emphasis metadata.
    pages = add_emphasis_metadata(pages=pages)
    # Convert the pages to Markdown.
    pages = add_markdown_format(pages)
    return pages


def parition_pdf_heuristic(
    doc_path: Path | str,
    executor: ProcessPoolExecutor | None = None,
    render_scale: float = 1.5,
    extract_table: bool = False,
    extract_image: bool = True,
    extract_page: bool = False,
) -> str:
    """Convert PDF document to Markdown."""
    # Parse the PDF with pdftext and convert it to Markdown.
    pages = dictionary_output(
        doc_path,
        sort=False,
        keep_chars=False,
        workers=None,
    )
    pages = parsed_pdf_to_markdown(pages)
    all_images = {}
    all_tables = {}

    if extract_image or extract_page:
        page_images, all_images = get_images_pdfium(
            doc_path,
            render_scale=render_scale,
        )
    else:
        page_images = []

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

        if extract_page:
            page["page_image"] = page_images[idx]
            page["page_text"] = f"Page {idx}"

        page["blocks"] += image_blocks
        page.pop("refs", None)

    return pages
