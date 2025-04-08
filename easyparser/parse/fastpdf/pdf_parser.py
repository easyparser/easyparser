"""Convert any document to Markdown."""

import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from pdftext.extraction import dictionary_output

from .pdf_table import get_images_pdfium, get_tables_img2table

DEFAULT_FONT_SIZE = 1.0
DEFAULT_MODE_FONT_SIZE = 10
DEFAULT_MODE_FONT_WEIGHT = 350
HEADER_MAX_LENGTH = 200
LINE_JOIN_CHAR = "\n"
SPAN_JOIN_CHAR = " "


def scale_bbox(bbox: list[float], width: float, height: float) -> list[float]:
    return [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]


def is_bbox_overlap(bbox_a: list[float], bbox_b: list[float]) -> bool:
    """Check if two bounding boxes overlap."""
    return not (
        bbox_a[0] >= bbox_b[2]
        or bbox_a[1] >= bbox_b[3]
        or bbox_a[2] <= bbox_b[0]
        or bbox_a[3] <= bbox_b[1]
    )


def union_bbox(bbox_list: list[list[float]]) -> list[float]:
    """Get the union of a list of bounding boxes."""
    min_x = min(bbox[0] for bbox in bbox_list)
    min_y = min(bbox[1] for bbox in bbox_list)
    max_x = max(bbox[2] for bbox in bbox_list)
    max_y = max(bbox[3] for bbox in bbox_list)
    return [min_x, min_y, max_x, max_y]


def get_non_overlap_lines(
    lines: list[dict[str, Any]],
    bbox: list[float],
) -> list[dict[str, Any]]:
    """Get the lines that do not overlap with a given bounding box."""
    non_overlap_lines = []
    for line in lines:
        line_bbox = line["bbox"]
        if not is_bbox_overlap(line_bbox, bbox):
            non_overlap_lines.append(line)
    return non_overlap_lines


def parsed_pdf_to_markdown(
    pages: list[dict[str, Any]],
) -> list[str]:
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

                    line["text"] = line_text
                    line_text += LINE_JOIN_CHAR
                    block_text += line_text
                block_text = block_text.rstrip()
                page_md += block_text + "\n\n"

                is_heading_block = (
                    all(line["md"]["bold"] for line in block["lines"])
                    and len(block_text.strip()) < HEADER_MAX_LENGTH
                )
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
                    is_heading_block
                    and median_font_size >= mode_font_size
                    and block_text.strip()
                )
                if is_heading_block:
                    block_text = f"### {block_text}"

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
            pages_md.append(page_md.strip())
        return output_pages

    # Add emphasis metadata.
    pages = add_emphasis_metadata(pages)
    # Convert the pages to Markdown.
    pages = add_markdown_format(pages)
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
    all_images = get_images_pdfium(doc_path)

    if extract_table:
        all_tables = get_tables_img2table(doc_path, executor=executor)

    for idx, page in enumerate(pages):
        if extract_table:
            text_blocks = page["blocks"]
            table_blocks = all_tables.get(idx, [])
            image_blocks = all_images.get(idx, [])

            block_to_table_mapping = defaultdict(list)

            # filter blocks overlap with tables base on bbox
            for text_bid, text_block in enumerate(text_blocks):
                text_bbox = text_block["bbox"]
                for table_bid, table in enumerate(table_blocks):
                    table_bbox = table["bbox"]
                    if is_bbox_overlap(text_bbox, table_bbox):
                        non_overlap_lines = get_non_overlap_lines(
                            text_block["lines"],
                            table_bbox,
                        )
                        if non_overlap_lines:
                            # update the text block with non-overlapping lines
                            text_blocks[text_bid]["lines"] = non_overlap_lines
                            text_blocks[text_bid]["bbox"] = union_bbox(
                                [line["bbox"] for line in non_overlap_lines]
                            )
                            text_blocks[text_bid]["text"] = SPAN_JOIN_CHAR.join(
                                line["text"] for line in non_overlap_lines
                            )
                        else:
                            text_blocks[text_bid] = None
                        block_to_table_mapping[text_bid].append(table_bid)

            # join the text blocks with the table blocks
            # and preserve the reading order
            text_with_table_blocks = []
            merged_table_indices = set()
            for text_bid, text_block in enumerate(text_blocks):
                if text_block is not None:
                    text_with_table_blocks.append(text_block)

                if block_to_table_mapping[text_bid]:
                    for table_bid in block_to_table_mapping[text_bid]:
                        if table_bid in merged_table_indices:
                            continue
                        text_with_table_blocks.append(table_blocks[table_bid])
                        merged_table_indices.add(table_bid)

            page["blocks"] = text_with_table_blocks + image_blocks
            page.pop("refs", None)

    return pages


def pages_to_markdown(pages: list[dict[str, Any]]) -> list[str]:
    """Convert a list of pages to Markdown."""
    md_text = ""
    for page in pages:
        for block in page["blocks"]:
            md_text += block["text"] + "\n\n"
    return md_text
