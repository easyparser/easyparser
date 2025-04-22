from collections import defaultdict
from typing import Any

import cv2
import numpy as np

SPAN_JOIN_CHAR = " "
SPAN_LINE_BREAK_CHAR = "⏎"
MIN_W_MULTIPLIER = 0.6


def draw_bboxes(img, bboxes, color=(0, 255, 0)):
    img_h, img_w = img.shape[:2]
    for box in bboxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * img_w)
        y1 = int(y1 * img_h)
        x2 = int(x2 * img_w)
        y2 = int(y2 * img_h)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )


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


def get_overlap_ratio(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Calculate the intersection over union (IoU) area of two bounding boxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox_a_area = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    bbox_b_area = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    # custom union area
    union_area = min(bbox_a_area, bbox_b_area)
    return intersection_area / union_area if union_area > 0 else 0


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


def merge_text_and_table_blocks(
    text_blocks: list[dict], table_blocks: list[dict]
) -> list[dict]:
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
                    text_blocks[text_bid].update(
                        {
                            "lines": non_overlap_lines,
                            "bbox": union_bbox(
                                [line["bbox"] for line in non_overlap_lines]
                            ),
                            "text": SPAN_JOIN_CHAR.join(
                                line["text"] for line in non_overlap_lines
                            ),
                        }
                    )
                else:
                    # mark this text block for later removal
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

    return text_with_table_blocks


def is_valid_span(span: dict[str, Any]) -> bool:
    span_width = span["bbox"][2] - span["bbox"][0]
    span_height = span["bbox"][3] - span["bbox"][1]
    return (
        span_width > 0
        and span_height > 0
        and len(span["text"]) > 0
        and (span_width / span_height > 1 or len(span["text"]) < 4)
    )


def spans_to_layout_text(
    spans: list[dict[str, Any]],
    w_multiplier: float = 0.9,
    h_multiplier: float = 1.1,
    filter_invalid_spans: bool = True,
    strip_spaces: bool = True,
    scale_on_conflict: bool = False,
) -> str:
    if filter_invalid_spans:
        spans = [span for span in spans if is_valid_span(span)]

    if len(spans) == 0:
        return ""

    median_c_height = (
        np.median([span["bbox"][3] - span["bbox"][1] for span in spans]) * h_multiplier
    )
    median_c_width = (
        np.median(
            [(span["bbox"][2] - span["bbox"][0]) / len(span["text"]) for span in spans]
        )
        * w_multiplier
    )

    bottom_pos = min(span["bbox"][3] for span in spans)
    left_pos = min(span["bbox"][0] for span in spans)

    # map bbox to grid
    mapped_bottomleft_spans_pos = [
        (
            int(round((span["bbox"][0] - left_pos) / median_c_width)),
            int(round((span["bbox"][3] - bottom_pos) / median_c_height)),
        )
        for span in spans
    ]

    # group spans by row
    rows = defaultdict(list)
    for idx, (x, y) in enumerate(mapped_bottomleft_spans_pos):
        rows[y].append((x, idx))

    # render the final text
    rendered = ""
    for row_idx in sorted(rows.keys()):
        row = rows[row_idx]

        # add newlines if needed
        cur_newline_count = rendered.count("\n")
        rendered += "\n" * (row_idx - cur_newline_count)

        # sort the spans in the row by x position
        row.sort(key=lambda x: x[0])

        line = ""
        for col_idx, span_idx in row:
            span = spans[span_idx]
            if scale_on_conflict and col_idx < len(line) - 1:
                # rescale horizontal space
                w_multiplier -= 0.15
                if w_multiplier >= MIN_W_MULTIPLIER:
                    print("Rescaling w_multiplier to", w_multiplier)
                    return spans_to_layout_text(
                        spans,
                        w_multiplier=w_multiplier,
                        h_multiplier=h_multiplier,
                        filter_invalid_spans=filter_invalid_spans,
                        strip_spaces=strip_spaces,
                        scale_on_conflict=scale_on_conflict,
                    )

            line += " " * max(1, col_idx - len(line))
            span_text = span["text"].replace("\r", "").replace("\n", "")
            if strip_spaces:
                span_text = span_text.strip()
            line += span_text

        rendered += line + "\n"

    return rendered


def fix_unicode_encoding(text: str) -> str:
    """Fix unicode encoding issues in the text."""
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
