import base64
from collections import defaultdict
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

SPAN_JOIN_CHAR = " "

MIN_W_MULTIPLIER = 0.7
MIN_H_MULTIPLIER = 0.5

RESCALE_DECREMENT = 0.1
DEFAULT_W_MULTIPLIER = 0.9
DEFAULT_H_MULTIPLIER = 1.1

MIN_SPAN_TEXT_LENGTH = 4
LINE_BREAK_CONFLICT_THRESHOLD = 2
LINE_BREAK_TOLERANCE_THRESHOLD = 8

VIZ_RECTANGLE_THICKNESS = 2


class OCRMode(StrEnum):
    """OCR mode for PDF layout parser."""

    AUTO = "auto"
    OFF = "off"
    ON = "on"


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
            VIZ_RECTANGLE_THICKNESS,
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


def get_bbox_w(bbox: list[float]) -> float:
    return bbox[2] - bbox[0]


def get_bbox_h(bbox: list[float]) -> float:
    return bbox[3] - bbox[1]


def get_bbox_area(bbox: list[float]) -> float:
    return get_bbox_w(bbox) * get_bbox_h(bbox)


def get_overlap_ratio(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Calculate the intersection over union (IoU) area of two bounding boxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox_b_area = get_bbox_area(bbox_b)

    # custom union area
    union_area = bbox_b_area
    if union_area > 0:
        return intersection_area / union_area if union_area > 0 else 0

    return 0


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


def filter_out_of_bounds_spans(
    spans: list[dict[str, Any]], img_w: int, img_h: int
) -> list[dict[str, Any]]:
    """Filter out spans that are out of bounds of the image."""
    filtered_spans = []
    for span in spans:
        scaled_bbox = scale_bbox(span["bbox"], img_w, img_h)
        if is_valid_bbox(scaled_bbox):
            filtered_spans.append(span)
    return filtered_spans


def is_valid_span(span: dict[str, Any]) -> bool:
    span_width = get_bbox_w(span["bbox"])
    span_height = get_bbox_h(span["bbox"])
    return (
        span_width > 0
        and span_height > 0
        and len(span["text"]) > 0
        and (span_width / span_height > 1 or len(span["text"]) < MIN_SPAN_TEXT_LENGTH)
    )


def is_valid_bbox(bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 <= 1 and y2 <= 1


def spans_to_layout_text(
    spans: list[dict[str, Any]] = None,
    lines: list[dict[str, Any]] = None,
    w_multiplier: float = DEFAULT_W_MULTIPLIER,
    h_multiplier: float = DEFAULT_H_MULTIPLIER,
    filter_invalid_spans: bool = True,
    strip_spaces: bool = True,
    rescale_on_conflict: bool = False,
    add_line_break_on_conflict: bool = True,
    sort_span: bool = False,
) -> str:
    def get_span_anchor_x(span: dict[str, Any]) -> float:
        return span["bbox"][0]

    def get_span_anchor_y(span: dict[str, Any]) -> float:
        if "line_id" in span:
            return lines[span["line_id"]]["bbox"][3]
        else:
            return span["bbox"][3]

    def get_non_space_text(span: dict[str, Any]) -> str:
        return span["text"].replace("\r", "").replace("\n", "").strip()

    if lines is None and spans is None:
        raise ValueError("Either lines or spans must be provided.")

    if lines is not None:
        # create spans from lines
        spans = []
        for line_id, line in enumerate(lines):
            for span_id, span in enumerate(line["spans"]):
                if get_non_space_text(span):
                    spans.append(
                        {
                            "text": span["text"],
                            "bbox": span["bbox"],
                            "line_id": line_id,
                            "start_of_line": span_id == 0,
                        }
                    )

    if filter_invalid_spans:
        spans = [span for span in spans if is_valid_span(span)]

    if len(spans) == 0:
        return ""

    median_c_height = (
        np.median([get_bbox_h(span["bbox"]) for span in spans]) * h_multiplier
    )
    median_c_width = (
        np.median([get_bbox_w(span["bbox"]) / len(span["text"]) for span in spans])
        * w_multiplier
    )

    bottom_pos = min(get_span_anchor_y(span) for span in spans)
    left_pos = min(get_span_anchor_x(span) for span in spans)

    # map bbox to grid
    try:
        mapped_bottomleft_spans_pos = [
            (
                int(round((get_span_anchor_x(span) - left_pos) / median_c_width)),
                int(round((get_span_anchor_y(span) - bottom_pos) / median_c_height)),
            )
            for span in spans
        ]
    except (ValueError, ZeroDivisionError):
        print("Error in mapping spans to grid positions.")
        return SPAN_JOIN_CHAR.join(span["text"] for span in spans)

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
        if sort_span:
            row.sort(key=lambda x: x[0])

        line = ""
        for col_idx, span_idx in row:
            span = spans[span_idx]
            is_horizontal_conflict = (
                col_idx < (len(line) - LINE_BREAK_CONFLICT_THRESHOLD)
                and len(get_non_space_text(span)) > 0
            )
            if is_horizontal_conflict:
                if rescale_on_conflict:
                    # rescale horizontal space
                    if w_multiplier - RESCALE_DECREMENT >= MIN_W_MULTIPLIER:
                        w_multiplier -= RESCALE_DECREMENT
                        return spans_to_layout_text(
                            spans=spans,
                            lines=lines,
                            w_multiplier=w_multiplier,
                            h_multiplier=h_multiplier,
                            filter_invalid_spans=filter_invalid_spans,
                            strip_spaces=strip_spaces,
                            rescale_on_conflict=rescale_on_conflict,
                            add_line_break_on_conflict=add_line_break_on_conflict,
                            sort_span=sort_span,
                        )
                    elif h_multiplier - RESCALE_DECREMENT >= MIN_H_MULTIPLIER:
                        # rescale vertical space
                        h_multiplier -= RESCALE_DECREMENT
                        return spans_to_layout_text(
                            spans=spans,
                            lines=lines,
                            w_multiplier=w_multiplier,
                            h_multiplier=h_multiplier,
                            filter_invalid_spans=filter_invalid_spans,
                            strip_spaces=strip_spaces,
                            rescale_on_conflict=rescale_on_conflict,
                            add_line_break_on_conflict=add_line_break_on_conflict,
                            sort_span=sort_span,
                        )
                # if span is start_of_line, add a line-break char
                if lines is not None and add_line_break_on_conflict:
                    if (
                        span["start_of_line"]
                        and len(line.strip().replace(" ", ""))
                        > LINE_BREAK_TOLERANCE_THRESHOLD
                    ) or (abs(col_idx - len(line)) > LINE_BREAK_TOLERANCE_THRESHOLD):
                        rendered += line + "\n"
                        line = ""

            line += " " * max(1, col_idx - len(line))
            span_text = span["text"].replace("\r", "").replace("\n", "")
            if strip_spaces:
                span_text = span_text.strip()
            line += span_text

        rendered += line + "\n"

    return rendered


def optimize_2d_rendered_text(
    rendered_text: str,
    optimize_rows: bool = True,
    optimize_cols: bool = False,
) -> str:
    """Optimize the 2d rendered text by removing extra spaces.
    Basically, there is no consecutive blank rows or columns in the text.
    """
    if not rendered_text:
        return ""

    # Split into lines
    lines = rendered_text.split("\n")
    if not lines:
        return ""

    if optimize_rows:
        # Remove consecutive empty lines
        optimized_lines = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if not (is_empty and prev_empty):
                optimized_lines.append(line)
            prev_empty = is_empty

        # Remove trailing empty lines
        while optimized_lines and not optimized_lines[-1].strip():
            optimized_lines.pop()

        # Remove leading empty lines
        while optimized_lines and not optimized_lines[0].strip():
            optimized_lines.pop(0)

        if not optimized_lines:
            return ""
    else:
        optimized_lines = lines

    if optimize_cols:
        # Find max line length
        max_len = max(len(line) for line in optimized_lines)
        # Pad lines to same length
        padded_lines = [line.ljust(max_len) for line in optimized_lines]

        # Remove consecutive space columns
        optimized_cols = []
        prev_empty_col = False
        for col_idx in range(max_len):
            col = [line[col_idx] for line in padded_lines]
            is_empty_col = all(c.isspace() for c in col)

            if not (is_empty_col and prev_empty_col):
                optimized_cols.append(col_idx)
            prev_empty_col = is_empty_col

        # Reconstruct text using optimized columns
        new_lines = []
        for line in optimized_lines:
            if len(line) > 0:
                new_line = "".join(line[i] for i in optimized_cols if i < len(line))
                new_lines.append(new_line.rstrip())

        return "\n".join(new_lines)

    return "\n".join(optimized_lines)


def get_block_order(block):
    try:
        order = max(
            [span["order"] for line in block["lines"] for span in line["spans"]]
        )
    except ValueError:
        order = -1
    return order


def group_lines_by_span_order(
    lines: list[dict[str, Any]],
    page_w: int,
    page_h: int,
) -> list[dict[str, Any]]:
    # group left over lines by consecutive span order
    lines = sorted(lines, key=lambda x: x["spans"][0]["order"])

    def group_to_block(group):
        group_spans = filter_out_of_bounds_spans(
            [span for line in group for span in line["spans"]],
            page_w,
            page_h,
        )
        if group_spans:
            return {
                "lines": group,
                "bbox": union_bbox([span["bbox"] for span in group_spans]),
                "type": "text",
            }
        return None

    # group lines by consecutive span order
    cur_group = []
    blocks = []
    last_span_order = -1
    while lines:
        line = lines.pop(0)
        if not line["spans"]:
            continue
        first_span_order = line["spans"][0]["order"]
        if not cur_group or first_span_order - last_span_order == 1:
            cur_group.append(line)
        else:
            if cur_group:
                cur_block = group_to_block(cur_group)
                if cur_block:
                    blocks.append(cur_block)
            # create new group
            cur_group = [line]

        last_span_order = line["spans"][-1]["order"]

    if cur_group:
        cur_block = group_to_block(cur_group)
        if cur_block:
            blocks.append(cur_block)

    return blocks


def fix_unicode_encoding(text: str) -> str:
    """Fix unicode encoding issues in the text."""
    return text.encode("utf16", "surrogatepass").decode("utf16", "ignore")


def pages_to_markdown(pages: list[dict[str, Any]]) -> list[str]:
    """Convert a list of pages to Markdown."""
    md_text = ""
    for page in pages:
        for block in page["blocks"]:
            md_text += block["text"] + "\n\n"
    return md_text


def is_2d_layout(spans, thres=0.5) -> bool:
    """
    Check if the combined area of the spans
    is greater than the threshold of union area
    """
    if len(spans) == 0:
        return False

    combined_bbox = union_bbox([span["bbox"] for span in spans])
    combined_area = get_bbox_area(combined_bbox)

    if not combined_area:
        return False

    sum_span_area = sum(get_bbox_area(span["bbox"]) for span in spans)
    ratio = sum_span_area / combined_area
    return ratio < thres


def crop_img_and_export_base64(img: np.ndarray, box: list[float]) -> str:
    """Crop the image and export it as base64."""
    x1, y1, x2, y2 = box
    x1 = int(x1 * img.shape[1])
    y1 = int(y1 * img.shape[0])
    x2 = int(x2 * img.shape[1])
    y2 = int(y2 * img.shape[0])

    cropped_img = img[y1:y2, x1:x2]
    _, buffer = cv2.imencode(".png", cropped_img)  # or '.jpg'
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return "data:image/png;base64," + img_base64


def crop_img_and_export_bytes(img: np.ndarray, box: list[float]) -> bytes:
    """Crop the image and export it as base64."""
    x1, y1, x2, y2 = box
    x1 = int(x1 * img.shape[1])
    y1 = int(y1 * img.shape[0])
    x2 = int(x2 * img.shape[1])
    y2 = int(y2 * img.shape[0])

    cropped_img = img[y1:y2, x1:x2]
    _, buffer = cv2.imencode(".png", cropped_img)  # or '.jpg'
    return buffer.tobytes()


def bytes_to_base64(bytes: Any, mime_type: str) -> str:
    """Convert a PIL image to base64."""
    img_base64 = base64.b64encode(bytes).decode("utf-8")
    return f"data:{mime_type};base64,{img_base64}"
