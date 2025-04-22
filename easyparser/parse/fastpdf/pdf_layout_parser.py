import math
from pathlib import Path
from typing import Any

import cv2
import pypdfium2
from pdftext.pdf.chars import deduplicate_chars, get_chars
from pdftext.pdf.pages import assign_scripts, get_lines, get_spans
from rapid_layout import RapidLayout, VisLayout

from easyparser.parse.fastpdf.util import (
    fix_unicode_encoding,
    get_overlap_ratio,
    scale_bbox,
    spans_to_layout_text,
    union_bbox,
)


def assign_lines_to_blocks(
    lines: list[dict[str, Any]],
    blocks_by_class: dict[str, list[dict[str, Any]]],
    iou_threshold: float = 0.3,
) -> tuple[dict[str, list[dict[str, Any]]], set[int]]:
    """Assign lines to blocks based on IoU."""
    assigned_line_indices = set()

    for class_name in CLASS_LIST:
        for block in blocks_by_class[class_name]:
            block["lines"] = []
            block_bbox = block["bbox"]
            for line_idx, line in enumerate(lines):
                if line_idx in assigned_line_indices:
                    continue
                line_bbox = line["bbox"]
                overlap_ratio = get_overlap_ratio(block_bbox, line_bbox)
                if overlap_ratio > iou_threshold:
                    block["lines"].append(line)
                    assigned_line_indices.add(line_idx)

    left_over_line_indices = set(range(len(lines))) - assigned_line_indices
    return blocks_by_class, left_over_line_indices


def get_block_order(block):
    try:
        order = max(
            [span["order"] for line in block["lines"] for span in line["spans"]]
        )
    except ValueError:
        order = -1
    return order


def group_lines_by_span_order(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # group left over lines by consecutive span order
    lines = sorted(lines, key=lambda x: x["spans"][0]["order"])

    def group_to_block(group):
        return {
            "lines": group,
            "bbox": union_bbox([line["bbox"] for line in group]),
            "type": "text",
        }

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
                blocks.append(group_to_block(cur_group))
            # create new group
            cur_group = [line]

        last_span_order = line["spans"][-1]["order"]

    if cur_group:
        blocks.append(group_to_block(cur_group))

    return blocks


QUOTE_LOOSEBOX: bool = True
SUPERSCRIPT_HEIGHT_THRESHOLD: float = 0.7
LINE_DISTANCE_THRESHOLD: float = 0.1
CLASS_LIST = ["title", "caption", "figure", "table", "equation", "text"]
IMAGE_CLASS_LIST = ["figure", "table", "equation"]
LAYOUT_CLASS_TO_BLOCK_TYPE = {
    "title": "heading",
    "caption": "text",
    "figure": "image",
    "table": "table",
    "equation": "formula",
    "text": "text",
}


def partition_pdf_layout(
    doc_path: Path | str,
    render_full_page: bool = False,
    debug_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    doc = pypdfium2.PdfDocument(doc_path)
    layout_engine = RapidLayout(
        model_type="yolov8n_layout_general6",
        iou_thres=0.5,
        conf_thres=0.4,
    )
    pages = []
    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.mkdir(parents=True, exist_ok=True)

    for page_idx in range(len(doc)):
        page = doc.get_page(page_idx)
        img = page.render(scale=1.5).to_numpy()
        image_page_h, image_page_w = img.shape[:2]

        # get words from pdfium
        textpage = page.get_textpage()
        page_bbox: list[float] = page.get_bbox()
        page_width = math.ceil(abs(page_bbox[2] - page_bbox[0]))
        page_height = math.ceil(abs(page_bbox[1] - page_bbox[3]))
        page_rotation = 0
        try:
            page_rotation = page.get_rotation()
        except:  # noqa: E722
            pass

        chars = deduplicate_chars(
            get_chars(textpage, page_bbox, page_rotation, QUOTE_LOOSEBOX)
        )
        spans = get_spans(
            chars,
            superscript_height_threshold=SUPERSCRIPT_HEIGHT_THRESHOLD,
            line_distance_threshold=LINE_DISTANCE_THRESHOLD,
            split_on_space=True,
        )
        lines = get_lines(spans)
        assign_scripts(
            lines,
            height_threshold=SUPERSCRIPT_HEIGHT_THRESHOLD,
            line_distance_threshold=LINE_DISTANCE_THRESHOLD,
        )

        # scale line bbox with page size
        for line in lines:
            line["bbox"] = scale_bbox(line["bbox"], page_width, page_height)

        # add order to spans for later sorting of semantic blocks
        for idx, span in enumerate(spans):
            span["order"] = idx

        if render_full_page:
            # render the whole page with layout-preserving text
            rendered_content = spans_to_layout_text(spans)
            pages.append(
                {
                    "blocks": [
                        {
                            "text": "```{}\n{}\n```".format(
                                "page",
                                fix_unicode_encoding(rendered_content),
                            ),
                            "bbox": union_bbox([line["bbox"] for line in lines]),
                            "type": "text",
                        }
                    ],
                    "page": page_idx,
                }
            )
        else:
            detected_boxes, scores, class_names, elapsed_time = layout_engine(img)
            scaled_boxes = [
                scale_bbox(box, image_page_w, image_page_h) for box in detected_boxes
            ]
            blocks_by_class = {class_name: [] for class_name in CLASS_LIST}

            for box, class_name in zip(scaled_boxes, class_names):
                class_name = class_name.lower()
                if class_name not in CLASS_LIST:
                    continue
                blocks_by_class[class_name].append(
                    {
                        "bbox": box,
                        "type": class_name,
                    }
                )

            # assign lines to blocks
            blocks_by_class, left_over_line_indices = assign_lines_to_blocks(
                lines,
                blocks_by_class,
            )

            print(
                f"Page {page_idx}: {len(detected_boxes)} "
                f"boxes detected in {elapsed_time:.2f} seconds"
            )

            # handle left-over lines
            left_over_blocks = group_lines_by_span_order(
                [lines[idx] for idx in left_over_line_indices]
            )

            # sort blocks by span order
            sorted_blocks = sorted(
                [block for blocks in blocks_by_class.values() for block in blocks]
                + left_over_blocks,
                key=get_block_order,
            )

            page_blocks = []
            for block in sorted_blocks:
                class_name = block["type"]
                if (
                    class_name not in IMAGE_CLASS_LIST
                    and len(block.get("lines", [])) == 0
                ):
                    continue

                block_lines = block.get("lines", [])
                block_spans = [span for line in block_lines for span in line["spans"]]

                if class_name in ["table", "equation", "figure"]:
                    is_table = class_name == "table"
                    block_text = spans_to_layout_text(
                        block_spans,
                        h_multiplier=1.0,
                        w_multiplier=0.80,
                        strip_spaces=False,
                        scale_on_conflict=is_table,
                    )
                    block_text = "```{}\n{}\n```".format(
                        class_name, fix_unicode_encoding(block_text).rstrip()
                    )
                else:
                    block_text = "".join([span["text"] for span in block_spans])
                    block_text = (
                        fix_unicode_encoding(block_text)
                        .replace("\n", " ")
                        .replace("\r", "")
                        .strip()
                    )

                if block_text or class_name in IMAGE_CLASS_LIST:
                    page_blocks.append(
                        {
                            "text": block_text,
                            "bbox": block["bbox"],
                            "type": LAYOUT_CLASS_TO_BLOCK_TYPE[class_name],
                            "lines": [
                                {
                                    "bbox": scale_bbox(
                                        span["bbox"], page_width, page_height
                                    ),
                                    "text": span["text"],
                                }
                                for line in block["lines"]
                                for span in line["spans"]
                            ],
                        }
                    )

            pages.append(
                {
                    "blocks": page_blocks,
                    "page": page_idx,
                }
            )

            if debug_path is not None:
                ploted_img = VisLayout.draw_detections(
                    img, detected_boxes, scores, class_names
                )
                if ploted_img is not None:
                    cv2.imwrite(str(debug_path / f"page_{page_idx}.png"), ploted_img)

    doc.close()
    return pages
