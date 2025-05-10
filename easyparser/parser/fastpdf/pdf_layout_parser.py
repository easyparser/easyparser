import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pypdfium2
from pdftext.pdf.chars import deduplicate_chars, get_chars
from pdftext.pdf.pages import get_lines, get_spans
from rapid_layout import RapidLayout, VisLayout
from rapidocr import RapidOCR

from easyparser.parser.fastpdf.util import (
    crop_img_and_export_base64,
    crop_img_and_export_bytes,
    fix_unicode_encoding,
    get_block_order,
    get_overlap_ratio,
    group_lines_by_span_order,
    is_2d_layout,
    is_valid_bbox,
    optimize_2d_rendered_text,
    scale_bbox,
    spans_to_layout_text,
    union_bbox,
)

QUOTE_LOOSEBOX: bool = True
SUPERSCRIPT_HEIGHT_THRESHOLD: float = 0.7
LINE_DISTANCE_THRESHOLD: float = 0.1
MIN_NUM_LINES_2D: int = 5
CLASS_LIST = ["title", "caption", "figure", "table", "equation", "text"]
IMAGE_CLASS_LIST = ["figure", "table", "equation"]
TEXT_CLASS_LIST = ["text", "title", "caption", "table"]
LAYOUT_CLASS_TO_BLOCK_TYPE = {
    "title": "heading",
    "caption": "text",
    "figure": "image",
    "table": "table",
    "equation": "formula",
    "text": "text",
}


class SingletonModelEngine:
    _instance: Optional["SingletonModelEngine"] = None
    _ocr_engine: Optional[RapidOCR] = None
    _layout_engine: Optional[RapidLayout] = None

    def __new__(cls) -> "SingletonModelEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def ocr_engine(self) -> RapidOCR:
        if self._ocr_engine is None:
            self._ocr_engine = RapidOCR(
                params={"Global.lang_det": "en_mobile", "Global.lang_rec": "en_mobile"}
            )
        return self._ocr_engine

    @property
    def layout_engine(self) -> RapidLayout:
        if self._layout_engine is None:
            self._layout_engine = RapidLayout(
                model_type="yolov8n_layout_general6", iou_thres=0.5, conf_thres=0.4
            )
        return self._layout_engine


def assign_lines_to_blocks(
    lines: list[dict[str, Any]],
    blocks_by_class: dict[str, list[dict[str, Any]]],
    iou_threshold: float = 0.3,
    scale_line_by_page_bbox: tuple[float, float] | None = None,
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
                line_bbox = (
                    line["bbox"]
                    if scale_line_by_page_bbox is None
                    else scale_bbox(line["bbox"], *scale_line_by_page_bbox)
                )
                if not is_valid_bbox(line_bbox):
                    continue
                overlap_ratio = get_overlap_ratio(block_bbox, line_bbox)
                if overlap_ratio > iou_threshold:
                    block["lines"].append(line)
                    assigned_line_indices.add(line_idx)

    left_over_line_indices = set(range(len(lines))) - assigned_line_indices
    return blocks_by_class, left_over_line_indices


def is_ocr_required(blocks: list[dict[str, Any]], ocr_thres=0.75) -> bool:
    # check if block type in TEXT_CLASS_LIST has at least one line
    line_counts = []
    for block in blocks:
        if block["type"] in TEXT_CLASS_LIST:
            line_counts.append(len(block["lines"]))

    num_blocks = len(line_counts)
    if num_blocks == 0:
        return True

    num_empty = len([x for x in line_counts if x == 0])
    return (num_empty / num_blocks) > ocr_thres


def do_ocr_page(
    img, page_idx: int, debug_path: Path | None = None, append_space: bool = True
):
    ocr_result = SingletonModelEngine().ocr_engine(img, use_cls=False)
    img_h, img_w = img.shape[:2]
    spans = []

    if ocr_result.boxes is None or ocr_result.txts is None:
        print(f"No OCR result on page {page_idx}")
        return spans

    for bbox, text in zip(ocr_result.boxes, ocr_result.txts):
        all_x = [p[0] for p in bbox]
        all_y = [p[1] for p in bbox]
        converted_bbox = [min(all_x), min(all_y), max(all_x), max(all_y)]
        spans.append(
            {
                "text": text + " " if append_space else text,
                "bbox": scale_bbox(converted_bbox, img_w, img_h),
                "order": len(spans),
            }
        )

    if debug_path is not None:
        ocr_result.vis(debug_path / f"ocr_page_{page_idx}.png")

    print(f"OCR elapsed time on page {page_idx}: {ocr_result.elapse:.2f} seconds")
    return spans


def get_text_pdfium(page: Any):
    textpage = page.get_textpage()
    page_bbox: list[float] = page.get_bbox()
    page_width = math.ceil(abs(page_bbox[2] - page_bbox[0]))
    page_height = math.ceil(abs(page_bbox[1] - page_bbox[3]))
    try:
        page_rotation = page.get_rotation()
    except:  # noqa: E722
        page_rotation = 0
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

    # add order to spans for later sorting of semantic blocks
    for idx, span in enumerate(spans):
        span["order"] = idx

    return lines, (page_width, page_height)


def get_text_ocr(page_img, page_idx, debug_path: Path | None = None):
    ocr_spans = do_ocr_page(page_img, page_idx, debug_path)
    return [{"bbox": span["bbox"], "spans": [span]} for span in ocr_spans]


def render_blocks(
    blocks: list[dict[str, Any]],
    page_shape: tuple[int, int],
    page_img: np.ndarray | None = None,
    optimize_2d_text: bool = True,
    is_ocr: bool = False,
    export_raw_img: bool = True,
) -> list[dict[str, Any]]:
    """Render blocks with metadata to final text."""
    page_blocks = []
    page_width, page_height = page_shape
    for block in blocks:
        class_name = block["type"]
        if class_name not in IMAGE_CLASS_LIST and len(block.get("lines", [])) == 0:
            continue

        block_lines = block.get("lines", [])
        block_spans = [span for line in block_lines for span in line["spans"]]
        is_text_2d = len(block_lines) > MIN_NUM_LINES_2D and is_2d_layout(block_spans)
        img_content = None

        if class_name in IMAGE_CLASS_LIST or (class_name == "text" and is_text_2d):
            is_table = class_name in ["table", "text"]
            is_figure = class_name == "figure"
            is_equation = class_name == "equation"

            # w_multiplier for equation is greater (more packed than others)
            w_multiplier = 0.6 if is_equation else 1.0
            h_multiplier = 0.8 if is_table else 1.2

            block_text = spans_to_layout_text(
                lines=block_lines,
                w_multiplier=w_multiplier,
                h_multiplier=h_multiplier,
                filter_invalid_spans=False,
                strip_spaces=False,
                rescale_on_conflict=is_table,
                sort_span=not is_figure,
                add_line_break_on_conflict=is_figure,
            )
            block_text = optimize_2d_rendered_text(
                block_text,
                optimize_rows=True,
                optimize_cols=optimize_2d_text or is_equation,
            )
            block_text = fix_unicode_encoding(block_text).rstrip()
            if page_img is not None:
                img_content = (
                    crop_img_and_export_bytes(page_img, block["bbox"])
                    if export_raw_img
                    else crop_img_and_export_base64(page_img, block["bbox"])
                )
            if block_text:
                block_text = f"```{class_name}\n{block_text}\n```"
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
                    "image": img_content,
                    "bbox": block["bbox"],
                    "type": LAYOUT_CLASS_TO_BLOCK_TYPE[class_name],
                    "lines": [
                        {
                            "bbox": (
                                span["bbox"]
                                if is_ocr
                                else scale_bbox(span["bbox"], page_width, page_height)
                            ),
                            "text": span["text"],
                        }
                        for line in block["lines"]
                        for span in line["spans"]
                    ],
                }
            )
    return page_blocks


def partition_pdf_layout(
    doc_path: Path | str,
    render_scale: float = 1.5,
    render_full_page: bool = False,
    optimize_2d_text: bool = False,
    debug_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    doc_path = Path(doc_path)
    is_image = doc_path.suffix.lower() != ".pdf"

    output_pages = []
    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.mkdir(parents=True, exist_ok=True)

    if not is_image:
        doc = pypdfium2.PdfDocument(doc_path)
        pdf_pages: dict[int, dict] = {}
        for page_idx in range(len(doc)):
            page_pdfium = doc.get_page(page_idx)
            page_img = page_pdfium.render(scale=render_scale).to_numpy()
            pdf_pages[page_idx] = {
                "metadata": page_pdfium,
                "image": page_img,
            }
    else:
        # load image
        page_img = cv2.imread(str(doc_path))
        pdf_pages = {
            0: {
                "metadata": None,
                "image": page_img,
            }
        }

    for page_idx, page_info in pdf_pages.items():
        page_pdfium, page_img = page_info["metadata"], page_info["image"]
        image_page_h, image_page_w = page_img.shape[:2]

        if page_pdfium:
            # get text information from pdfium
            lines, (page_width, page_height) = get_text_pdfium(page_pdfium)
        else:
            lines = []
            page_width = page_height = 1

        if render_full_page:
            # use OCR to get text if lines is empty
            is_ocr = len(lines) == 0
            if is_ocr:
                lines = get_text_ocr(page_img, page_idx, debug_path)

            if not lines:
                continue
            # render the whole page with layout-preserving text
            rendered_content = spans_to_layout_text(
                lines=lines,
                filter_invalid_spans=not is_ocr,
            )
            output_pages.append(
                {
                    "blocks": [
                        {
                            "text": "```{}\n{}\n```".format(
                                "page",
                                fix_unicode_encoding(rendered_content),
                            ),
                            "bbox": (
                                union_bbox([line["bbox"] for line in lines])
                                if is_ocr
                                else scale_bbox(
                                    union_bbox([line["bbox"] for line in lines]),
                                    page_width,
                                    page_height,
                                )
                            ),
                            "lines": [
                                {
                                    "bbox": (
                                        span["bbox"]
                                        if is_ocr
                                        else scale_bbox(
                                            span["bbox"], page_width, page_height
                                        )
                                    ),
                                    "text": span["text"],
                                }
                                for line in lines
                                for span in line["spans"]
                            ],
                            "type": "text",
                        }
                    ],
                    "page": page_idx,
                }
            )
        else:
            detected_boxes, scores, class_names, elapsed_time = (
                SingletonModelEngine().layout_engine(page_img)
            )
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

            print(
                f"Page {page_idx}: {len(detected_boxes)} "
                f"boxes detected in {elapsed_time:.2f} seconds"
            )

            if lines:
                # assign lines to blocks
                blocks_by_class, left_over_line_indices = assign_lines_to_blocks(
                    lines,
                    blocks_by_class,
                    scale_line_by_page_bbox=(page_width, page_height),
                )
                # handle left-over lines
                left_over_blocks = group_lines_by_span_order(
                    [lines[idx] for idx in left_over_line_indices],
                    page_width,
                    page_height,
                )
                # rescale left-over blocks
                for block in left_over_blocks:
                    block["bbox"] = scale_bbox(block["bbox"], page_width, page_height)
                # combine blocks
                all_blocks = [
                    block for blocks in blocks_by_class.values() for block in blocks
                ] + left_over_blocks
            else:
                all_blocks = []

            # check if OCR is required
            is_ocr = is_ocr_required(all_blocks)
            if is_ocr:
                ocr_lines = get_text_ocr(page_img, page_idx, debug_path)
                # assign lines to blocks
                blocks_by_class, left_over_line_indices = assign_lines_to_blocks(
                    ocr_lines,
                    blocks_by_class,
                    scale_line_by_page_bbox=None,
                )
                # handle left-over lines
                left_over_blocks = group_lines_by_span_order(
                    [ocr_lines[idx] for idx in left_over_line_indices],
                    page_w=1,
                    page_h=1,
                )
                # rewrite blocks
                all_blocks = [
                    block for blocks in blocks_by_class.values() for block in blocks
                ] + left_over_blocks

            # sort blocks by span order
            sorted_blocks = sorted(
                all_blocks,
                key=get_block_order,
            )
            output_pages.append(
                {
                    "blocks": render_blocks(
                        sorted_blocks,
                        page_shape=(page_width, page_height),
                        page_img=page_img,
                        optimize_2d_text=optimize_2d_text,
                        is_ocr=is_ocr,
                    ),
                    "page": page_idx,
                }
            )

            if debug_path is not None:
                ploted_img = VisLayout.draw_detections(
                    page_img, detected_boxes, scores, class_names
                )
                if ploted_img is not None:
                    cv2.imwrite(
                        str(debug_path / f"layout_page_{page_idx}.png"), ploted_img
                    )

    if not is_image:
        doc.close()
    return output_pages
