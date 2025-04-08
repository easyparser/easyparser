import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pypdfium2
from img2table.document import PDF
from img2table.ocr.pdf import PdfOCR
from img2table.tables.image import TableImage

MIN_CONFIDENCE = 50


def detect_tables_single_page(img):
    try:
        table_image = TableImage(img=img, min_confidence=MIN_CONFIDENCE)
        output = table_image.extract_tables(
            implicit_columns=False,
            implicit_rows=True,
            borderless_tables=True,
        )
    except:  # noqa
        output = []
    return output


def check_valid_table(table, col_thres=0.3):
    # check if every column / row has more than `thres`
    # number of cells has non-empty value
    if not table:
        return False

    table_content = list(table.content.values())
    col_count = len(table_content[0])
    row_count = len(table_content)
    col_fill_count_dict = {col: 0 for col in range(col_count)}

    for row in table_content:
        for cell_idx, cell in enumerate(row):
            col_fill_count_dict[cell_idx] += 1 if cell.value else 0

    return all(
        [col_fill_count_dict[col] / row_count > col_thres for col in range(col_count)]
    ) and (row_count > 2)


def get_images_pdfium(pdf_path: str):
    pdf = pypdfium2.PdfDocument(pdf_path)
    output_images = defaultdict(list)

    for idx in range(len(pdf)):
        page = pdf.get_page(idx)

        page_bbox: list[float] = page.get_bbox()
        page_width = math.ceil(abs(page_bbox[2] - page_bbox[0]))
        page_height = math.ceil(abs(page_bbox[1] - page_bbox[3]))

        for obj in page.get_objects(
            filter=[
                pypdfium2.raw.FPDF_PAGEOBJ_IMAGE,
                pypdfium2.raw.FPDF_PAGEOBJ_FORM,
            ],
            max_depth=1,
        ):
            try:
                x1, y2, x2, y1 = obj.get_pos()
                scaled_bbox = [
                    x1 / page_width,
                    1 - y1 / page_height,
                    x2 / page_width,
                    1 - y2 / page_height,
                ]
                x1, y2, x2, y1 = scaled_bbox
                min_x, max_x = min(x1, x2), max(x1, x2)
                min_y, max_y = min(y1, y2), max(y1, y2)

                # check if pos within range [0..1]
                if min_x < 0 or min_y < 0 or max_x > 1 or max_y > 1:
                    continue

                output_images[idx].append(
                    {
                        "type": "image",
                        "bbox": scaled_bbox,
                        "text": "",
                    }
                )
            except Exception as exc:
                print(f"pdfium Image extraction failure: {exc}")

    return output_images


def get_tables_img2table(path: str, executor: ProcessPoolExecutor | None):
    # Extract tables from document
    doc = PDF(path)
    ocr = PdfOCR()

    if executor is None:
        detected_tables = [detect_tables_single_page(img) for img in doc.images]
    else:
        futures = [
            executor.submit(detect_tables_single_page, img) for img in doc.images
        ]
        detected_tables = [f.result() for f in futures]

    tables = {idx: table_list for idx, table_list in enumerate(detected_tables)}
    tables = doc.get_table_content(
        ocr=ocr, tables=tables, min_confidence=MIN_CONFIDENCE
    )

    output_tables = {}
    for page_idx, page_tables in tables.items():
        page_image_h, page_image_w = doc.images[page_idx].shape[:2]
        output_tables[page_idx] = [
            {
                "text": table.html,
                "title": table.title if table.title else "",
                "bbox": [
                    float(table.bbox.x1 / page_image_w),
                    float(table.bbox.y1 / page_image_h),
                    float(table.bbox.x2 / page_image_w),
                    float(table.bbox.y2 / page_image_h),
                ],
                "type": "table",
                "rows": [
                    [
                        [
                            cell.bbox.x1 / page_image_w,
                            cell.bbox.y1 / page_image_h,
                            cell.bbox.x2 / page_image_w,
                            cell.bbox.y2 / page_image_h,
                        ]
                        for cell in row
                    ]
                    for row in table.content.values()
                ],
            }
            for table in page_tables
            if check_valid_table(table)
        ]
    return output_tables
