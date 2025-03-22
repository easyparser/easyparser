from concurrent.futures import ProcessPoolExecutor

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


def img2table_get_tables(path: str, executor: ProcessPoolExecutor | None):
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
