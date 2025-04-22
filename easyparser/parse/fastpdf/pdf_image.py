import math
from collections import defaultdict

import pypdfium2


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

    pdf.close()
    return output_images
