from pathlib import Path

import cv2
from img2table.document import PDF


def plot_blocks(path: str, pages: list[dict], output_path: str, pad: int = 5):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    is_image = Path(path).suffix.lower() != ".pdf"

    if not is_image:
        doc = PDF(path)
        doc_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in doc.images]
    else:
        doc_images = [cv2.imread(path)]

    for idx, img in enumerate(doc_images):
        page = pages[idx]
        page_image_h, page_image_w = img.shape[:2]

        for block in page["blocks"]:
            x1, y1, x2, y2 = block["bbox"]
            x1 = x1 * page_image_w - pad
            y1 = y1 * page_image_h - pad
            x2 = x2 * page_image_w + pad
            y2 = y2 * page_image_h + pad

            for line in block.get("lines", []):
                line_x1, line_y1, line_x2, line_y2 = line["bbox"]
                line_x1 = line_x1 * page_image_w
                line_y1 = line_y1 * page_image_h
                line_x2 = line_x2 * page_image_w
                line_y2 = line_y2 * page_image_h

                cv2.rectangle(
                    img,
                    (int(line_x1), int(line_y1)),
                    (int(line_x2), int(line_y2)),
                    (0, 255, 0),
                    2,
                )

            chunk_type = block.get("type", "other")
            if chunk_type == "text":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(
                img,
                block["type"] + ": " + block["text"][:20].replace("\n", " "),
                (int(x1), int(y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            if block["type"] == "table" and "rows" in block:
                for row in block["rows"]:
                    for cell in row:
                        x1, y1, x2, y2 = cell
                        x1 = x1 * page_image_w
                        y1 = y1 * page_image_h
                        x2 = x2 * page_image_w
                        y2 = y2 * page_image_h
                        cv2.rectangle(
                            img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 0, 0),
                            2,
                        )

        cv2.imwrite(
            str(output_path / f"page_{idx}.png"),
            img,
        )
