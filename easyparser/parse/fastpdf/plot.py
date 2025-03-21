from pathlib import Path

import cv2
from img2table.document import PDF


def plot_blocks(path: str, pages: list[dict], output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    doc = PDF(path)
    for idx, img in enumerate(doc.images):
        page = pages[idx]
        page_image_w, page_image_h = img.shape[:2]

        for block in page["blocks"]:
            x1, y1, x2, y2 = block["bbox"]
            x1 = x1 * page_image_w
            y1 = y1 * page_image_h
            x2 = x2 * page_image_w
            y2 = y2 * page_image_h
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(
                img,
                block["type"] + ": " + block["text"][:20].replace("\n", " "),
                (int(x1), int(y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            if block["type"] == "table":
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

        cv2.imwrite(str(output_path / f"page_{idx}.png"), img)
