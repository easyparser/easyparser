from collections import defaultdict
from pathlib import Path

import cv2

from easyparser.base import Chunk, ChunkGroup


def plot_pdf(pdf_path: str, chunks: ChunkGroup, output_path: str):
    from img2table.document import PDF

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    doc = PDF(pdf_path)

    page_to_chunks = defaultdict(list)

    for chunk in chunks:
        chunk_page = chunk.origin.location.get("page", -1) - 1
        page_to_chunks[chunk_page].append(chunk)

    for idx, img in enumerate(doc.images):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        page_chunks: list[Chunk] = page_to_chunks[idx]
        page_image_h, page_image_w = img.shape[:2]

        for chunk_id, chunk in enumerate(page_chunks):
            chunk_type = chunk.ctype
            x1, y1, x2, y2 = chunk.origin.location["bbox"]
            x1 = x1 * page_image_w
            y1 = y1 * page_image_h
            x2 = x2 * page_image_w
            y2 = y2 * page_image_h

            if chunk_type in ["text", "para", "list"]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(
                img,
                f"({chunk_id}) {chunk_type}",
                (int(x1), int(y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

        cv2.imwrite(str(output_path / f"page_{idx}.png"), img)


def plot_img(img_path: str, chunks: ChunkGroup, output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    img = cv2.imread(img_path)

    page_image_h, page_image_w = img.shape[:2]

    for chunk_id, chunk in enumerate(chunks):
        chunk_type = chunk.ctype
        x1, y1, x2, y2 = chunk.origin.location["bbox"]
        x1 = x1 * page_image_w
        y1 = y1 * page_image_h
        x2 = x2 * page_image_w
        y2 = y2 * page_image_h

        if chunk_type in ["text", "para", "list"]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        cv2.putText(
            img,
            f"({chunk_id}) {chunk_type}",
            (int(x1), int(y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    cv2.imwrite(str(output_path / "page_0.png"), img)


if __name__ == "__main__":
    import sys

    from easyparser.controller import Controller
    from easyparser.parser import DoclingPDF, FastPDF, UnstructuredPDF

    ctrl = Controller()
    pdf_path = sys.argv[1]
    root = ctrl.as_root_chunk(pdf_path)

    methods = [FastPDF, UnstructuredPDF, DoclingPDF]
    for method in methods:
        chunks = method.run(root)
        plot_pdf(pdf_path, chunks, f"debug_{method.__name__.lower()}")
