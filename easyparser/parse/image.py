import logging
from collections import defaultdict

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType, Origin

logger = logging.getLogger(__name__)


class RapidOCRImageText(BaseOperation):

    @classmethod
    def run(
        cls,
        chunk: Chunk | ChunkGroup,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
        **kwargs,
    ) -> ChunkGroup:
        """Use RapidOCR do OCR and use Gemini API to transcribe table and figure.

        Args:
            gemini_api_key: if None, it will try to read from environment
                variable `GEMINI_API_KEY`.
            gemini_model: the supported Gemini model names are:
                "gemini-2.0-flash", "gemini-2.5-pro-exp-03-25". If None,
                use "gemini-2.0-flash" as default.
        """
        import os

        import cv2
        from rapid_layout import RapidLayout
        from rapid_layout.utils.post_prepross import compute_iou
        from rapidocr import RapidOCR

        client = None
        gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        layout_engine = RapidLayout(model_type="doclayout_docstructbench")
        ocr_engine = None

        # Resolve chunk
        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])

        output = ChunkGroup()
        for mc in chunk:
            img = cv2.imread(mc.origin.location)

            # Detect layout
            lboxes, lconfs, lclasses, _ = layout_engine(img)
            if len(lclasses) == 0 or (
                len(set(lclasses)) == 1 and lclasses[0] == "figure"
            ):
                # No text detected
                vlm_mode = True
                print(f"Using llm mode: {lclasses}")
            else:
                vlm_mode = False
                print(f"Using ocr mode: {lclasses}")
                if ocr_engine is None:
                    ocr_engine = RapidOCR(
                        params={
                            "Global.lang_det": "en_mobile",
                            "Global.lang_rec": "en_mobile",
                        }
                    )

            if vlm_mode and client is None:
                if gemini_api_key is None:
                    raise ValueError(
                        "Please supply `gemini_api_key` or set `GEMINI_API_KEY`"
                    )
                from google import genai

                client = genai.Client(api_key=gemini_api_key)

            if vlm_mode and client is not None:
                from PIL import Image

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                response = client.models.generate_content(
                    model=gemini_model or "gemini-2.0-flash",
                    contents=["Describe this image, in markdown format", pil_img],
                )
                mc.text = response.text
                output.append(mc)
                continue

            ocr_result = ocr_engine(mc.origin.location)
            if not ocr_result.txts:
                output.append(mc)
                continue

            # Inefficient brute force check
            ocr_boxes = ocr_result.boxes[:, (0, 2), :].reshape(-1, 4)
            ocr_box_classes = {}
            childs = defaultdict(list)
            id2chunk = {mc.id: mc}
            for _i, ocr_box in enumerate(ocr_boxes):
                child = Chunk(
                    mimetype="text/plain",
                    ctype=CType.Inline,
                    content=ocr_result.txts[_i],
                    text=ocr_result.txts[_i],
                    origin=Origin(source_id=mc.id, location=ocr_box.tolist()),
                )
                id2chunk[child.id] = child
                iou = compute_iou(ocr_box, lboxes)
                largest_iou = iou.max()
                if largest_iou > 0:
                    idx_lclass = iou.argmax()
                    if idx_lclass not in ocr_box_classes:
                        chunk_lclass = Chunk(mimetype="text/plain")
                        if lclasses[idx_lclass] == "table":
                            chunk_lclass.ctype = CType.Table
                        elif lclasses[idx_lclass] == "figure":
                            chunk_lclass.ctype = CType.Figure
                        elif lclasses[idx_lclass] == "title":
                            chunk_lclass.ctype = CType.Header
                        else:
                            chunk_lclass.ctype = CType.Para
                        chunk_lclass.origin = Origin(
                            source_id=mc.id, location=lboxes[idx_lclass].tolist()
                        )
                        chunk_lclass.parent = mc
                        childs[mc.id].append(chunk_lclass)
                        id2chunk[chunk_lclass.id] = chunk_lclass
                        ocr_box_classes[idx_lclass] = chunk_lclass

                    child.parent = ocr_box_classes[idx_lclass]
                    childs[child.parent_id].append(child)
                else:
                    child.parent = mc
                    childs[mc.id].append(child)

            for parent_id, children in childs.items():
                for _i, child in enumerate(children[1:], start=1):
                    child.prev = children[_i - 1]
                    children[_i - 1].next = child
                id2chunk[parent_id].child = children[0]

            for parent_id in reversed(childs.keys()):
                if id2chunk[parent_id].ctype == CType.Table:
                    if client is None and gemini_api_key is not None:
                        from google import genai

                        client = genai.Client(api_key=gemini_api_key)
                    if client is not None:
                        from PIL import Image

                        x1, y1, x2, y2 = id2chunk[parent_id].origin.location
                        pil_img = Image.fromarray(
                            img[int(y1) : int(y2), int(x1) : int(x2)]
                        )
                        response = client.models.generate_content(
                            model=gemini_model or "gemini-2.0-flash",
                            contents=["Extract the table in markdown format", pil_img],
                        )
                        id2chunk[parent_id].text = response.text
                        continue
                elif id2chunk[parent_id].ctype == CType.Figure:
                    if client is None and gemini_api_key is not None:
                        from google import genai

                        client = genai.Client(api_key=gemini_api_key)
                    if client is not None:
                        from PIL import Image

                        x1, y1, x2, y2 = id2chunk[parent_id].origin.location
                        pil_img = Image.fromarray(
                            img[int(y1) : int(y2), int(x1) : int(x2)]
                        )
                        response = client.models.generate_content(
                            model=gemini_model or "gemini-2.0-flash",
                            contents=["Describe the image in markdown format", pil_img],
                        )
                        id2chunk[parent_id].text = response.text
                        continue
                children = childs[parent_id]
                id2chunk[parent_id].text = " ".join(child.text for child in children)

            output.append(mc)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return [
            "rapidocr",
            "rapid-layout",
            "rapid_table",
            "opencv-python",
            "google-genai",
            "pillow",
        ]
