import logging
from collections import defaultdict

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType, Origin
from easyparser.models import completion

logger = logging.getLogger(__name__)


class RapidOCRImageText(BaseOperation):

    @classmethod
    def run(
        cls,
        chunk: Chunk | ChunkGroup,
        caption: str | bool = False,
        **kwargs,
    ) -> ChunkGroup:
        """Use RapidOCR do OCR and use VLM to transcribe table and figure.

        Args:
            caption: whether to use VLM to transcribe table and figure. If True,
                use the default VLM, if a string is provided, use the LLM with that
                alias, if False, disable captioning. Defaults to False.
        """
        import cv2
        from rapid_layout import RapidLayout
        from rapid_layout.utils.post_prepross import compute_iou
        from rapidocr import RapidOCR

        layout_engine = RapidLayout(model_type="doclayout_docstructbench")
        ocr_engine = None

        # Resolve chunk
        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])

        output = ChunkGroup()
        for mc in chunk:
            img = cv2.imread(mc.origin.location)
            logger.info(f"Parsing {mc.origin.location}")

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

            if vlm_mode and caption:
                from PIL import Image

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                mc.text = completion(
                    "Describe this image, in markdown format",
                    attachments=[pil_img],
                    model=caption if isinstance(caption, str) else None,
                )
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
                    if caption:
                        from PIL import Image

                        x1, y1, x2, y2 = id2chunk[parent_id].origin.location
                        pil_img = Image.fromarray(
                            img[int(y1) : int(y2), int(x1) : int(x2)]
                        )
                        id2chunk[parent_id].text = completion(
                            "Extract the table in markdown format",
                            attachments=[pil_img],
                            model=caption if isinstance(caption, str) else None,
                        )
                        continue
                elif id2chunk[parent_id].ctype == CType.Figure:
                    if caption:
                        from PIL import Image

                        x1, y1, x2, y2 = id2chunk[parent_id].origin.location
                        pil_img = Image.fromarray(
                            img[int(y1) : int(y2), int(x1) : int(x2)]
                        )
                        id2chunk[parent_id].text = completion(
                            "Describe the image in markdown format",
                            attachments=[pil_img],
                            model=caption if isinstance(caption, str) else None,
                        )
                        continue

            output.append(mc)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return [
            "rapidocr",
            "rapid-layout",
            "rapid_table",
            "opencv-python",
            "pillow",
        ]
