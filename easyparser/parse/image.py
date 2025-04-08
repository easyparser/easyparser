from collections import defaultdict

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType, Origin


class RapidOCRImageText(BaseOperation):

    @classmethod
    def run(cls, chunk: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """OCR text in image with RapidOCR engine."""
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

            # Detect layout
            lboxes, lconfs, lclasses, _ = layout_engine(img)
            if len(lclasses) == 0 or (
                len(set(lclasses)) == 1 and lclasses[0] == "figure"
            ):
                # No text detected
                llm_mode = True
                print(f"Using llm mode: {lclasses}")
            else:
                llm_mode = False
                print(f"Using ocr mode: {lclasses}")
                if ocr_engine is None:
                    ocr_engine = RapidOCR(
                        params={
                            "Global.lang_det": "en_mobile",
                            "Global.lang_rec": "en_mobile",
                        }
                    )

            if llm_mode:
                print("TODO: integrate with LLM")
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
                        chunk_lclass.origin = mc.origin
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
                children = childs[parent_id]
                id2chunk[parent_id].text = " ".join(child.text for child in children)

            output.append(mc)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["rapidocr", "rapid-layout", "rapid_table", "opencv-python"]
