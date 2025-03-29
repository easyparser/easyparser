from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_docx
from easyparser.parse.pandoc_engine import PandocEngine

docx_path = str(Path(__file__).parent.parent / "assets" / "with_image.docx")


def test_pandoc():
    root = mime_docx.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    # for idx, chunk in enumerate(chunks):
    #     if "png" in str(chunk).lower():
    #         import pdb; pdb.set_trace()
    #         print(f"Chunk {idx} contains image.")
    assert len(chunks) > 0
    assert root.id in chunks.groups
    assert isinstance(chunks[0], Chunk)


if __name__ == "__main__":
    test_pandoc()
