from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_pptx
from easyparser.parse.pptx import PptxParser

pptx_path = str(Path(__file__).parent.parent / "assets" / "normal.pptx")


def test_pptx_fast():
    root = mime_pptx.as_root_chunk(pptx_path)
    chunks = PptxParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
