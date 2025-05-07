from pathlib import Path

from easyparser.base import Chunk
from easyparser.parse.pptx import PptxParser
from easyparser.router import FileCoordinator

pptx_path = str(Path(__file__).parent.parent / "assets" / "normal.pptx")
pptx_short = str(Path(__file__).parent.parent / "assets" / "short_image.pptx")
_file = FileCoordinator()


def test_pptx_fast():
    root = _file.as_root_chunk(pptx_path)
    chunks = PptxParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_pptx_image():
    root = _file.as_root_chunk(pptx_short)
    chunks = PptxParser.run(root, caption=True)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
