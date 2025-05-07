from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_docx
from easyparser.parser.pandoc_engine import PandocEngine

docx_path = str(Path(__file__).parent.parent / "assets" / "with_image.docx")


def test_pandoc_docx():
    root = mime_docx.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
