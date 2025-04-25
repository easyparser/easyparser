from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_epub
from easyparser.parse.pandoc_engine import PandocEngine

file_path = str(Path(__file__).parent.parent / "assets" / "long.epub")


def test_pandoc_epub():
    root = mime_epub.as_root_chunk(file_path)
    chunks = PandocEngine.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
