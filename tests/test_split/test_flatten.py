from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_docx
from easyparser.parser.pandoc_engine import PandocEngine
from easyparser.split.split import FlattenToMarkdown

docx_path = str(Path(__file__).parent.parent / "assets" / "with_image.docx")


def test_flatten():
    root = mime_docx.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    flattened = FlattenToMarkdown.run(chunks[0], max_size=512)
    assert isinstance(flattened[0], Chunk)
