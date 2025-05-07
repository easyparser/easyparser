from pathlib import Path

from easyparser.base import Chunk
from easyparser.controller import Controller
from easyparser.parser.pandoc_engine import PandocEngine
from easyparser.split.split import FlattenToMarkdown

docx_path = str(Path(__file__).parent.parent / "assets" / "with_image.docx")

ctrl = Controller()


def test_flatten():
    root = ctrl.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    flattened = FlattenToMarkdown.run(chunks[0], max_size=512)
    assert isinstance(flattened[0], Chunk)
