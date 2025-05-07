from pathlib import Path

from easyparser.base import Chunk
from easyparser.controller import Controller
from easyparser.parser.text import TextParser

file_path = str(Path(__file__).parent.parent / "assets" / "long.txt")
ctrl = Controller()


def test_text():
    root = ctrl.as_root_chunk(file_path)
    chunks = TextParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
