from pathlib import Path

from easyparser.base import Chunk
from easyparser.parser.directory import DirectoryParser
from easyparser.router import FileCoordinator

path = str(Path(__file__).parent.parent)
_file = FileCoordinator()


def test_directory():
    root = _file.as_root_chunk(path)
    chunks = DirectoryParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
