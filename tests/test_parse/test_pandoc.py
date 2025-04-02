from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_docx
from easyparser.parse.pandoc_engine import PandocEngine

docx_path = str(Path(__file__).parent.parent / "assets" / "lz.md")


def test_pandoc():
    root = mime_docx.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    chunk = list(chunks.iter_groups())[0][0]
    assert isinstance(chunk, Chunk)


if __name__ == "__main__":
    test_pandoc()
