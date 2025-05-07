from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_md
from easyparser.parser.md import Markdown

md_path1 = str(Path(__file__).parent.parent / "assets" / "lz.md")


def test_parse_markdown():
    root = mime_md.as_root_chunk(md_path1)
    chunks = Markdown.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
