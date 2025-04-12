from pathlib import Path

from easyparser.mime import mime_md
from easyparser.parse.md import Markdown

md_path1 = str(Path(__file__).parent.parent / "assets" / "lz.md")


def test_parse_markdown():
    root = mime_md.as_root_chunk(md_path1)
    chunks = Markdown.run(root)
    chunk = list(chunks.iter_groups())[0][0]
    chunk.print_graph()
    assert len(chunks) > 0
