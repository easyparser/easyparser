from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_docx
from easyparser.parser.pandoc_engine import PandocEngine
from easyparser.split.agentic_chunker import AgenticChunker
from easyparser.split.split import FlattenToMarkdown

docx_path = str(Path(__file__).parent.parent / "assets" / "with_image.docx")


def test_agentic_chunker():
    # parse the file
    root = mime_docx.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    chunk = chunks[0]

    # flatten to markdown
    flattened = FlattenToMarkdown.run(chunk, max_size=512)
    output = AgenticChunker.run(flattened[0].clone(next=None))
    assert isinstance(output[0], Chunk)


test_agentic_chunker()
