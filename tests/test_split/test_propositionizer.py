from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_docx
from easyparser.parse.pandoc_engine import PandocEngine
from easyparser.split.propositionizer import Propositionizer

docx_path = str(Path(__file__).parent.parent / "assets" / "with_image.docx")


def test_propositionizer():
    # parse the file
    root = mime_docx.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    chunk = chunks[0]

    # run propositionizer
    temp = chunk.child.next.next.next.next.next.next.next.clone(next=None)
    output = Propositionizer.run(temp)

    assert isinstance(output[0], Chunk)
