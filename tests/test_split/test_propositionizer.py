from pathlib import Path

from easyparser.base import Chunk
from easyparser.controller import Controller
from easyparser.parser.pandoc_engine import PandocEngine
from easyparser.split.propositionizer import Propositionizer

docx_path = str(Path(__file__).parent.parent / "assets" / "with_image.docx")

ctrl = Controller()


def test_propositionizer():
    # parse the file
    root = ctrl.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    chunk = chunks[0]

    # run propositionizer
    temp = chunk.child.next.next.next.next.next.next.next.clone()
    output = Propositionizer.run(temp)

    assert isinstance(output[0], Chunk)
