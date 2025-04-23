import json
from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_json
from easyparser.split.text import ChunkJsonString

json_path = str(Path(__file__).parent.parent / "assets" / "long.json")


def test_json():
    root = mime_json.as_root_chunk(json_path)
    with open(root.origin.location) as f:
        root.content = str(json.load(f))
    chunks = ChunkJsonString.run(root, chunk_size=2000, chunk_overlap=100)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
