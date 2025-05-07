from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_mp4
from easyparser.parser.video import VideoWhisperParser

mp4_path = str(Path(__file__).parent.parent / "assets" / "jfk_30.mp4")


def test_mp4():
    root = mime_mp4.as_root_chunk(mp4_path)
    chunks = VideoWhisperParser.run(root, include_segments=True)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
