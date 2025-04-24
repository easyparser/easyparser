from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_mp3, mime_wav
from easyparser.parse.audio import AudioWhisperParser

mp3_path = str(Path(__file__).parent.parent / "assets" / "jfk_apollo_49.mp3")
wav_path = str(Path(__file__).parent.parent / "assets" / "jfk_apollo_49.wav")


def test_mp3():
    root = mime_mp3.as_root_chunk(mp3_path)
    chunks = AudioWhisperParser.run(root, include_segments=True)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_wav():
    root = mime_wav.as_root_chunk(wav_path)
    chunks = AudioWhisperParser.run(root, include_segments=True)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
