from pathlib import Path

from easyparser.base import Chunk
from easyparser.controller import Controller
from easyparser.parser.dict_list import JsonParser, TomlParser, YamlParser
from easyparser.split.text import ChunkJsonString

json_path = str(Path(__file__).parent.parent / "assets" / "long.json")
toml_path = str(Path(__file__).parent.parent / "assets" / "long.toml")
yaml_path = str(Path(__file__).parent.parent / "assets" / "long.yaml")

ctrl = Controller()


def test_chunk_json():
    root = ctrl.as_root_chunk(json_path)
    root = JsonParser.run(root)[0]
    chunks = ChunkJsonString.run(root.child, chunk_size=2000, chunk_overlap=100)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_chunk_toml():
    root = ctrl.as_root_chunk(toml_path)
    root = TomlParser.run(root)[0]
    chunks = ChunkJsonString.run(root.child, chunk_size=200, chunk_overlap=10)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_chunk_yaml():
    root = ctrl.as_root_chunk(yaml_path)
    root = YamlParser.run(root)[0]
    chunks = ChunkJsonString.run(root.child, chunk_size=500, chunk_overlap=0)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
