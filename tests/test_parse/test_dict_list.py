from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_json, mime_toml, mime_yaml
from easyparser.parser.dict_list import JsonParser, TomlParser, YamlParser

json_path = str(Path(__file__).parent.parent / "assets" / "long.json")
toml_path = str(Path(__file__).parent.parent / "assets" / "long.toml")
yaml_path = str(Path(__file__).parent.parent / "assets" / "long.yaml")


def test_parse_json():
    root = mime_json.as_root_chunk(json_path)
    chunks = JsonParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.content, str)


def test_parse_toml():
    root = mime_toml.as_root_chunk(toml_path)
    chunks = TomlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.content, str)


def test_parse_yaml():
    root = mime_yaml.as_root_chunk(yaml_path)
    chunks = YamlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.content, str)
