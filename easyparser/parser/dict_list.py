import json

from easyparser.base import BaseOperation, Chunk, ChunkGroup


class JsonParser(BaseOperation):

    @classmethod
    def run(cls, chunks: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """Load JSON data from a file or string into stringified Python dictionary"""
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for chunk in chunks:
            with open(chunk.origin.location) as f:
                chunk.content = repr(json.load(f))
            output.append(chunk)

        return output


class TomlParser(BaseOperation):

    @classmethod
    def run(cls, chunks: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """Load TOML data from a file or string into stringified Python dictionary"""
        try:
            import tomllib
        except ImportError:
            import toml as tomllib

        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for chunk in chunks:
            with open(chunk.origin.location) as f:
                chunk.content = repr(tomllib.load(f))
            output.append(chunk)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        try:
            import tomllib  # noqa: F401
        except Exception:
            return ["toml"]

        return []


class YamlParser(BaseOperation):

    @classmethod
    def run(cls, chunks: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """Load YAML data from a file or string into stringified Python dictionary"""
        import yaml

        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for chunk in chunks:
            with open(chunk.origin.location) as f:
                chunk.content = repr(yaml.safe_load(f))
            output.append(chunk)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["pyyaml"]
