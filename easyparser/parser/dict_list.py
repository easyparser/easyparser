import json

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType


class JsonParser(BaseOperation):

    @classmethod
    def run(cls, chunks: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """Load JSON data from a file or string into stringified Python dictionary"""
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for chunk in chunks:
            with open(chunk.origin.location) as f:
                content = repr(json.load(f))
            ch = Chunk(
                ctype=CType.Div,
                content=content,
                mimetype="text/plain",
            )
            chunk.add_children(ch)
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
                content = repr(tomllib.load(f))
            ch = Chunk(
                ctype=CType.Div,
                content=content,
                mimetype="text/plain",
            )
            chunk.add_children(ch)
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
                content = repr(yaml.safe_load(f))
            ch = Chunk(
                ctype=CType.Div,
                content=content,
                mimetype="text/plain",
            )
            chunk.add_children(ch)
            output.append(chunk)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["pyyaml"]
