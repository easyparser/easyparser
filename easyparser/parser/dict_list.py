import json
import logging

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType
from easyparser.mime import MimeType

logger = logging.getLogger(__name__)


class JsonParser(BaseOperation):

    @classmethod
    def run(cls, chunks: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """Load JSON data from a file or string into stringified Python dictionary"""
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for chunk in chunks:
            logger.info(f"Parsing {chunk.origin.location}")
            with open(chunk.origin.location) as f:
                content = repr(json.load(f))
            ch = Chunk(
                ctype=CType.Div,
                content=content,
                mimetype=MimeType.text,
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

            _read_mode = "rb"
        except ImportError:
            import toml as tomllib

            _read_mode = "r"

        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for chunk in chunks:
            logger.info(f"Parsing {chunk.origin.location}")
            with open(chunk.origin.location, mode=_read_mode) as f:
                content = repr(tomllib.load(f))
            ch = Chunk(
                ctype=CType.Div,
                content=content,
                mimetype=MimeType.text,
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
            logger.info(f"Parsing {chunk.origin.location}")
            with open(chunk.origin.location) as f:
                content = repr(yaml.safe_load(f))
            ch = Chunk(
                ctype=CType.Div,
                content=content,
                mimetype=MimeType.text,
            )
            chunk.add_children(ch)
            output.append(chunk)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["pyyaml"]
