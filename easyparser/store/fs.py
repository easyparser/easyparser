import builtins
import json
from pathlib import Path

from easyparser.base import BaseStore, Chunk


class FileStore(BaseStore):
    """File-backed chunk store"""

    def __init__(self, path: str | Path):
        self._path: Path = Path(path).resolve()
        self._path.mkdir(parents=True, exist_ok=True)

    def __contains__(self, id):
        return (self._path / f"{id}.json").exists()

    def get(self, id):
        """Get a chunk by id"""
        file_path = self._path / f"{id}.json"
        with open(file_path) as f:
            data = json.load(f)
            # internal attributes
            _id = data.pop("id")
            _history = data.pop("_history", None)

        chunk = Chunk(**data)

        # fill the history
        if _history:
            chunk._history = _history
        chunk.store = self
        chunk.id = _id
        return chunk

    def fetch_content(self, chunk: Chunk):
        content_path = self._path / f"{chunk.id}.content"
        if not content_path.exists():
            return None
        with open(content_path, "rb") as f:
            return f.read()

    def save(self, chunk: Chunk):
        file_path = self._path / f"{chunk.id}.json"
        chunk_dict = chunk.as_dict()
        content = None
        if isinstance(chunk_dict["content"], builtins.bytes):
            content = chunk_dict.pop("content")

        # dump the lightweight part
        with open(file_path, "w") as f:
            json.dump(chunk.as_dict(), f)

        # dump the content if any
        if content is not None:
            content_path = self._path / f"{chunk.id}.content"
            with open(content_path, "wb") as f:
                f.write(content)

    def delete(self, chunk: Chunk):
        """Delete the chunk from the directory

        @TODO: delete the relations as well
        """
        file_path = self._path / f"{chunk.id}.json"
        if file_path.exists():
            file_path.unlink()

        content_path = self._path / f"{chunk.id}.content"
        if content_path.exists():
            content_path.unlink()
