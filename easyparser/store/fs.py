import json
from pathlib import Path

from easyparser.base import BaseStore, Chunk


class FileStore(BaseStore):
    """File-backed chunk store"""

    def __init__(self, path: str | Path):
        self._path: Path = Path(path).resolve()

    def __contains__(self, id):
        return (self._path / f"{id}.json").exists()

    def get(self, id):
        """Get a chunk by id"""
        file_path = self._path / f"{id}.json"
        with open(file_path) as f:
            data = json.load(f)
            # internal attributes
            _history = data.pop("_history", None)

        chunk = Chunk(**data)

        # fill the history
        if _history:
            chunk._history = _history
        chunk.store = self
        return chunk

    def fetch_content(self, chunk: Chunk):
        content_path = self._path / f"{chunk.id}.content"
        if not content_path.exists():
            return None
        with open(content_path, "rb") as f:
            return f.read()

    def save(self, chunk: Chunk):
        file_path = self._path / f"{chunk.id}.json"

        # dump the lightweight part
        with open(file_path, "w") as f:
            json.dump(chunk.as_dict(), f)

        # dump the content if any
        if chunk._content is not None:
            content_path = self._path / f"{chunk.id}.content"
            with open(content_path, "wb") as f:
                f.write(chunk._content)

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
