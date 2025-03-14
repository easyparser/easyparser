from easyparser.base import BaseStore, Chunk


class MemoryStore(BaseStore):
    """Memory-backed chunk store

    Suitable to expose chunk store interface to a group of chunks.
    """

    def __init__(self, chunks: dict | None = None):
        self._chunks = chunks or {}

    def __contains__(self, id):
        return id in self._chunks

    def get(self, id):
        return self._chunks[id]

    def fetch_content(self, chunk: Chunk):
        return self._chunks[chunk.id].content

    def save(self, chunk: Chunk):
        self._chunks[chunk.id] = chunk

    def delete(self, chunk: Chunk):
        """Delete the chunk from the directory

        @TODO: delete the relations as well
        """
        del self._chunks[chunk.id]
