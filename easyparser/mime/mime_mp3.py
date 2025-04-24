import hashlib
from pathlib import Path

from easyparser.base import Chunk, CType, Origin


def as_root_chunk(path: str) -> Chunk:
    """From a mp3 file to a base chunk"""
    path = str(Path(path).resolve())
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    chunk = Chunk(
        mimetype="audio/mpeg",
        ctype=CType.Para,
        origin=Origin(location=path),
        metadata={
            "file_hash": file_hash,
        },
    )
    chunk.id = f"mp3_{hashlib.sha256(path.encode()).hexdigest()}"
    return chunk
