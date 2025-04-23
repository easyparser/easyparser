import hashlib
from pathlib import Path

from easyparser.base import Chunk, CType, Origin


def as_root_chunk(path: str) -> Chunk:
    """From a pdf file to a base chunk"""
    path = str(Path(path).resolve())
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    chunk = Chunk(
        mimetype=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ctype=CType.Para,
        origin=Origin(location=path),
        metadata={
            "file_hash": file_hash,
        },
    )
    chunk.id = f"xlsx_{hashlib.sha256(path.encode()).hexdigest()}"
    return chunk
