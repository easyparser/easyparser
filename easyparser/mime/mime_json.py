import hashlib
from pathlib import Path

from easyparser.base import Chunk, Origin


def as_root_chunk(path: str) -> Chunk:
    """From a json file to a base chunk"""
    path = str(Path(path).resolve())
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    chunk = Chunk(
        mimetype="application/json",
        origin=Origin(location=path),
        metadata={
            "file_hash": file_hash,
        },
    )
    chunk.id = f"json_{hashlib.sha256(path.encode()).hexdigest()}"
    return chunk
