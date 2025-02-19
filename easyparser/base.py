from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class Origin:
    source: str
    location: Any
    getter: str


@dataclass
class Document:
    id: str
    text: str
    content: Any
    dtype: str
    origin: Optional[Origin]
    metadata: dict
