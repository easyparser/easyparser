from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class Origin:
    source: str
    location: Any
    getter: str
    file_type: str


@dataclass
class Snippet:
    id: str
    text: str
    content: Any
    dtype: str
    origin: Optional[Origin] = None
    metadata: dict = field(default_factory=dict)
