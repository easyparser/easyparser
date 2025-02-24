from dataclasses import dataclass, field
from typing import Any, Optional


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

    # def save(self, path: str):
    #     with open(path, "w") as f:
    #         f.write(self.text)

    # export
    # methods to compose snippets to a larger file
