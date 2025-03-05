import uuid
from typing import Any, Literal


class Origin:
    """Represent the origin of an object from another object

    Args:
        source_id: source object id.
        location: location of the object in the source object.
    """

    def __init__(
        self,
        source_id: str,
        location: None | dict = None,
    ):
        self.source_id = source_id
        self.location = location


class Chunk:
    """Mandatory fields for an object represented in `easyparser`.

    Args:
        id: unique identifier for the object.
        mimetype: mimetype of the object, plays a crucial role in determining how an
            object is processed and rendered.
        content: content of the object, can be anything (text or bytes), that can be
            understood from the mimetype.
        text: text representation of the object.
        parent: parent object id. Defaults to None.
        children: list of children object ids. Defaults to None.
        next: next object id. Defaults to None.
        prev: previous object id. Defaults to None.
        origin: the location of this object in relative to the parent.
        metadata: metadata of the object, a free-style dictionary.
    """

    def __init__(
        self,
        mimetype: str,
        content: Any = None,
        text: str = "",
        parent: None | str | "Chunk" = None,
        children: None | list = None,
        next: None | str | "Chunk" = None,
        prev: None | str | "Chunk" = None,
        origin: None | Origin = None,
        metadata: None | dict = None,
    ):
        self.id: str = uuid.uuid4().hex
        self.mimetype = mimetype
        self.content = content
        self.text = text
        self._parent = parent
        self.children = children
        self.next = next
        self.prev = prev
        self.origin = origin
        self.metadata = metadata

    def parent(self, pool=None) -> "Chunk":
        """Get the parent object"""
        raise NotImplementedError

    def parent_id(self) -> str | None:
        if isinstance(self._parent, str):
            return self._parent
        if isinstance(self._parent, Chunk):
            return self._parent.id

    def render(
        self,
        manager=None,
        format: Literal["text", "markdown", "html"] = "text",
        executor=None,
    ):
        """Select the executor type to render the object

        Args:
            manager: object manager to get the related objects
            return_type: type of the return value. Defaults to "text".
            executor: executor to render the object. Defaults to None.
        """
        raise NotImplementedError


class ChunkManager:
    def __init__(self, objs=None):
        self.objs = objs or {}

    """The object manager that manages all the objects"""

    def save(self, path):
        """Save all objects to a directory"""
        ...

    @classmethod
    def load(cls, path):
        """Load all objects from a directory"""
        ...
