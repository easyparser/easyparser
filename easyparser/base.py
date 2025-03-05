import uuid
from typing import Any


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


class Obj:
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
        parent: None | str = None,
        children: None | list = None,
        next: None | str = None,
        prev: None | str = None,
        origin: None | Origin = None,
        metadata: None | dict = None,
    ):
        self.id: str = uuid.uuid4().hex
        self.mimetype = mimetype
        self.content = content
        self.text = text
        self.parent = parent
        self.children = children
        self.next = next
        self.prev = prev
        self.origin = origin
        self.metadata = metadata
