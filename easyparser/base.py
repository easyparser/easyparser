import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal


class Origin:
    """Represent the origin of an object from another object

    !IMPORTANT: the Origin must be serializable to JSON.

    Args:
        source_id: source object id.
        location: location of the object in the source object. The exact value of
            the location is dependent on the source object. For example, if the
            source object is a folder, the location can be the path of the object;
            if the source object is a PDF file, the location can be a dictionary
            contains the page number and the position of the object.
    """

    def __init__(
        self,
        source_id: str = "",
        location: Any = None,
        metadata: dict | None = None,
    ):
        self.source_id = source_id
        self.location = location
        self.metadata = metadata


class Chunk:
    """Mandatory fields for an object represented in `easyparser`.

    !IMPORTANT: all fields, except `content`, must be serializable to JSON.

    Args:
        id: unique identifier for the object.
        mimetype: mimetype of the object, plays a crucial role in determining how an
            object is processed and rendered. The official list of mimetypes:
            https://www.iana.org/assignments/media-types/media-types.xhtml
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
        parent: "None | str | Chunk" = None,
        children: None | list = None,
        next: "None | str | Chunk" = None,
        prev: "None | str | Chunk" = None,
        origin: None | Origin = None,
        metadata: None | dict = None,
    ):
        self.id: str = uuid.uuid4().hex
        self.mimetype = mimetype
        self._content = content
        self.text = text
        self._parent = parent
        self.children = children
        self.next = next
        self.prev = prev
        self.origin = origin
        self.metadata = metadata
        self._path = None

    def __str__(self):
        text = self.text
        if len(self.text) > 80:
            text = f"{self.text[:50]}... ({len(self.text[50:].split())} more words)"
        return f"Chunk(id={self.id[:5]}..., mimetype={self.mimetype}, text={text})"

    @property
    def content(self):
        """Lazy loading of the content of the object"""
        if (
            self._content is None
            and self._path is not None
            and Path(self._path).exists()
        ):
            with open(self._path, "rb") as f:
                self._content = f.read()
        return self._content

    @content.setter
    def content(self, value):
        """Set the content of the object"""
        self._content = value

    def parent(self, pool=None) -> "Chunk":
        """Get the parent object"""
        raise NotImplementedError

    @property
    def manager(self):
        """Get the active chunk manager"""
        return get_manager()

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

    def save(self, directory):
        """Save the chunk into the directory"""
        ...

    @classmethod
    def load(cls, path) -> "Chunk": ...


class ChunkGroup:
    """An interface for a group of related chunk"""

    def __init__(self, objs=None, path=None):
        self.objs = objs or {}

    """The object manager that manages all the objects"""

    def save(self, path):
        """Save all objects to a directory"""
        ...


class BaseOperation:
    """Almost all operations on Chunk should eventually subclass from this. This class
    defines the interface so that:
        - Operations can be used as tool in agentic workflow.
        - Common interface for chunk

    When subclassing this class:
        - The subclass **must** implement the `.run` method.
        - The subclass **must** call super().__init__() if it overrides the __init__
        method.
        - The subclass **might** implement the `.as_tool` method. If not implemented,
        the method will inspect the `.run` method's signature to get the necessary
        arguments, and inspect the `.run` method's docstring to get the description.

    """

    def __init__(self, *args, **kwargs):
        self._tool_desc: dict | None = None
        self._default_params: dict = {}

    @staticmethod
    def run(*chunk: Chunk, **kwargs) -> list[Chunk] | Chunk:
        """Run the operation on the chunk"""
        raise NotImplementedError

    def __call__(self, *chunk: Chunk, **kwargs) -> list[Chunk] | Chunk:
        if self._default_params:
            kwargs.update(self._default_params)
        return self.run(*chunk, **kwargs)

    def as_tool(self) -> dict:
        """Return the necessary parameters for the operation.

        If not subclassed, this method will inspect the `.run` method's signature
        """
        if self._params is None:
            self._params = {}
        return self._params

    def default(self, **kwargs):
        """Set default parameters for the operation"""
        self._default_params.update(kwargs)


class OperationManager:
    """Map mimetype to suitable operation"""

    def __init__(self, executors: dict | None = None, refiners: dict | None = None):
        # key: mimetype, value: list of supported operations for the mimetype
        self._executors = executors or defaultdict(list)
        self._refiners = refiners or defaultdict(list)

    def __enter__(self):
        _managers.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _managers.pop()

    def get_executor(self, mimetype): ...

    def add_executor(self, mimetype, operation): ...

    def update_executor(self, mimetype, operation): ...

    def get_refiner(self, mimetype): ...

    def add_refiner(self, mimetype, operation): ...

    def update_refiner(self, mimetype, operation): ...

    def as_tools(self, *mimetype: str) -> list[dict]:
        """Export all executors and refiners as tools"""
        ...

    def executor_as_tools(self, mimetype: str) -> list[dict]:
        """Export all executors as tools"""
        ...

    def refiner_as_tools(self, mimetype: str) -> list[dict]:
        """Export all refiners as tools"""
        ...

    @classmethod
    def from_default(cls) -> "OperationManager":
        """Construct an operation manager with default executors and refiners"""
        raise NotImplementedError

    def save(self, path):
        """Save all operations to a directory"""
        ...

    @classmethod
    def load(cls, path):
        """Load all operations from a directory"""
        ...


_managers = deque()


def get_manager():
    return _managers[-1]
