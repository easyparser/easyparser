import builtins
import inspect
import re
import uuid
from collections import defaultdict, deque
from typing import Any, Literal, get_type_hints


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

    def as_dict(self):
        return {
            "source_id": self.source_id,
            "location": self.location,
            "metadata": self.metadata,
        }


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
        self._children = children
        self._next = next
        self._prev = prev
        self.origin = origin
        self.metadata = metadata

        self._store: "BaseStore | None" = None

    def __str__(self):
        text = self.text
        if len(self.text) > 80:
            text = f"{self.text[:50]}... ({len(self.text[50:].split())} more words)"
        return f"Chunk(id={self.id[:5]}..., mimetype={self.mimetype}, text={text})"

    @property
    def content(self):
        """Lazy loading of the content of the object"""
        if self._content is None and self._store is not None:
            self._content = self._store.fetch_content(self)
        return self._content

    @content.setter
    def content(self, value):
        """Set the content of the object"""
        self._content = value

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, value: "BaseStore"):
        self._store = value

    @property
    def parent(self) -> "Chunk | None":
        """Get the parent object"""
        if isinstance(self._parent, Chunk):
            return self._parent
        if isinstance(self._parent, str):
            if not self._store:
                raise ValueError("Must provide `store` to load the parent")
            self._parent = self._store.get(self._parent)
            return self._parent

    @property
    def parent_id(self) -> str | None:
        if isinstance(self._parent, str):
            return self._parent
        if isinstance(self._parent, Chunk):
            return self._parent.id

    @property
    def next(self) -> "Chunk | None":
        """Get the next object"""
        if isinstance(self._next, Chunk):
            return self._next
        if isinstance(self._next, str):
            if not self._store:
                raise ValueError("Must provide `store` to load the next")
            self._next = self._store.get(self._next)
            return self._next
        if self._next is not None:
            raise ValueError("`.next` must be a Chunk or a id of a chunk")

    @next.setter
    def next(self, value):
        if isinstance(value, (Chunk, str)):
            self._next = value
        else:
            raise ValueError("`.next` must be a Chunk or a id of a chunk")

    @property
    def next_id(self) -> str | None:
        if isinstance(self._next, str):
            return self._next
        if isinstance(self._next, Chunk):
            return self._next.id

    @property
    def prev(self) -> "Chunk | None":
        """Get the previous object"""
        if isinstance(self._prev, Chunk):
            return self._prev
        if isinstance(self._prev, str):
            if not self._store:
                raise ValueError("Must provide `store` to load the prev")
            self._prev = self._store.get(self._prev)
            return self._prev
        if self._prev is not None:
            raise ValueError("`.prev` must be a Chunk or a id of a chunk")

    @prev.setter
    def prev(self, value):
        if isinstance(value, (Chunk, str)):
            self._prev = value
        else:
            raise ValueError("`.prev` must be a Chunk or a id of a chunk")

    @property
    def prev_id(self) -> str | None:
        if isinstance(self._prev, str):
            return self._prev
        if isinstance(self._prev, Chunk):
            return self._prev.id

    @property
    def children_ids(self) -> list[str]:
        if self._children is None:
            return []
        ids = []
        for child in self._children:
            if isinstance(child, str):
                ids.append(child)
            if isinstance(child, Chunk):
                ids.append(child.id)
        return ids

    @property
    def children(self) -> list["Chunk"]:
        if self._children is None:
            return []
        for idx in range(len(self._children)):
            child = self._children[idx]
            if isinstance(child, str):
                if not self._store:
                    raise ValueError("Must provide `store` to load the children")
                child = self._store.get(child)
                self._children[idx] = child
        return self._children

    def add_child(self, child: "Chunk | str"):
        if self._children is None:
            self._children = []
        self._children.append(child)

    def render(self, format: Literal["plain", "markdown", "html"] = "plain") -> str:
        """Select the executor type to render the object

        Args:
            format: the format of the output. Defaults to "text".
        """
        current = self.text
        if not self.children:
            return current
        for child in self.children:
            current += "\n\n" + child.render(format) + "\n\n"
        return current

    def as_dict(self):
        return {
            "id": self.id,
            "mimetype": self.mimetype,
            "text": self.text,
            "parent": self.parent_id,
            "children": self.children_ids,
            "next": self.next_id,
            "prev": self.prev_id,
            "origin": self.origin.as_dict() if self.origin else None,
            "metadata": self.metadata,
        }

    def save(self):
        """Save the chunk into the directory"""
        if self._store is None:
            raise ValueError("Must provide `store` to save the chunk")
        self._store.save(self)


class ChunkGroup:
    """An interface for a group of related chunk"""

    def __init__(self, chunks=[], path=None):
        self._chunks = chunks or []
        self._roots: None | list = None  # idx to self._chunks
        self._path = path
        self._store: "BaseStore | None" = None

    def __bool__(self):
        return bool(self._chunks)

    def __getitem__(self, idx):
        return self._chunks[idx]

    def __iter__(self):
        return iter(self._chunks)

    def __len__(self):
        return len(self._chunks)

    def save(self, path):
        """Save all objects to a directory"""
        self._path = path
        for obj in self._chunks:
            obj.save(path)

    def roots(self):
        """Get chunks that have no parent"""
        if self._roots is None:
            self._roots, result = [], []
            for idx, chunk in enumerate(self._chunks):
                if not chunk.parent:
                    result.append(chunk)
                    self._roots.append(idx)
            return result

        return [self._chunks[idx] for idx in self._roots]

    def non_roots(self):
        """Get chunks that have a parent"""
        return [chunk for chunk in self._chunks if chunk._parent]

    def leafs(self):
        """Get chunks that have no children"""
        return [chunk for chunk in self._chunks if not chunk._children]

    def attach_store(self, store):
        self._store = store
        for chunk in self._chunks:
            chunk.store = store

    @property
    def store(self):
        return self._store


class BaseStore:
    """Base class for organizing and persisting chunk"""

    def __contains__(self, id: str) -> bool:
        """Check if the chunk exists in the store"""
        raise NotImplementedError

    def get(self, id: str) -> Chunk:
        """Get the chunk by id"""
        raise NotImplementedError

    def fetch_content(self, chunk: Chunk):
        """Fetch the content of the chunk"""
        raise NotImplementedError

    def save(self, chunk: Chunk):
        """Save the chunk to the store"""
        raise NotImplementedError

    def save_group(self, group: ChunkGroup):
        """Save the group to the store"""
        for chunk in group:
            self.save(chunk)

    def delete(self, chunk: Chunk):
        """Delete the chunk from the store"""
        raise NotImplementedError


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

    supported_mimetypes: list[str] = []
    _tool_desc: dict | None = None

    def __init__(self, *args, **kwargs):
        self._default_params = kwargs

    @staticmethod
    def run(chunk: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        raise NotImplementedError

    def __call__(self, *chunk: Chunk, **kwargs) -> ChunkGroup:
        if self._default_params:
            for key, value in self._default_params.items():
                kwargs.setdefault(key, value)
        return self.run(*chunk, **kwargs)

    @classmethod
    def as_tool(cls) -> dict:
        """Return the necessary parameters for the operation.

        If not subclassed, this method will inspect the `.run` method's signature.
        Any non-optional argument will be considered as a required parameter. Any
        non-Python built-in type will be ignored.

        The resulting dictionary will have the following keys:
            - name (str): name of the operation
            - description (str): description of the operation
            - params (dict): parameters for the operation, which will
        """
        if cls._tool_desc is not None:
            return cls._tool_desc

        signature = inspect.signature(cls.run)
        docstring = inspect.getdoc(cls.run) or ""

        # parse the description from the docstring from beginning to Args
        description = ""
        if docstring:
            parts = re.split(r"\n\s*Args:", docstring, 1)
            description = parts[0].strip()
            description = " ".join([line.strip() for line in description.split("\n")])

        # parse parameter descriptions from docstring
        param_descriptions = {}
        if len(docstring.split("Args:")) > 1:
            args_section = docstring.split("Args:")[1]
            param_pattern = re.compile(
                r"\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*?)(?=\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)",  # noqa: E501
                re.DOTALL,
            )
            matches = param_pattern.findall(args_section)

            for param_name, param_desc in matches:
                clean_desc = re.sub(r"\s+", " ", param_desc.strip())
                param_descriptions[param_name] = clean_desc

        # get type hints
        type_hints = get_type_hints(cls.run)

        # build parameters dictionary
        parameters = {}
        for name, param in signature.parameters.items():
            # skip *args, **kwargs, chunk param, and parameters without type annotations
            if name == "kwargs" or name == "chunk" or name not in type_hints:
                continue

            # skip non-builtin types
            type_anno = type_hints.get(name)
            type_name = getattr(type_anno, "__name__", str(type_anno))
            if not hasattr(builtins, type_name):
                continue

            param_info = {
                "type": type_name,
                "required": param.default is param.empty,
            }

            # add default value if available
            if param.default is not param.empty:
                param_info["default"] = param.default

            # add description if available
            if name in param_descriptions:
                param_info["description"] = param_descriptions[name]

            parameters[name] = param_info

        return {
            "name": cls.__qualname__,
            "description": description,
            "parameters": parameters,
        }

    @classmethod
    def py_dependency(cls) -> list[str]:
        """Return the Python dependencies"""
        return []

    def default(self, **kwargs):
        """Update the default parameters for the operation"""
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
