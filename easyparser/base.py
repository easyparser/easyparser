import builtins
import inspect
import logging
import re
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Literal, get_type_hints

logger = logging.getLogger(__name__)


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

    def asdict(self):
        return {
            "source_id": self.source_id,
            "location": self.location,
            "metadata": self.metadata,
        }


class CType:
    """Collection of chunk types and its utility"""

    # Chunk with this type will be interpreted as the same level with the parent chunk
    # (e.g. long text)
    Inline = "inline"

    # Chunk with this type will be interpreted as the child of the parent chunk
    Para = "para"
    List = "list"
    Table = "table"
    Header = "header"
    Figure = "figure"
    Code = "code"

    __available_types = None

    @classmethod
    def available_types(cls) -> list:
        if cls.__available_types is None:
            cls.__available_types = [
                "inline",
                "para",
                "list",
                "table",
                "header",
                "figure",
                "code",
            ]

        return cls.__available_types

    @classmethod
    def markdown(cls, chunk) -> str | None:
        """Represent chunk and its children as markdown text

        Args:
            chunk: the chunk to be represented as markdown text

        Returns:
            str: the markdown text
        """
        # If the chunk already has a text, return it
        if chunk.text:
            return chunk.text

        # Otherwise, reconstruct the text from the children
        text: str = chunk.content if isinstance(chunk.content, str) else ""
        child = chunk.child
        while child:
            if child.ctype == CType.Header:
                text += f"\n\n{'#' * (child.origin.location['level'] + 1)} {child.text}"
            elif child.ctype == CType.Table:
                text += f"\n\n{child.text}"
            elif child.ctype == CType.List:
                text += f"\n\n- {child.text}"
            else:
                text += f"\n\n{child.text}"

            child = child.next

        return text


class Chunk:
    """Mandatory fields for an object represented in `easyparser`.

    !IMPORTANT: all fields, except `content`, must be serializable to JSON.

    Args:
        id: unique identifier for the object.
        mimetype: mimetype of the object, plays a crucial role in determining how an
            object is processed and rendered. The official list of mimetypes:
            https://www.iana.org/assignments/media-types/media-types.xhtml
        ctype: the chunk type of object, 1 of CType enum.
        content: content of the object, can be anything (text or bytes), that can be
            understood from the mimetype.
        text: text representation of the object.
        summary: text summary of the object, used as short description in case the
            content is large.
        parent: parent object id. Default to None.
        child: the first child id. Default to None.
        next: next object id. Default to None.
        prev: previous object id. Default to None.
        origin: the location of this object in relative to the parent.
        metadata: metadata of the object, a free-style dictionary.
    """

    ctype_class = CType

    def __init__(
        self,
        mimetype: str,
        ctype: CType | str = CType.Inline,
        content: Any = None,
        text: str = "",
        summary: str = "",
        parent: "None | str | Chunk" = None,
        child: "None | list | Chunk" = None,
        next: "None | str | Chunk" = None,
        prev: "None | str | Chunk" = None,
        origin: None | Origin = None,
        metadata: None | dict = None,
        history: None | list = None,
    ):
        self.id: str = uuid.uuid4().hex
        self.mimetype = mimetype
        self.ctype = ctype
        self._content = content
        self.text = text
        self.summary = summary
        self._parent = parent
        self._child = child
        self._next = next
        self._prev = prev
        self.origin = origin
        self.metadata = metadata

        # internal use
        self._history: list = history or []
        self._store: "BaseStore | None" = None

    def __str__(self):
        if isinstance(self.content, str):
            text = self.content
            if len(text) > 80:
                text = f"{text[:50]}... ({len(text[50:].split())} more words)"
            text = text.replace("\n", " ")
            content = f"content={text}"
        else:
            content = f"mimetype={self.mimetype}"

        return self.__class__.__name__ + f"(ctype={self._ctype}, {content})"

    def __repr__(self):
        return self.__str__()

    @property
    def ctype(self):
        """Get the chunk type of the object"""
        return self._ctype

    @ctype.setter
    def ctype(self, value):
        """Set the chunk type of the object"""
        self._ctype = value

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
    def history(self) -> list:
        return self._history

    @history.setter
    def history(self, value: list):
        self._history = value

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

    @parent.setter
    def parent(self, value):
        if value is None or isinstance(value, (Chunk, str)):
            self._parent = value
        else:
            raise ValueError("`.parent` must be a Chunk or a id of a chunk")

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
        if value is None or isinstance(value, (Chunk, str)):
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
        if value is None or isinstance(value, (Chunk, str)):
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
    def child(self) -> "Chunk | None":
        """Get the child object"""
        if isinstance(self._child, Chunk):
            return self._child
        if isinstance(self._child, str):
            if not self._store:
                raise ValueError("Must provide `store` to load the child")
            self._child = self._store.get(self._child)
            return self._child
        if self._child is not None:
            raise ValueError("`.child` must be a Chunk or a id of a chunk")

    @child.setter
    def child(self, value):
        if value is None or isinstance(value, (Chunk, str)):
            self._child = value
        else:
            raise ValueError("`.child` must be a Chunk or a id of a chunk")

    @property
    def last_child(self) -> "Chunk | None":
        """Get the last child object"""
        child = self.child
        while child and child.next:
            child = child.next
        return child

    @property
    def child_id(self) -> str | None:
        if isinstance(self._child, str):
            return self._child
        if isinstance(self._child, Chunk):
            return self._child.id

    def render(
        self,
        format: Literal["plain", "markdown", "2d", "html"] = "plain",
        **kwargs,
    ) -> str:
        """Select the executor type to render the object

        Args:
            format: the format of the output. Defaults to "text".
        """
        if format == "plain":
            current = self.text

            if current:
                return current

            current = self.content if isinstance(self.content, str) else ""
            child = self.child
            while child:
                rendered_child = child.render(format=format, **kwargs).strip()
                if not rendered_child:
                    child = child.next
                    continue

                if child.ctype == "inline":
                    separator = " "
                elif child.ctype == "list":
                    separator = "\n"
                else:
                    separator = "\n\n"

                current += separator + rendered_child
                child = child.next
        elif format == "markdown":
            import textwrap

            if not kwargs:
                # header_level, list_level
                parent = self.parent
                kwargs = {"header_level": 0}
                while parent:
                    if parent.ctype == "header":
                        kwargs["header_level"] += 1
                    parent = parent.parent

            # Keep track of header to correctly render the header tag
            if self.ctype == "header":
                kwargs["header_level"] += 1

            if self.text:
                # It means the chunk is pre-rendered, and don't need to render again
                current = self.text
            else:
                current = self.content if isinstance(self.content, str) else ""
                if (
                    self.ctype == "header"
                    and kwargs.get("header_level", 0) > 0
                    and not current.startswith("#")
                ):
                    current = f"{'#' * kwargs['header_level']} {current}"
                child = self.child
                while child:
                    # Don't strip left whitespace because of list indentation
                    rendered_child = (
                        child.render(format=format, **kwargs).rstrip().lstrip("\n")
                    )
                    rendered_child = textwrap.dedent(rendered_child)
                    if not rendered_child:
                        child = child.next
                        continue
                    if child.ctype == "inline":
                        separator = " "
                    elif child.ctype == "list":
                        rendered_child = textwrap.indent(rendered_child, "  ")
                        separator = "\n"
                    else:
                        separator = "\n\n"

                    current += separator + rendered_child
                    child = child.next
        else:
            raise NotImplementedError(
                f"Render as `format={format}` is not yet supported"
            )

        return current

    def asdict(self):
        return {
            "id": self.id,
            "mimetype": self.mimetype,
            "content": self.content,
            "text": self.text,
            "parent": self.parent_id,
            "child": self.child_id,
            "next": self.next_id,
            "prev": self.prev_id,
            "origin": self.origin.asdict() if self.origin else None,
            "metadata": self.metadata,
            "_history": self._history,
        }

    def save(self):
        """Save the chunk into the directory"""
        if self._store is None:
            raise ValueError("Must provide `store` to save the chunk")
        self._store.save(self)

    def merge(self, chunk: "Chunk"):
        """Merge the content, metadata, and child of other chunk to this chunk

        Args:
            chunk: the other chunk to merge with this chunk
        """
        if self.mimetype != chunk.mimetype:
            raise ValueError("Cannot merge chunk with different mimetype")

        # Add the content
        self.content += chunk.content
        self.text += chunk.text
        self.summary += chunk.summary

        # Add the metadata
        if self.metadata is None and chunk.metadata is not None:
            self.metadata = chunk.metadata
        elif self.metadata is not None and chunk.metadata is not None:
            for key, value in chunk.metadata.items():
                if key in self.metadata:
                    self.metadata[key] += value
                else:
                    self.metadata[key] = value

        # Child
        our_last_child = self.last_child
        their_child = chunk.child
        if our_last_child and their_child:
            our_last_child.next = their_child
            their_child.prev = our_last_child
            their_child.parent = self
        elif their_child:
            self.child = their_child
            their_child.parent = self

    def apply(self, fn: Callable[["Chunk", int], None], depth: int = 0):
        """Apply a function to the chunk and all its children"""
        fn(self, depth)
        child = self.child
        while child:
            child.apply(fn, depth=depth + 1)
            child = child.next

    def print_graph(self):
        def print_node(node, depth=0):
            print("    " * depth, node)

        self.apply(print_node)


class ChunkGroup:
    """An interface for a group of related chunk"""

    def __init__(self, chunks: list | None = None, root: Chunk | None = None):
        self._roots: dict[str, Chunk] = {}
        self._chunks: dict[str | None, list] = {}

        self._root_id = None
        if root is not None:
            self._roots[root.id] = root
            self._root_id = root.id

        if chunks is not None or self._root_id is not None:
            self._chunks[self._root_id] = chunks or []

        self._store: "BaseStore | None" = None

    @property
    def store(self):
        return self._store

    @property
    def groups(self):
        return self._chunks

    def __bool__(self):
        return bool(len(self))

    def __getitem__(self, idx):
        for chunks in self._chunks.values():
            if idx < len(chunks):
                return chunks[idx]
            idx -= len(chunks)

    def __iter__(self):
        for chunks in self._chunks.values():
            yield from chunks

    def __len__(self):
        count = sum(len(chunks) for chunks in self._chunks.values())
        return count

    def append(self, chunk: Chunk):
        if len(self._chunks) > 1:
            raise ValueError(
                "Cannot append when ChunkGroup has multiple roots. "
                "Please specify the root to append: "
                "`.groups[root_id_str].append(chunk)`"
            )
        elif len(self._chunks) == 0:
            self._chunks[self._root_id] = []

        if self._root_id not in self._chunks:
            self._root_id = list(self._chunks.keys())[0]

        self._chunks[self._root_id].append(chunk)

    def extend(self, chunks: list[Chunk]):
        if len(self._chunks) > 1:
            raise ValueError(
                "Cannot extend when ChunkGroup has multiple roots. "
                "Please specify the root to extend: "
                "`.groups[root_id_str].extend(chunks)`"
            )
        elif len(self._chunks) == 0:
            self._chunks[self._root_id] = []

        if self._root_id not in self._chunks:
            self._root_id = list(self._chunks.keys())[0]

        self._chunks[self._root_id].extend(chunks)

    def iter_groups(self):
        for root_id, chunks in self._chunks.items():
            if root_id is None:
                yield root_id, chunks
            else:
                root_node = self._roots[root_id]
                yield root_node, chunks

    def add_group(self, group: "ChunkGroup"):
        """Add another ChunkGroup to the current chunk group"""
        for root, chunks in group.iter_groups():
            if isinstance(root, Chunk):
                self._roots[root.id] = root
                root_id = root.id
            else:
                root_id = None

            if root_id not in self._chunks:
                self._chunks[root_id] = []
            self._chunks[root_id].extend(chunks)

    def attach_store(self, store):
        self._store = store
        for chunks in self._chunks.values():
            for chunk in chunks:
                chunk.store = store


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
        for root, chunks in group.iter_groups():
            if isinstance(root, Chunk):
                self.save(root)
            for chunk in chunks:
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

    @classmethod
    def run(cls, chunk: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        raise NotImplementedError

    @classmethod
    def name(cls, **kwargs) -> str:
        """Return the name of the operation to keep track in history"""
        fn = cls.__name__
        return f"{fn}({', '.join([f'{k}={v}' for k, v in kwargs.items()])})"

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
